"""LendingClub 评分卡全流程：数据 → WOE/Logistic → OOT → PSI/Vintage → 阈值 → LightGBM → 报告。"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import statsmodels

from . import plots
from .binning import binning_details, fit_all_binnings, woe_frame
from .calibration import (
    ALL_SEGMENT,
    IsotonicCalibrator,
    PlattCalibrator,
    apply_lifetime_conversion,
    calibrate_frame,
    emergence_by_term_structure,
    grade_lifetime_calibration,
    risk_layer_emergence,
    risk_layer_within_term,
    term_emergence,
    threshold_sensitivity,
    threshold_stability,
    to_lifetime_pd,
)
from .config import (
    CN_NAMES,
    FEATURE_NAMES,
    ID_COLUMN,
    TARGET,
    TARGET_VINTAGE,
    PipelineConfig,
)
from .data import (
    Preprocessor,
    data_quality_summary,
    emergence_curve,
    feature_label,
    load_base,
    time_split,
    vintage_tables,
)
from .evaluation import (
    approval_at_rate,
    binary_metrics,
    bootstrap_auc,
    calibration_metrics,
    calibration_table,
    choose_threshold,
    grade_table,
    lift_table,
    psi,
    psi_level,
    psi_table,
    score_psi_by_period,
    swap_set_analysis,
    threshold_confusion,
    threshold_scan,
)
from .policy import DecisionPolicy
from .model import (
    build_sensitivity_frame,
    coefficient_table,
    fit_lightgbm,
    lgbm_importance,
    model_formula,
    select_and_fit,
    sensitivity_ablation,
    vif_table,
)
from .monitoring import build_monitoring_plan
from .scorecard import build_scorecard

LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
    LOGGER.setLevel(level)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"无法序列化: {type(value)}")


def _save_table(df: pd.DataFrame | None, path: Path, index: bool = False) -> None:
    if df is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")


def _metrics_frame(metrics: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for dataset, values in metrics.items():
        rows.append({"数据集": dataset, **values})
    return pd.DataFrame(rows)


def _threshold_summary(
    chosen,
    conf_oot: dict,
    lgbm_chosen,
    lgbm_conf_oot: dict,
    lgbm_same_rate: pd.Series,
) -> pd.DataFrame:
    rows = [
        {
            "模型": "WOE 评分卡",
            "阈值来源": "验证集 2016 目标坏账率",
            "阈值": chosen["threshold"],
            "批准率": conf_oot["approval_rate"],
            "批准坏账率": conf_oot["approved_bad_rate"],
            "坏客户捕获率": conf_oot["bad_capture"],
            "审阅": "验证集选中；跨期复核",
        },
        {
            "模型": "LightGBM",
            "阈值来源": "验证集 2016 目标坏账率",
            "阈值": lgbm_chosen["threshold"],
            "批准率": lgbm_conf_oot["approval_rate"],
            "批准坏账率": lgbm_conf_oot["approved_bad_rate"],
            "坏客户捕获率": lgbm_conf_oot["bad_capture"],
            "审阅": "模型对照",
        },
        {
            "模型": "LightGBM",
            "阈值来源": "对齐评分卡 OOT 批准率",
            "阈值": lgbm_same_rate["threshold"],
            "批准率": lgbm_same_rate["approval_rate"],
            "批准坏账率": lgbm_same_rate["approved_bad_rate"],
            "坏客户捕获率": lgbm_same_rate["bad_capture"],
            "审阅": "业务同批准率对比",
        },
    ]
    return pd.DataFrame(rows)


def run_pipeline(cfg: PipelineConfig | None = None, refresh_data: bool = False) -> dict:
    cfg = cfg or PipelineConfig()
    configure_logging()
    cfg.ensure_dirs()
    started = time.perf_counter()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = Path(cfg.output_dir)

    LOGGER.info("=" * 78)
    LOGGER.info("LendingClub 信用风险评分卡 / OOT 监控项目")
    LOGGER.info("数据: %s", Path(cfg.data_dir) / cfg.raw_file)
    LOGGER.info("输出: %s", out)
    LOGGER.info("=" * 78)

    # ------------------------------------------------------------------ 1. 数据
    base_all = load_base(cfg, refresh=refresh_data)
    dq = data_quality_summary(base_all, cfg.performance_mob)
    vintage_comp, vintage_resolved = vintage_tables(base_all)
    emergence_all = emergence_curve(base_all, cfg)
    _save_table(emergence_all, out / "tables" / "emergence_curve.csv")
    train, valid, oot = time_split(base_all, cfg)

    # 建模标签使用 MOB 对齐的 vintage 口径；终身口径 bad_flag 仅用于对比展示。
    model_target = TARGET_VINTAGE
    LOGGER.info(
        "建模标签口径: %s（MOB%d vintage）；对照口径: %s（终身，已终态样本）",
        model_target,
        cfg.performance_mob,
        TARGET,
    )

    # ------------------------------------------------------------------ 2. 预处理
    pre = Preprocessor(cfg).fit(train)
    X_train = pre.transform(train)
    X_valid = pre.transform(valid)
    X_oot = pre.transform(oot)
    y_train = train[model_target].astype(int)
    y_valid = valid[model_target].astype(int)
    y_oot = oot[model_target].astype(int)
    pre.save(out / "models" / "preprocessor.joblib")
    LOGGER.info(
        "预处理完成: train=%s, valid=%s, oot=%s, 特征=%s",
        X_train.shape,
        X_valid.shape,
        X_oot.shape,
        len(pre.feature_names_),
    )

    # ------------------------------------------------------------------ 3. 分箱/IV
    binnings, iv_table = fit_all_binnings(X_train, y_train, cfg)
    iv_table["selected_initial"] = iv_table["iv"] >= cfg.iv_threshold
    _save_table(iv_table, out / "tables" / "iv.csv")
    _save_table(binning_details(binnings, None), out / "tables" / "binning_details_all.csv")

    X_train_woe_all = woe_frame(X_train, binnings, iv_table["feature"].tolist())
    selection = select_and_fit(X_train_woe_all, y_train, iv_table, cfg)
    selected = selection.features
    _save_table(selection.coef_table, out / "tables" / "coefficients.csv")
    _save_table(binning_details(binnings, selected), out / "tables" / "binning_details.csv")
    _save_table(selection.log, out / "tables" / "selection_log.csv")

    # VIF 双口径：剔除前（初始入模集）与剔除后（最终入模集）。
    # 旧版只输出剔除后的 VIF，被剔除变量的共线性信息全部丢失，看到的"干净 VIF"
    # 只是幸存者的 VIF。两张表对照才能回答"当初为什么必须剔除某个变量"。
    vif_initial = vif_table(X_train_woe_all[selection.initial_features])
    vif_initial.insert(0, "stage", "剔除前（初始入模集）")
    vif_initial["dropped_later"] = ~vif_initial["feature"].isin(selected)
    _save_table(vif_initial, out / "tables" / "vif_initial.csv")

    vif_final = vif_table(woe_frame(X_train, binnings, selected))
    vif_final.insert(0, "stage", "剔除后（最终入模集）")
    _save_table(vif_final, out / "tables" / "vif.csv")
    LOGGER.info(
        "VIF: 剔除前最大 %.2f（%s 个变量，其中 %s 个后被剔除）-> 剔除后最大 %.2f（%s 个变量）",
        float(vif_initial["VIF"].max()),
        len(selection.initial_features),
        int(vif_initial["dropped_later"].sum()),
        float(vif_final["VIF"].max()),
        len(selected),
    )
    LOGGER.info("入模变量: %s", ", ".join(selected))

    X_train_woe = X_train_woe_all[selected]
    X_valid_woe = woe_frame(X_valid, binnings, selected)
    X_oot_woe = woe_frame(X_oot, binnings, selected)
    p_train = selection.model.predict(X_train_woe)
    p_valid = selection.model.predict(X_valid_woe)
    p_oot = selection.model.predict(X_oot_woe)

    # ------------------------------------------------------------------ 4. 指标（vintage 口径）
    # 以下 metrics 是 MOB12 vintage 口径下的排序能力指标。注意 AUC/KS 是排序指标，
    # 对单调变换不变，但**按期限分段的终身换算不是全局单调的**，会改变跨期限的相对
    # 次序，因此后面还会单独报告终身口径的 AUC，两者都保留才能看清期限结构的影响。
    metrics = {
        "train": binary_metrics(y_train, p_train),
        "valid": binary_metrics(y_valid, p_valid),
        "oot": binary_metrics(y_oot, p_oot),
    }
    bootstrap = bootstrap_auc(y_oot, p_oot, n_boot=cfg.bootstrap_rounds, seed=cfg.random_state)
    calibration = {
        "scorecard": calibration_table(y_valid, p_valid),
        "scorecard_oot": calibration_table(y_oot, p_oot),
    }
    _save_table(calibration["scorecard_oot"], out / "tables" / "calibration_oot.csv")
    _save_table(lift_table(y_oot, p_oot), out / "tables" / "lift_scorecard_oot.csv")
    _save_table(_metrics_frame(metrics), out / "tables" / "metrics_scorecard.csv")

    # ------------------------------------------------------------------ 4b. 校准 + 终身换算
    # 两个独立的问题在这里一起解决：
    #   (a) 模型 PD 未校准（旧版 OOT 校准斜率 0.819、截距 0.036，系统性偏低）；
    #   (b) MOB12 vintage 口径的绝对值只有终身坏账率的约 30%，业务无法直接使用。
    # 顺序必须是"先校准到 vintage 口径，再换算到终身口径"，反过来会把口径差异
    # 错误地当成校准误差吸收掉。
    #
    # 校准器只用验证集 fit，OOT 只 transform，不引入未来信息。
    platt = PlattCalibrator().fit(p_valid, y_valid)
    isotonic = IsotonicCalibrator().fit(p_valid, y_valid)
    calibrator_comparison = calibrate_frame(p_oot, y_oot, platt, isotonic)
    _save_table(calibrator_comparison, out / "tables" / "calibrator_comparison_oot.csv")
    joblib.dump({"platt": platt, "isotonic": isotonic}, out / "models" / "calibrators.joblib")

    p_valid_cal = platt.transform(p_valid)
    p_oot_cal = platt.transform(p_oot)

    # 暴露比例：按贷款期限分段（36 期 e≈0.346，60 期 e≈0.257，相差 35%）
    term_e = term_emergence(base_all, cfg)
    emergence_grade = risk_layer_emergence(base_all, cfg, by="grade")
    emergence_within = risk_layer_within_term(base_all, cfg, by="grade")
    emergence_year_struct = emergence_by_term_structure(base_all, term_e)
    _save_table(term_e, out / "tables" / "emergence_by_term.csv")
    _save_table(emergence_grade, out / "tables" / "emergence_by_grade.csv")
    _save_table(emergence_within, out / "tables" / "emergence_grade_within_term.csv")
    _save_table(emergence_year_struct, out / "tables" / "emergence_by_year_structure.csv")

    # 逐笔终身换算（必须按每笔贷款的期限查 e，用总体 e 会错估长期限贷款）
    valid_lt = apply_lifetime_conversion(
        pd.DataFrame(
            {
                "prob_model": p_valid,
                "prob_calibrated": p_valid_cal,
                "term_num": valid["term_num"].to_numpy(),
            }
        ),
        term_e,
    )
    oot_lt = apply_lifetime_conversion(
        pd.DataFrame(
            {
                "prob_model": p_oot,
                "prob_calibrated": p_oot_cal,
                "term_num": oot["term_num"].to_numpy(),
            }
        ),
        term_e,
    )
    p_valid_lt = valid_lt["prob_lifetime"].to_numpy()
    p_oot_lt = oot_lt["prob_lifetime"].to_numpy()

    # 终身口径的排序能力与校准质量（用于与 vintage 口径对比）
    metrics_lifetime = {
        "valid": binary_metrics(y_valid, p_valid_lt),
        "oot": binary_metrics(y_oot, p_oot_lt),
    }
    _save_table(
        _metrics_frame(metrics_lifetime), out / "tables" / "metrics_scorecard_lifetime.csv"
    )
    # 校准质量。注意这里只做评估，绝不在 OOT 上重新拟合校准器——校准器一律只在
    # 验证集 fit，OOT 只 transform/评估，否则构成数据泄漏。
    #
    # 口径必须配对：calibration_metrics 的 actual 取自标签，而标签是 vintage 口径，
    # 所以只有 vintage 尺度上的 PD 能直接用它评估。终身尺度的校验改用加权终身坏账率
    # E[y/e]：它与阈值扫描、捕获率使用同一个估计量；mean(y)/mean(e) 是比值估计，
    # 会因期限结构与坏账率相关而与阈值表不一致，故只保留 E[y/e]。
    observed_lt_rate = float(
        np.mean(y_oot.to_numpy(dtype=float) / oot_lt["emergence_frac"].to_numpy(dtype=float))
    )
    calibration_by_scale = pd.DataFrame(
        [
            {
                "pd_scale": "vintage_mob12",
                "label_scale": "vintage_mob12",
                **calibration_metrics(y_oot, p_oot),
            },
            {
                "pd_scale": "vintage_calibrated",
                "label_scale": "vintage_mob12",
                **calibration_metrics(y_oot, p_oot_cal),
            },
            {
                "pd_scale": "lifetime",
                "label_scale": "lifetime（E[y/e] 加权估计）",
                "mean_pred": float(np.mean(p_oot_lt)),
                "actual": observed_lt_rate,
                "scale_gap": observed_lt_rate - float(np.mean(p_oot_lt)),
                "ratio": observed_lt_rate / float(np.mean(p_oot_lt))
                if float(np.mean(p_oot_lt))
                else np.nan,
                "note": "终身尺度只比较均值水平；分段精度见 grade_calibration_lifetime_oot.csv",
            },
        ]
    )
    _save_table(calibration_by_scale, out / "tables" / "calibration_by_scale.csv")
    pd_scale_comparison = pd.DataFrame(
        [
            {
                "口径": "MOB12 vintage（建模标签）",
                "OOT平均PD": float(np.mean(p_oot_cal)),
                "OOT实测坏账率": float(np.mean(y_oot)),
                "说明": "各 cohort 统一截到 MOB12，跨期可比，但绝对值偏低",
            },
            {
                "口径": "终身（业务决策口径）",
                "OOT平均PD": float(np.mean(p_oot_lt)),
                "OOT实测坏账率": observed_lt_rate,
                "说明": "按期限分段换算，E[y/e] 加权估计，可直接与业务坏账率目标对比",
            },
        ]
    )
    _save_table(pd_scale_comparison, out / "tables" / "pd_scale_comparison.csv")
    LOGGER.info(
        "口径对比: vintage 平均PD %.4f / 实测 %.4f；终身 平均PD %.4f；OOT AUC vintage %.4f -> 终身 %.4f",
        float(np.mean(p_oot_cal)),
        float(np.mean(y_oot)),
        float(np.mean(p_oot_lt)),
        metrics["oot"]["auc"],
        metrics_lifetime["oot"]["auc"],
    )

    # ------------------------------------------------------------------ 5. 评分卡
    card = build_scorecard(selection.model, binnings, cfg)
    card.save(out / "scorecard.json", out / "scorecard.csv")
    scored_oot = card.score_frame(X_oot)
    scored_oot.insert(0, ID_COLUMN, oot[ID_COLUMN].to_numpy())
    scored_oot["prob_model"] = p_oot
    # 三个口径并列保存，报告中必须明确标注用的是哪一个：
    #   prob_model       未校准的原始 vintage PD（旧版口径，系统性偏低）
    #   prob_calibrated  验证集校准后的 vintage PD（MOB12 尺度，跨期可比）
    #   prob_lifetime    按期限换算的终身 PD（业务决策尺度，可与坏账率目标直接对比）
    scored_oot["prob_calibrated"] = p_oot_cal
    scored_oot["prob_lifetime"] = p_oot_lt
    scored_oot["emergence_frac"] = oot_lt["emergence_frac"].to_numpy()
    scored_oot["lifetime_clipped"] = oot_lt["lifetime_clipped"].to_numpy()
    scored_oot["term_num"] = oot["term_num"].to_numpy()
    scored_oot["grade"] = card.grade_of(scored_oot["score"])
    scored_oot["target"] = y_oot.to_numpy()
    scored_oot["target_lifetime"] = y_oot.to_numpy() / scored_oot["emergence_frac"].to_numpy()
    scored_oot["issue_quarter"] = oot["issue_quarter"].to_numpy()
    scored_oot["issue_year"] = oot["issue_year"].to_numpy()
    _save_table(scored_oot, out / "predictions" / "oot_scores.csv")
    grade_tbl = grade_table(y_oot, scored_oot["score"], scored_oot["grade"])
    _save_table(grade_tbl, out / "tables" / "grade_stats_oot.csv")

    # 旧版口径的分段校准表：拿 MOB 不对齐的实测坏账率与未校准 PD 相比。
    # 保留它是为了展示"1.4x 低估倍数"其实是口径差异造成的假象。
    grade_cal = (
        scored_oot.groupby("grade", observed=True)
        .agg(
            count=("target", "size"),
            avg_pred_pd=("prob_model", "mean"),
            actual_bad_rate=("target", "mean"),
            avg_score=("score", "mean"),
        )
        .reset_index()
    )
    grade_cal["legacy_note"] = (
        "vintage 标签 vs 未校准 PD；旧版误读为模型低估，实为口径差异"
    )
    _save_table(grade_cal, out / "tables" / "grade_calibration_oot.csv")

    # 新口径的分段校准表：预测与实测统一到终身尺度，ratio 才是真实校准误差。
    grade_cal_lt = grade_lifetime_calibration(
        scored_oot, term_e, by="grade", pd_col="prob_calibrated"
    )
    _save_table(grade_cal_lt, out / "tables" / "grade_calibration_lifetime_oot.csv")
    LOGGER.info(
        "分段校准（终身口径）ratio: %s",
        {
            str(r.grade): round(r.ratio_lifetime, 3)
            for r in grade_cal_lt.itertuples(index=False)
        }
        if len(grade_cal_lt)
        else "N/A",
    )

    # ------------------------------------------------------------------ 6. PSI / Vintage
    train_scored = card.score_frame(X_train)
    psi_valid = psi(train_scored["score"], card.score_frame(X_valid)["score"])
    psi_oot = psi(train_scored["score"], scored_oot["score"])
    LOGGER.info("评分 PSI: valid=%.4f (%s), oot=%.4f (%s)", psi_valid, psi_level(psi_valid), psi_oot, psi_level(psi_oot))

    psi_features = psi_table(X_train, X_oot, selected, n_bins=10)
    _save_table(psi_features, out / "tables" / "psi_features_oot.csv")

    # 全量申请样本评分（用于季度 PSI 与 Vintage）。
    #
    # **PSI 的样本口径修正**：PSI 衡量的是申请人群评分分布随时间的漂移，与标签是否
    # 成熟无关，因此必须用**全量申请人群**。旧版用 `resolved` 筛选，导致 2018Q4 只剩
    # 5,018 条终态样本，PSI 被小样本噪声主导（算出 0.0958，逼近黄色预警线），而这与
    # 真实的客群漂移无关。改用全量后 2018Q4 有 12.8 万条，PSI 才有解释力。
    all_pop = base_all.loc[base_all["issue_year"] >= 2013].copy()
    # 按年分块评分，避免一次性构造 200 万+ x 90 特征矩阵导致内存峰值过高
    scored_parts = []
    for year, group in all_pop.groupby("issue_year", observed=True):
        part = card.score_frame(pre.transform(group))
        part["issue_quarter"] = group["issue_quarter"].to_numpy()
        part["issue_year"] = year
        part["term_num"] = group["term_num"].to_numpy()
        # 两种标签口径并列：终身口径（已终态样本才有值）与 MOB12 vintage 口径
        part["target_legacy"] = group[TARGET].to_numpy()
        part["target_vintage"] = group[TARGET_VINTAGE].to_numpy()
        part["resolved"] = group["resolved"].to_numpy()
        part["evaluable"] = group["evaluable"].to_numpy()
        scored_parts.append(
            part[
                [
                    "score",
                    "prob_from_score",
                    "issue_quarter",
                    "issue_year",
                    "term_num",
                    "target_legacy",
                    "target_vintage",
                    "resolved",
                    "evaluable",
                ]
            ]
        )
    all_scored = pd.concat(scored_parts, ignore_index=True)
    all_scored["score"] = all_scored["score"].astype(int)
    psi_quarter = score_psi_by_period(train_scored["score"], all_scored)
    # 标注每个季度的样本量：PSI 在小样本期不可靠，必须让读者看到 n
    psi_n = all_scored.groupby("issue_quarter", observed=True).size().rename("n_pop")
    psi_quarter = psi_quarter.merge(psi_n, left_on="period", right_index=True, how="left")
    psi_quarter["psi_reliable"] = psi_quarter["n_pop"] >= cfg.psi_min_period_n
    _save_table(psi_quarter, out / "tables" / "psi_quarter.csv")

    # Vintage 表：同时给出三种口径的坏账率，这是本项目最核心的一张对照表。
    #   legacy    旧口径，只统计已终态样本（随成熟度非单调偏差）
    #   vintage   MOB12 对齐口径（跨期可比，但绝对值偏低）
    #   lifetime  vintage 按期限换算到终身（业务决策口径）
    e_map = dict(zip(term_e["segment"].astype(str), term_e["emergence_frac"]))
    e_all = float(e_map.get(ALL_SEGMENT, np.nan))
    all_scored["_e"] = (
        all_scored["term_num"].astype(float).round().astype("Int64").astype(str).map(e_map)
    ).astype(float).fillna(e_all)

    vintage_rows = []
    for q, g in all_scored.groupby("issue_quarter", observed=True):
        ev = g.loc[g["evaluable"]]
        res = g.loc[g["resolved"]]
        v_rate = float(ev["target_vintage"].mean()) if len(ev) else np.nan
        e_mean = float(ev["_e"].mean()) if len(ev) else np.nan
        vintage_rows.append(
            {
                "issue_quarter": q,
                "n_pop": len(g),
                "avg_score": float(g["score"].mean()),
                "avg_pd": float(g["prob_from_score"].mean()),
                "n_resolved": len(res),
                "legacy_bad_rate": float(res["target_legacy"].mean()) if len(res) else np.nan,
                "n_evaluable_mob12": len(ev),
                "vintage_mob12_bad_rate": v_rate,
                "mean_emergence_frac": e_mean,
                # 终身口径：逐笔加权后求和 / 样本数，等价于按期限结构加权换算
                "lifetime_bad_rate": float((ev["target_vintage"] / ev["_e"]).mean())
                if len(ev)
                else np.nan,
            }
        )
    vintage_scores = pd.DataFrame(vintage_rows)
    vintage_table = vintage_comp.merge(
        vintage_scores, on="issue_quarter", how="left", suffixes=("", "_new")
    )
    _save_table(vintage_table, out / "tables" / "vintage.csv")
    _save_table(vintage_scores, out / "tables" / "vintage_bad_rate_by_scale.csv")
    LOGGER.info(
        "Vintage 三口径坏账率已输出（legacy / MOB12 vintage / lifetime），共 %s 个季度",
        len(vintage_scores),
    )

    # ------------------------------------------------------------------ 7. 阈值扫描
    # 阈值必须在**终身口径**上选择。原因：MOB12 vintage 口径的整体坏账率只有 6~7%，
    # 用它去卡"批准坏账率 ≤ 15%"的约束，几乎所有阈值都自动达标，约束形同虚设，
    # 选出来的阈值没有业务意义。终身口径才能直接对应"这批贷款最终会损失多少"。
    #
    # 终身口径标签用逆暴露比例加权：y_lt = y_vintage / e(term)。因为
    #   E[y_vintage / e] = e·p_lifetime / e = p_lifetime
    # 该加权标签是终身坏账率的无偏估计，可直接参与阈值扫描与坏账率约束判断。
    y_valid_lt = y_valid.to_numpy(dtype=float) / valid_lt["emergence_frac"].to_numpy()
    y_oot_lt = y_oot.to_numpy(dtype=float) / oot_lt["emergence_frac"].to_numpy()
    label_scale_check = pd.DataFrame(
        [
            {
                "dataset": "valid_2016",
                "vintage_bad_rate": float(y_valid.mean()),
                "lifetime_bad_rate_weighted": float(y_valid_lt.mean()),
                "mean_e": float(valid_lt["emergence_frac"].mean()),
            },
            {
                "dataset": "oot",
                "vintage_bad_rate": float(y_oot.mean()),
                "lifetime_bad_rate_weighted": float(y_oot_lt.mean()),
                "mean_e": float(oot_lt["emergence_frac"].mean()),
            },
        ]
    )
    _save_table(label_scale_check, out / "tables" / "label_scale_check.csv")

    # 终身口径网格需覆盖更宽的 PD 范围（vintage 口径 0.02~0.60 不够）
    grid_lt = np.arange(0.02, 0.901, 0.01)
    scan_valid_lt = threshold_scan(y_valid_lt, p_valid_lt, thresholds=grid_lt)
    scan_oot_lt = threshold_scan(y_oot_lt, p_oot_lt, thresholds=grid_lt)
    chosen = choose_threshold(
        scan_valid_lt, cfg.threshold_target_bad_rate, cfg.threshold_min_approval
    )
    conf_oot = threshold_confusion(y_oot_lt, p_oot_lt, float(chosen["threshold"]))
    oot_diagnostic = choose_threshold(
        scan_oot_lt, cfg.threshold_target_bad_rate, cfg.threshold_min_approval
    )
    _save_table(scan_valid_lt, out / "tables" / "threshold_scan_valid.csv")
    _save_table(scan_oot_lt, out / "tables" / "threshold_scan_oot.csv")
    # vintage 口径扫描保留作对照，说明为什么不能用它选阈值
    _save_table(
        threshold_scan(y_valid, p_valid_cal),
        out / "tables" / "threshold_scan_valid_vintage.csv",
    )
    _save_table(
        pd.DataFrame(
            [
                {
                    **chosen.to_dict(),
                    "dataset": "valid",
                    "pd_scale": "lifetime",
                    "label_scale": "lifetime_weighted",
                }
            ]
        ),
        out / "tables" / "threshold_selected.csv",
    )

    # 打分落地策略：校准器、期限暴露比例、终身 PD 阈值、PD 等级带。
    # 只保存参数，不保存模型；score_csv.py 与 pipeline 共用同一份口径。
    policy = DecisionPolicy(
        platt_a=platt.a_,
        platt_b=platt.b_,
        emergence_by_term={
            str(r.segment): float(r.emergence_frac) for r in term_e.itertuples(index=False)
        },
        emergence_default=float(
            term_e.loc[term_e["segment"] == ALL_SEGMENT, "emergence_frac"].iloc[0]
        ),
        threshold_lifetime=float(chosen["threshold"]),
        grade_pd_cutoffs=tuple(cfg.grade_pd_cutoffs),
        target_bad_rate=cfg.threshold_target_bad_rate,
    )
    policy.save(out / "decision_policy.json")

    # 阈值敏感性：多个目标坏账率下的阈值与批准率，暴露选择规则对误差的敏感度
    sens = threshold_sensitivity(
        y_oot_lt,
        p_oot_lt,
        target_bad_rates=tuple(cfg.threshold_sensitivity_targets),
        min_approval=cfg.threshold_min_approval,
        grid=grid_lt,
    )
    _save_table(sens, out / "tables" / "threshold_sensitivity.csv")
    stab = threshold_stability(
        y_oot_lt,
        p_oot_lt,
        base_target=cfg.threshold_target_bad_rate,
        perturbation=cfg.threshold_sensitivity_step,
        min_approval=cfg.threshold_min_approval,
    )
    _save_table(stab, out / "tables" / "threshold_stability.csv")
    LOGGER.info(
        "阈值（终身口径）: 选中 PD=%.4f，OOT 批准率 %.4f，批准坏账率 %.4f（目标 ≤%.2f）",
        float(chosen["threshold"]),
        conf_oot["approval_rate"],
        conf_oot["approved_bad_rate"],
        cfg.threshold_target_bad_rate,
    )
    LOGGER.info(
        "阈值敏感性: 目标 %.0f%%/%.0f%%/%.0f%% -> 批准率 %s",
        100 * cfg.threshold_sensitivity_targets[0],
        100 * cfg.threshold_sensitivity_targets[1],
        100 * cfg.threshold_sensitivity_targets[2],
        [round(float(x), 4) for x in sens["approval_rate"].to_numpy()],
    )

    # ------------------------------------------------------------------ 8. LightGBM
    lgbm = fit_lightgbm(X_train, y_train, X_valid, y_valid, cfg)
    p_lgbm_train = lgbm.predict(X_train)
    p_lgbm_valid = lgbm.predict(X_valid)
    p_lgbm_oot = lgbm.predict(X_oot)
    lgbm_metrics = {
        "train": binary_metrics(y_train, p_lgbm_train),
        "valid": binary_metrics(y_valid, p_lgbm_valid),
        "oot": binary_metrics(y_oot, p_lgbm_oot),
    }
    importance = lgbm_importance(lgbm)
    _save_table(importance, out / "tables" / "lgbm_importance.csv")
    _save_table(_metrics_frame(lgbm_metrics), out / "tables" / "metrics_lgbm.csv")
    _save_table(calibration_table(y_oot, p_lgbm_oot), out / "tables" / "calibration_lgbm_oot.csv")
    _save_table(lift_table(y_oot, p_lgbm_oot), out / "tables" / "lift_lgbm_oot.csv")
    joblib.dump(lgbm, out / "models" / "lgbm.joblib", compress=3)

    # ------------------ 8b. 公平对照：拆解 AUC 增量的来源 ------------------
    # 旧版报告称"LightGBM 使用同一套特征，保证对照公平"，但实际是：
    #   评分卡  = 14 个筛选后变量（WOE + 逻辑回归）
    #   LightGBM = 82 个预处理后变量（树模型）
    # 两者同时在"变量数量"和"模型形式"两个维度上不同，因此 +0.043 AUC 无法归因。
    # 这里补第三个模型：LightGBM 只用评分卡最终的那 14 个变量，形成可归因的对照。
    #   评分卡(14变量, 线性)  vs  同变量 LightGBM(14变量, 树)  -> 模型形式贡献
    #   同变量 LightGBM       vs  LightGBM-full(82变量, 树)   -> 变量数量贡献
    # 用 WOE 编码后的 14 个变量喂给树模型，保证与评分卡输入信息完全一致。
    lgbm_12 = fit_lightgbm(
        X_train_woe, y_train, X_valid_woe, y_valid, cfg, features=selected
    )
    p_lgbm12_oot = lgbm_12.predict(X_oot_woe)
    p_lgbm12_valid = lgbm_12.predict(X_valid_woe)
    m_lgbm12 = {
        "valid": binary_metrics(y_valid, p_lgbm12_valid),
        "oot": binary_metrics(y_oot, p_lgbm12_oot),
    }
    joblib.dump(lgbm_12, out / "models" / "lgbm_same12.joblib", compress=3)

    auc_card = float(metrics["oot"]["auc"])
    auc_lgbm12 = float(m_lgbm12["oot"]["auc"])
    auc_lgbm_full = float(lgbm_metrics["oot"]["auc"])
    attribution = pd.DataFrame(
        [
            {
                "model": "WOE 评分卡（14 变量，线性）",
                "n_features": len(selected),
                "model_form": "WOE + Logistic",
                "oot_auc": auc_card,
                "oot_ks": metrics["oot"]["ks"],
                "role": "基线",
            },
            {
                "model": "LightGBM-同变量（14 变量，树）",
                "n_features": len(selected),
                "model_form": "Gradient Boosting Tree",
                "oot_auc": auc_lgbm12,
                "oot_ks": m_lgbm12["oot"]["ks"],
                "role": "隔离模型形式",
            },
            {
                "model": "LightGBM-full（82 变量，树）",
                "n_features": len(lgbm.features),
                "model_form": "Gradient Boosting Tree",
                "oot_auc": auc_lgbm_full,
                "oot_ks": lgbm_metrics["oot"]["ks"],
                "role": "隔离变量数量",
            },
        ]
    )
    attribution["auc_vs_previous"] = attribution["oot_auc"].diff()
    _save_table(attribution, out / "tables" / "model_attribution_oot.csv")
    LOGGER.info(
        "AUC 增量拆解: 总增量 %.4f = 模型形式 %.4f + 变量数量 %.4f",
        auc_lgbm_full - auc_card,
        auc_lgbm12 - auc_card,
        auc_lgbm_full - auc_lgbm12,
    )

    # ------------------ 8c. FICO / grade 敏感性对照（消融实验）------------------
    # 旧版报告称"新数据含 FICO，为保持与上一版特征口径一致，主模型不使用"，且
    # RAW_COLUMNS 里读了 grade 并注明"仅用于敏感性对照"——但两者都从未真的做对照。
    # 结果是 OOT AUC 0.6646 缺乏参照系：无法判断是方法问题还是特征集天花板问题。
    #
    # 这里用同一套树模型与切分，只改变是否加入 FICO / grade，量化边际贡献。
    # 基线直接复用上面的 LightGBM-full（配置完全相同），不重复训练。
    sens_train = build_sensitivity_frame(train)
    sens_valid = build_sensitivity_frame(valid)
    sens_oot = build_sensitivity_frame(oot)
    ablation = sensitivity_ablation(
        X_train, y_train, X_valid, y_valid, X_oot, y_oot,
        sens_train, sens_valid, sens_oot, cfg,
        base_model=lgbm, base_p_oot=p_lgbm_oot,
    )
    _save_table(ablation, out / "tables" / "sensitivity_ablation.csv")
    if len(ablation):
        _save_table(ablation.attrs.get("iv", pd.DataFrame()), out / "tables" / "sensitivity_iv.csv")
        ablation_iv = ablation.attrs.get("iv", pd.DataFrame())
    else:
        ablation_iv = pd.DataFrame()
    LOGGER.info(
        "敏感性对照完成: %s",
        {
            r.feature_set: round(r.oot_auc, 4)
            for r in ablation.itertuples(index=False)
        }
        if len(ablation)
        else "N/A",
    )

    tree_df = lgbm.model.booster_.trees_to_dataframe()
    scorecard_json_kb = (out / "scorecard.json").stat().st_size / 1024
    lgbm_model_mb = (out / "models" / "lgbm.joblib").stat().st_size / (1024 * 1024)
    lgbm_model_kb = lgbm_model_mb * 1024
    cum_gain = importance["gain_pct"].cumsum()

    # 单笔打分的计算量比值。旧版用 "树数 / 变量数" = 2034/12 = 169.5，这个口径站不住：
    # 树的开销取决于每棵树的深度，不是树的棵数除以变量数。这里改成可辩护的口径——
    # 单笔打分的**节点比较次数**：树模型需沿每棵树从根走到叶（深度次比较），
    # 评分卡需对每个变量查一次分箱表（变量数次比较）。
    lgbm_tree_depths = (
        tree_df.loc[tree_df["split_feature"].isna(), "node_depth"]
        .groupby(tree_df.loc[tree_df["split_feature"].isna(), "tree_index"])
        .mean()
    )
    lgbm_node_comparisons = float(lgbm_tree_depths.sum())
    scorecard_lookups = float(len(selected))
    interpretability = {
        "scorecard_features": len(selected),
        "lgbm_features": len(lgbm.features),
        "feature_ratio": len(lgbm.features) / max(len(selected), 1),
        "scorecard_bins": int(len(card.table()) - 1),
        "lgbm_trees": int(tree_df["tree_index"].nunique()),
        "lgbm_leaves": int(tree_df["split_feature"].isna().sum()),
        "lgbm_max_depth": int(tree_df["node_depth"].max()),
        "lgbm_avg_depth": float(tree_df["node_depth"].mean()),
        # 单笔打分节点比较次数（可辩护口径），替代旧版无意义的 score_ops_ratio
        "lgbm_node_comparisons_per_score": lgbm_node_comparisons,
        "scorecard_lookups_per_score": scorecard_lookups,
        "scoring_ops_ratio": lgbm_node_comparisons / max(scorecard_lookups, 1.0),
        "top10_gain": float(importance["gain_pct"].head(10).sum()),
        "top20_gain": float(importance["gain_pct"].head(20).sum()),
        "features_for_80pct_gain": int((cum_gain < 0.8).sum() + 1),
        "auc_gain": float(auc_lgbm_full - auc_card),
        "auc_gain_model_form": float(auc_lgbm12 - auc_card),
        "auc_gain_feature_count": float(auc_lgbm_full - auc_lgbm12),
        "auc_gain_per_extra_feature": float(
            (auc_lgbm_full - auc_lgbm12) / max(len(lgbm.features) - len(selected), 1)
        ),
        "auc_gain_per_100_trees": float(
            (auc_lgbm_full - auc_card) / max(tree_df["tree_index"].nunique() / 100, 1)
        ),
        "scorecard_json_kb": float(scorecard_json_kb),
        "lgbm_model_mb": float(lgbm_model_mb),
        "model_size_ratio": float(lgbm_model_kb / max(scorecard_json_kb, 1e-6)),
    }

    calibration_metrics_dict = {
        "scorecard": calibration_metrics(y_oot, p_oot),
        "lgbm": calibration_metrics(y_oot, p_lgbm_oot),
    }
    cal_metrics_frame = pd.DataFrame(
        [{"model": name, **values} for name, values in calibration_metrics_dict.items()]
    )
    _save_table(cal_metrics_frame, out / "tables" / "calibration_metrics_oot.csv")

    # LightGBM 对照组同样做校准与终身换算，否则与评分卡的阈值不可比。
    # 校准器各自在自己的验证集预测上 fit，OOT 只 transform。
    lgbm_platt = PlattCalibrator().fit(p_lgbm_valid, y_valid)
    p_lgbm_valid_cal = lgbm_platt.transform(p_lgbm_valid)
    p_lgbm_oot_cal = lgbm_platt.transform(p_lgbm_oot)
    lgbm_oot_lt = apply_lifetime_conversion(
        pd.DataFrame(
            {
                "prob_model": p_lgbm_oot,
                "prob_calibrated": p_lgbm_oot_cal,
                "term_num": oot["term_num"].to_numpy(),
            }
        ),
        term_e,
    )
    lgbm_valid_lt = apply_lifetime_conversion(
        pd.DataFrame(
            {
                "prob_model": p_lgbm_valid,
                "prob_calibrated": p_lgbm_valid_cal,
                "term_num": valid["term_num"].to_numpy(),
            }
        ),
        term_e,
    )
    p_lgbm_oot_lt = lgbm_oot_lt["prob_lifetime"].to_numpy()
    p_lgbm_valid_lt = lgbm_valid_lt["prob_lifetime"].to_numpy()
    joblib.dump(lgbm_platt, out / "models" / "lgbm_calibrator.joblib")

    lgbm_scan_valid = threshold_scan(y_valid_lt, p_lgbm_valid_lt, thresholds=grid_lt)
    lgbm_scan_oot = threshold_scan(y_oot_lt, p_lgbm_oot_lt, thresholds=grid_lt)
    lgbm_chosen = choose_threshold(lgbm_scan_valid, cfg.threshold_target_bad_rate, cfg.threshold_min_approval)
    lgbm_conf_oot = threshold_confusion(y_oot_lt, p_lgbm_oot_lt, float(lgbm_chosen["threshold"]))
    lgbm_same_rate = approval_at_rate(y_oot_lt, p_lgbm_oot_lt, conf_oot["approval_rate"], thresholds=grid_lt)
    threshold_summary = _threshold_summary(chosen, conf_oot, lgbm_chosen, lgbm_conf_oot, lgbm_same_rate)
    _save_table(threshold_summary, out / "tables" / "threshold_summary.csv")
    _save_table(lgbm_scan_valid, out / "tables" / "threshold_scan_lgbm_valid.csv")
    _save_table(lgbm_scan_oot, out / "tables" / "threshold_scan_lgbm_oot.csv")

    # swap-set 分析：两个模型都在终身口径 PD 上排序，标签同样用加权终身口径。
    # 批准人数取自评分卡选中阈值，保证 swap-in / swap-out 规模可比。
    swap_summary, swap_groups = swap_set_analysis(
        y_oot_lt, p_oot_lt, p_lgbm_oot_lt, n_approve=int(conf_oot["approved_n"])
    )
    _save_table(pd.DataFrame([swap_summary]), out / "tables" / "swap_set_summary.csv")
    _save_table(swap_groups, out / "tables" / "swap_set_groups.csv")

    model_comparison = pd.DataFrame(
        [
            {"model": "WOE 评分卡（14 变量）", "pd_scale": "vintage_mob12", "auc": metrics["oot"]["auc"], "ks": metrics["oot"]["ks"], "gini": metrics["oot"]["gini"], "pr_auc": metrics["oot"]["pr_auc"], "brier": metrics["oot"]["brier"], "log_loss": metrics["oot"]["log_loss"]},
            {"model": "LightGBM-同变量（14 变量）", "pd_scale": "vintage_mob12", "auc": m_lgbm12["oot"]["auc"], "ks": m_lgbm12["oot"]["ks"], "gini": m_lgbm12["oot"]["gini"], "pr_auc": m_lgbm12["oot"]["pr_auc"], "brier": m_lgbm12["oot"]["brier"], "log_loss": m_lgbm12["oot"]["log_loss"]},
            {"model": "LightGBM-full（82 变量）", "pd_scale": "vintage_mob12", "auc": lgbm_metrics["oot"]["auc"], "ks": lgbm_metrics["oot"]["ks"], "gini": lgbm_metrics["oot"]["gini"], "pr_auc": lgbm_metrics["oot"]["pr_auc"], "brier": lgbm_metrics["oot"]["brier"], "log_loss": lgbm_metrics["oot"]["log_loss"]},
        ]
    )
    _save_table(model_comparison, out / "tables" / "model_comparison_oot.csv")

    # ------------------------------------------------------------------ 9. 图表
    fig = out / "figures"
    figures: dict[str, Path] = {}

    def add(name: str, func, *args, **kwargs) -> None:
        path = fig / f"{name}.png"
        figures[name] = path
        func(*args, path=path, **kwargs)

    add("01_sample_flow", plots.plot_sample_flow, base_all)
    add("02_vintage", plots.plot_vintage, vintage_table)
    add("03_target_oot", plots.plot_target_distribution, oot, model_target)
    add("04_missing_train", plots.plot_missing, train)
    numeric_selected = [f for f in selected if f in CN_NAMES][:8]
    add("05_feature_distributions", plots.plot_feature_distributions, X_train, numeric_selected)
    add("06_iv", plots.plot_iv, iv_table)
    add("07_woe", plots.plot_woe, binnings, selected)
    # ROC/KS 是排序指标，用 vintage 口径 PD 即可（终身换算是分段单调，会轻微改变
    # 跨期限次序，但排序能力主要由 vintage PD 决定）。
    add("08_roc_oot", plots.plot_roc, y_oot, {"WOE 评分卡": p_oot_cal, "LightGBM-同变量": p_lgbm12_oot, "LightGBM-full": p_lgbm_oot_cal})
    add("09_ks_oot", plots.plot_ks, y_oot, p_oot_cal)
    add(
        "10_calibration_oot",
        plots.plot_calibration,
        {"WOE 评分卡": calibration["scorecard_oot"], "LightGBM": calibration_table(y_oot, p_lgbm_oot)},
    )
    add("11_lift_oot", plots.plot_lift, y_oot, {"WOE 评分卡": p_oot_cal, "LightGBM-full": p_lgbm_oot_cal})
    add("12_psi_quarter", plots.plot_psi_quarter, psi_quarter, cfg)
    add("13_psi_features", plots.plot_psi_features, psi_features, cfg)
    # 阈值扫描图使用终身口径，与阈值选择口径一致
    add("14_threshold_scan", plots.plot_threshold_scan, scan_valid_lt, scan_oot_lt, float(chosen["threshold"]), target_bad_rate=cfg.threshold_target_bad_rate)
    add("15_score_distribution", plots.plot_score_distribution, scored_oot, "target", card.grade_cutoffs)
    add("16_grade", plots.plot_grade, grade_tbl)
    add("17_lgbm_importance", plots.plot_lgbm_importance, importance)
    add("18_model_comparison", plots.plot_model_comparison, model_comparison)
    add("19_swap_set", plots.plot_swap_set, swap_summary, swap_groups)
    add(
        "20_calibration_curve",
        plots.plot_calibration_diagnostics,
        y_oot,
        p_oot_cal,
        p_lgbm_oot_cal,
        calibration_metrics_dict,
    )

    # ------------------------------------------------------------------ 10. 监控与报告
    oot_years = dq.loc[dq["issue_year"].isin([2017, 2018])]
    current_oot = (
        (oot_years["current_rate"] * oot_years["total"]).sum() / oot_years["total"].sum()
        if len(oot_years) and oot_years["total"].sum() > 0
        else 0.0
    )
    row_2018 = dq.loc[dq["issue_year"] == 2018, "current_rate"]
    current_2018 = float(row_2018.iloc[0]) if len(row_2018) else 0.0
    ctx = {
        "cfg": cfg,
        "generated_at": generated_at,
        "runtime_seconds": time.perf_counter() - started,
        "base_shape": base_all.shape,
        "train_shape": train.shape,
        "valid_shape": valid.shape,
        "oot_shape": oot.shape,
        "data_quality": dq,
        "vintage": vintage_table,
        "vintage_comp": vintage_comp,
        "iv_table": iv_table,
        "binnings": binnings,
        "selected_features": selected,
        "coef_table": selection.coef_table,
        "vif": vif_final,
        "vif_initial": vif_initial,
        "model": selection.model,
        "metrics": {"logit": metrics, "lgbm": lgbm_metrics},
        "metrics_lifetime": metrics_lifetime,
        "bootstrap": bootstrap,
        "calibration": calibration,
        "calibration_by_scale": calibration_by_scale,
        "calibrator_comparison": calibrator_comparison,
        "pd_scale_comparison": pd_scale_comparison,
        "label_scale_check": label_scale_check,
        "emergence_curve": emergence_all,
        "emergence_by_term": term_e,
        "emergence_by_grade": emergence_grade,
        "emergence_grade_within_term": emergence_within,
        "emergence_by_year_structure": emergence_year_struct,
        "grade_calibration_lifetime": grade_cal_lt,
        "model_attribution": attribution,
        "sensitivity_ablation": ablation,
        "sensitivity_iv": ablation_iv,
        "vintage_by_scale": vintage_scores,
        "term_emergence": term_e,
        "threshold_sensitivity": sens,
        "threshold_stability": stab,
        "lift_scorecard": lift_table(y_oot, p_oot_cal),
        "lift_lgbm": lift_table(y_oot, p_lgbm_oot_cal),
        "scorecard": card,
        "scored_oot": scored_oot,
        "grade_table": grade_tbl,
        "grade_calibration": grade_cal,
        "psi_valid": psi_valid,
        "psi_oot": psi_oot,
        "psi_features": psi_features,
        "psi_quarter": psi_quarter,
        "threshold_scan_valid": scan_valid_lt,
        "threshold_scan_oot": scan_oot_lt,
        "chosen_threshold": chosen,
        "threshold_confusion": conf_oot,
        "oot_diagnostic": oot_diagnostic,
        "threshold_summary": threshold_summary,
        "lgbm_importance": importance,
        "model_comparison": model_comparison,
        "lgbm_best_iteration": lgbm.best_iteration,
        "figures": figures,
        "current_share_oot": float(current_oot),
        "current_share_2018": float(current_2018),
        "lgbm_conf": lgbm_conf_oot,
        "lgbm_same_rate": lgbm_same_rate,
        "calibration_metrics": calibration_metrics_dict,
        "swap_summary": swap_summary,
        "swap_groups": swap_groups,
        "interpretability_cost": interpretability,
    }

    _save_table(dq, out / "tables" / "data_quality.csv")
    _save_table(vintage_table, out / "tables" / "vintage.csv")
    _save_table(threshold_summary, out / "tables" / "threshold_summary.csv")
    _save_table(pd.DataFrame([conf_oot]), out / "tables" / "confusion_oot.csv")
    _save_table(_metrics_frame({"scorecard": metrics["oot"], "lgbm": lgbm_metrics["oot"]}), out / "tables" / "metrics_oot.csv")

    if cfg.make_report:
        # 生成监控模板，但不覆盖人工维护的 v3 monitoring_plan.md。
        # v3 HTML 报告由 scripts/build_v3_report.py 读取产物生成，避免 pipeline
        # 用 v1 报告生成器覆盖 outputs/report.html。
        LOGGER.info("生成监控模板（monitoring_plan_generated.md）...")
        build_monitoring_plan(ctx, out / "monitoring_plan_generated.md")

    metadata = {
        "project": "LendingClub 信用风险评分卡 / OOT 监控",
        "generated_at": generated_at,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "config": {
            "train": [cfg.train_start, cfg.train_end],
            "valid": [cfg.valid_start, cfg.valid_end],
            "oot": [cfg.oot_start, cfg.oot_end],
            "iv_threshold": cfg.iv_threshold,
            "max_features": cfg.max_features,
            "base_score": cfg.base_score,
            "pdo": cfg.pdo,
            "base_odds": cfg.base_odds,
            "snapshot_date": cfg.snapshot_date,
            "performance_mob": cfg.performance_mob,
            "threshold_target_bad_rate": cfg.threshold_target_bad_rate,
        },
        "data": {
            "base_rows": int(len(base_all)),
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "oot_rows": int(len(oot)),
            # 坏账率一律用 MOB 对齐的 vintage 口径（建模标签），不再用终身口径 bad_flag，
            # 否则与模型实际学习的目标不一致，会误导读者。
            "target_scale": f"vintage_mob{cfg.performance_mob}",
            "bad_rate": float(train[model_target].mean()),
            "bad_rate_valid": float(valid[model_target].mean()),
            "bad_rate_oot": float(oot[model_target].mean()),
            "oot_upper_bound_note": (
                f"OOT 上界 {cfg.oot_end} = 快照日 {cfg.snapshot_date} − {cfg.performance_mob} 个月；"
                "2018Q2+ 观察期不足，无法在 MOB 对齐口径下评估，已排除"
            ),
            "current_share_oot": float(ctx["current_share_oot"]),
            "current_share_2018": float(ctx["current_share_2018"]),
        },
        "selected_features": selected,
        "grade_pd_cutoffs": [float(c) for c in cfg.grade_pd_cutoffs],
        "decision_policy_file": "decision_policy.json",
        "metrics": {"scorecard": metrics, "lgbm": lgbm_metrics},
        "metrics_lifetime": metrics_lifetime,
        "bootstrap_auc_oot": bootstrap,
        "psi": {"valid": psi_valid, "oot": psi_oot, "min_period_n": cfg.psi_min_period_n},
        "calibration_oot": calibration_metrics_dict,
        # 校准与终身换算是本次修正的核心，单独记录便于复核
        "calibration": {
            "platt_a": platt.a_,
            "platt_b": platt.b_,
            "oot_mean_pd_raw": float(np.mean(p_oot)),
            "oot_mean_pd_calibrated": float(np.mean(p_oot_cal)),
            "oot_mean_pd_lifetime": float(np.mean(p_oot_lt)),
            "oot_observed_vintage_bad_rate": float(np.mean(y_oot)),
            "oot_observed_lifetime_bad_rate": observed_lt_rate,
        },
        "emergence_by_term": {
            r.segment: float(r.emergence_frac) for r in term_e.itertuples(index=False)
        },
        "swap_set": swap_summary,
        "interpretability_cost": interpretability,
        "threshold": {
            # 阈值一律在终身口径 PD 上选择，故 threshold 是终身坏账率的决策边界
            "pd_scale": "lifetime",
            "selected": float(chosen["threshold"]),
            "approval_rate_oot": conf_oot["approval_rate"],
            "approved_bad_rate_oot": conf_oot["approved_bad_rate"],
            "rejected_bad_rate_oot": (
                conf_oot["rejected_bad"] / conf_oot["rejected_n"]
                if conf_oot["rejected_n"]
                else float("nan")
            ),
            "bad_capture_oot": conf_oot["bad_capture"],
            "target_bad_rate": cfg.threshold_target_bad_rate,
            "lifetime_bad_rate_oot": observed_lt_rate,
        },
        "threshold_sensitivity": sens.to_dict(orient="records"),
        "model_attribution": attribution.to_dict(orient="records"),
        "scorecard": {
            "base_points": card.base_points,
            "factor": card.factor,
            "offset": card.offset,
            "grade_cutoffs": card.grade_cutoffs,
        },
        "lgbm_best_iteration": lgbm.best_iteration,
        "versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "platform": platform.platform(),
        },
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    LOGGER.info("=" * 78)
    LOGGER.info("完成: %.1f 秒", metadata["runtime_seconds"])
    LOGGER.info("v3 报告: 运行 scripts/build_v3_report.py")
    LOGGER.info("监控方案: %s", out / "monitoring_plan.md")
    LOGGER.info("评分卡: %s", out / "scorecard.csv")
    LOGGER.info("=" * 78)
    return ctx
