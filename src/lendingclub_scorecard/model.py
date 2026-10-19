"""逻辑回归评分卡与 LightGBM 对照模型。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .binning import BinningResult, woe_frame
from .config import CN_NAMES, PipelineConfig
from .data import feature_label

LOGGER = logging.getLogger(__name__)


@dataclass
class LogitModel:
    result: object
    features: list[str]

    @property
    def intercept(self) -> float:
        return float(self.result.params["const"])

    def predict(self, X_woe: pd.DataFrame) -> np.ndarray:
        X = sm.add_constant(X_woe[self.features], has_constant="add")
        return np.asarray(self.result.predict(X), dtype=float)


def fit_logit(X_woe: pd.DataFrame, y: pd.Series) -> LogitModel:
    features = list(X_woe.columns)
    X = sm.add_constant(X_woe[features], has_constant="add")
    y = pd.Series(y).astype(int)
    try:
        result = sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Newton 法失败，改用 BFGS: %s", exc)
        result = sm.Logit(y, X).fit(disp=0, method="bfgs", maxiter=500)
    if not result.mle_retvals.get("converged", True):
        LOGGER.warning("逻辑回归未完全收敛")
    LOGGER.info(
        "逻辑回归: n=%s, 变量=%s, pseudo R2=%.4f",
        f"{len(y):,}",
        len(features),
        result.prsquared,
    )
    return LogitModel(result=result, features=features)


def coefficient_table(model: LogitModel) -> pd.DataFrame:
    result = model.result
    conf = result.conf_int()
    rows = []
    for name in ["const", *model.features]:
        rows.append(
            {
                "feature": name,
                "name_cn": "常数项" if name == "const" else feature_label(name),
                "coef": float(result.params[name]),
                "std_err": float(result.bse[name]),
                "z": float(result.params[name] / result.bse[name]),
                "p_value": float(result.pvalues[name]),
                "ci_lower": float(conf.loc[name, 0]),
                "ci_upper": float(conf.loc[name, 1]),
                "odds_ratio": float(np.exp(result.params[name])),
            }
        )
    return pd.DataFrame(rows)


def model_formula(model: LogitModel) -> str:
    parts = [f"{model.intercept:+.4f}"]
    for feature in model.features:
        parts.append(f"{model.result.params[feature]:+.4f}*WOE({feature})")
    return "logit(PD) = " + " ".join(parts)


def vif_table(X_woe: pd.DataFrame) -> pd.DataFrame:
    X = sm.add_constant(X_woe, has_constant="add")
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            value = float(variance_inflation_factor(X.values, i))
        except Exception:  # noqa: BLE001
            value = float("nan")
        rows.append({"feature": col, "name_cn": feature_label(col), "VIF": value})
    return pd.DataFrame(rows)


@dataclass
class SelectionResult:
    model: LogitModel
    features: list[str]
    coef_table: pd.DataFrame
    log: pd.DataFrame
    # 逐步删减前的初始入模集（IV 达标 + 相关性去重后），用于"剔除前 VIF"诊断
    initial_features: list[str] = field(default_factory=list)


def candidate_features(
    iv_table: pd.DataFrame,
    X_woe: pd.DataFrame,
    cfg: PipelineConfig,
) -> list[str]:
    """IV 达标且分箱有效的候选变量（即逐步删减的起点集合）。

    单独抽出是为了让流程层能在**剔除前**的候选集上计算 VIF。旧版只在最终入模
    变量上算 VIF，被剔除变量的共线性信息全部丢失，看到的"干净 VIF"只是幸存者
    的 VIF。候选集 VIF 才能回答"当初为什么必须剔除某个变量"。
    """
    return [
        row.feature
        for row in iv_table.itertuples(index=False)
        if row.iv >= cfg.iv_threshold and row.n_bins >= 2 and row.feature in X_woe.columns
    ]


def select_and_fit(
    X_woe: pd.DataFrame,
    y: pd.Series,
    iv_table: pd.DataFrame,
    cfg: PipelineConfig,
) -> SelectionResult:
    """IV 排序 + 相关性去重 + p 值/系数方向逐步删减。"""
    candidates = candidate_features(iv_table, X_woe, cfg)
    selected: list[str] = []
    corr = X_woe.corr().abs()
    for feature in candidates:
        if len(selected) >= cfg.max_features:
            break
        if selected and corr.loc[feature, selected].max() >= cfg.corr_threshold:
            continue
        selected.append(feature)
    if len(selected) < 5:
        selected = candidates[: min(5, len(candidates))]

    # 剔除前的初始入模集（IV 达标 + 相关性去重后，逐步删减前）。
    # 保留它是为了在流程层计算"剔除前 VIF"，与"剔除后 VIF"对照，暴露幸存者偏差。
    initial_features = list(selected)

    log_rows = []

    def _fit(features: list[str]) -> LogitModel:
        return fit_logit(X_woe[features], y)

    def _diagnostics(model: LogitModel, feature: str) -> dict:
        """被剔除变量在**剔除时刻**的共线性与方向状态。

        旧版只在最终入模变量上算 VIF，被剔除变量的共线性信息全部丢失，导致
        "负系数为何出现"无法诊断——VIF 表看着干净（最大 2.35），其实那只是
        幸存者的 VIF，属于幸存者偏差。这里在剔除发生时，用**当时的模型**（变量
        仍在其中）记录 VIF、与保留变量的最大相关、以及单变量方向，使剔除可复盘。

        判读：若 univariate_corr_with_y 为正（该变量单看是风险因子）但多变量系数
        为负，即典型的共线性符号翻转（抑制效应），剔除是合理的；若单变量本就为负，
        则说明该变量方向与业务预期不符，需在报告中标注而非默默剔除。
        """
        kept = list(model.features)  # 此刻 feature 仍在 kept 中
        diag = {
            "vif_at_removal": np.nan,
            "max_abs_corr_with_kept": np.nan,
            "max_corr_partner": "",
            "univariate_corr_with_y": np.nan,
        }
        # 1. 剔除时刻的 VIF（在当前模型的全部变量上算）
        if feature in kept:
            try:
                design = sm.add_constant(X_woe[kept], has_constant="add")
                idx = list(design.columns).index(feature)
                diag["vif_at_removal"] = float(variance_inflation_factor(design.values, idx))
            except Exception:  # noqa: BLE001
                pass
        # 2. 与其余保留变量的最大相关
        others = [c for c in kept if c != feature and c in X_woe.columns]
        if feature in X_woe.columns and others:
            corrs = X_woe[others].corrwith(X_woe[feature]).abs()
            diag["max_abs_corr_with_kept"] = float(corrs.max())
            diag["max_corr_partner"] = str(corrs.idxmax())
        # 3. 单变量方向：WOE 已统一为坏/好方向，单变量相关应为正
        if feature in X_woe.columns:
            try:
                diag["univariate_corr_with_y"] = float(np.corrcoef(X_woe[feature], y)[0, 1])
            except Exception:  # noqa: BLE001
                pass
        return diag

    model = _fit(selected)
    for _ in range(10):
        pvalues = model.result.pvalues.drop(labels=["const"], errors="ignore")
        worst_p = float(pvalues.max()) if len(pvalues) else 0.0
        worst_feature = str(pvalues.idxmax()) if len(pvalues) else None
        if worst_p <= cfg.p_value_threshold and len(selected) > 5:
            break
        if worst_p <= cfg.p_value_threshold:
            break
        if len(selected) <= 5:
            break
        log_rows.append(
            {
                "step": "p值剔除",
                "feature": worst_feature,
                "name_cn": feature_label(worst_feature),
                "p_value": worst_p,
                "coef": float(model.result.params.get(worst_feature, np.nan)),
                **_diagnostics(model, worst_feature),
            }
        )
        selected.remove(worst_feature)
        model = _fit(selected)

    # WOE 已统一为坏/好方向，系数应为正；负系数视为不稳定并剔除
    for _ in range(5):
        coefs = model.result.params.drop(labels=["const"], errors="ignore")
        negative = coefs[coefs < 0]
        if negative.empty or len(selected) <= 5:
            break
        worst_feature = str(negative.idxmin())
        log_rows.append(
            {
                "step": "负系数剔除",
                "feature": worst_feature,
                "name_cn": feature_label(worst_feature),
                "p_value": float(model.result.pvalues[worst_feature]),
                "coef": float(negative.min()),
                **_diagnostics(model, worst_feature),
            }
        )
        selected.remove(worst_feature)
        model = _fit(selected)

    coef = coefficient_table(model)
    log_columns = [
        "step",
        "feature",
        "name_cn",
        "p_value",
        "coef",
        "vif_at_removal",
        "max_abs_corr_with_kept",
        "max_corr_partner",
        "univariate_corr_with_y",
    ]
    selection_log = (
        pd.DataFrame(log_rows).reindex(columns=log_columns)
        if log_rows
        else pd.DataFrame(columns=log_columns)
    )
    LOGGER.info("最终入模变量 %s 个: %s", len(selected), ", ".join(selected))
    return SelectionResult(
        model=model,
        features=selected,
        coef_table=coef,
        log=selection_log,
        initial_features=initial_features,
    )


@dataclass
class LGBMModel:
    model: lgb.LGBMClassifier
    features: list[str]
    best_iteration: int

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X[self.features])[:, 1], dtype=float)


def fit_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cfg: PipelineConfig,
    features: list[str] | None = None,
) -> LGBMModel:
    """训练 LightGBM 对照组。

    features 为空时使用 X_train 的全部有效列（默认行为）；传入列名子集时只用这些列，
    用于构造"与评分卡同变量"的公平对照，从而把 AUC 增量拆解为模型形式与变量数量
    两部分贡献。早停仍只用验证集，不接触 OOT。
    """
    if features is None:
        features = [c for c in X_train.columns if X_train[c].nunique() > 1]
    else:
        features = [
            c
            for c in features
            if c in X_train.columns and X_train[c].nunique() > 1
        ]
        if not features:
            raise RuntimeError("指定的 features 在训练集中全部无效，无法构造公平对照")
    params = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=cfg.lgbm_n_estimators,
        learning_rate=cfg.lgbm_learning_rate,
        num_leaves=cfg.lgbm_num_leaves,
        min_child_samples=cfg.lgbm_min_child_samples,
        subsample=cfg.lgbm_subsample,
        colsample_bytree=cfg.lgbm_colsample_bytree,
        reg_alpha=cfg.lgbm_reg_alpha,
        reg_lambda=cfg.lgbm_reg_lambda,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        verbose=-1,
    )
    LOGGER.info("训练 LightGBM: %s 行 x %s 特征", f"{len(X_train):,}", len(features))
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train[features],
        y_train,
        eval_X=X_valid[features],
        eval_y=y_valid,
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(cfg.lgbm_early_stopping, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best_iteration = int(model.best_iteration_ or cfg.lgbm_n_estimators)
    LOGGER.info("LightGBM 最佳迭代轮数: %s", best_iteration)
    return LGBMModel(model=model, features=features, best_iteration=best_iteration)


def lgbm_importance(model: LGBMModel) -> pd.DataFrame:
    booster = model.model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    frame = pd.DataFrame(
        {
            "feature": model.features,
            "name_cn": [feature_label(f) for f in model.features],
            "gain": gain,
            "split": split,
        }
    )
    frame["gain_pct"] = frame["gain"] / frame["gain"].sum()
    return frame.sort_values("gain", ascending=False).reset_index(drop=True)


# ============================================================ 敏感性对照（消融实验）
def build_sensitivity_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """从原始帧构造敏感性对照列（FICO 中值、grade 序数编码）。

    这些列**只用于消融实验**，绝不进入主模型特征矩阵（Preprocessor 有断言拦截）。
    grade 用序数编码而非 one-hot：LC 的 grade A~G 本身是有序风险等级，序数编码
    能保留这个顺序，也让树模型的分裂更可解释。
    """
    out = pd.DataFrame(index=raw.index)
    if "fico_range_low" in raw.columns and "fico_range_high" in raw.columns:
        out["fico_mid"] = (
            raw["fico_range_low"].astype(float) + raw["fico_range_high"].astype(float)
        ) / 2.0
    if "grade" in raw.columns:
        order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        out["grade_ord"] = raw["grade"].astype(str).str.strip().str.upper().map(order)
    return out


def univariate_iv(x, y, n_bins: int = 10) -> float:
    """单变量信息值（IV），用于量化某个变量的独立预测力。

    等频分箱，WOE = ln(坏占比/好占比)，IV = Σ(坏占比−好占比)×WOE。
    加 0.5 的连续性校正避免空箱产生 inf。经验判读：
    IV<0.02 无用；0.02~0.1 弱；0.1~0.3 中；0.3~0.5 强；>0.5 可疑（需查泄漏）。
    """
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return float("nan")
    try:
        bins = pd.qcut(x, n_bins, duplicates="drop")
    except ValueError:
        return float("nan")
    grouped = pd.DataFrame({"y": y.to_numpy(), "bin": bins}).groupby("bin", observed=True)["y"]
    bad = grouped.sum()
    good = grouped.count() - bad
    bad_pct = (bad + 0.5) / (bad.sum() + 0.5 * len(bad))
    good_pct = (good + 0.5) / (good.sum() + 0.5 * len(good))
    woe = np.log(bad_pct / good_pct)
    return float(((bad_pct - good_pct) * woe).sum())


def sensitivity_ablation(
    X_train: pd.DataFrame,
    y_train,
    X_valid: pd.DataFrame,
    y_valid,
    X_oot: pd.DataFrame,
    y_oot,
    sens_train: pd.DataFrame,
    sens_valid: pd.DataFrame,
    sens_oot: pd.DataFrame,
    cfg: PipelineConfig,
    base_model: LGBMModel | None = None,
    base_p_oot: np.ndarray | None = None,
) -> pd.DataFrame:
    """FICO / grade 消融实验：量化"主模型不用 FICO"这一取舍的真实代价。

    旧版报告称"新数据含 FICO，为保持与上一版特征口径一致，主模型不使用"，但从
    未量化这个决定损失了多少区分能力。结果是 OOT AUC 0.6646 缺乏参照系——读者
    无法判断这是方法问题还是特征集天花板问题。

    本实验用同一个树模型（LightGBM）、同一套切分与标签，只改变是否加入 FICO /
    grade，形成干净的消融对照：
        base        主模型口径（预处理变量，无 FICO、无 grade）
        +fico       只加 FICO 中值
        +grade      只加 LC 评级序数
        +fico+grade 两者都加
    AUC 增量即该组变量的边际贡献。同时报告单变量 IV，用于判断 FICO 是否是
    "强到足以解释整个 AUC 差距"的变量。

    base_model / base_p_oot：若调用方已用相同特征集训练过基线模型，传入即可复用，
    避免重复训练（基线模型与 LightGBM-full 配置完全一致）。

    注意 grade 属于平台自身评级，主模型剔除它是有意的（复用等于借用 LC 的模型）。
    这里只做敏感性对照，不改变主模型口径。
    """
    sets = {
        "base（主模型口径，无 FICO/grade）": [],
        "+fico": ["fico_mid"],
        "+grade": ["grade_ord"],
        "+fico+grade": ["fico_mid", "grade_ord"],
    }

    # 单变量 IV：在训练集上算，衡量独立预测力
    iv_rows = []
    for col in ("fico_mid", "grade_ord"):
        if col in sens_train.columns:
            iv_rows.append(
                {
                    "variable": col,
                    "iv": univariate_iv(sens_train[col], y_train),
                    "coverage_train": float(sens_train[col].notna().mean()),
                    "coverage_oot": float(sens_oot[col].notna().mean())
                    if col in sens_oot.columns
                    else np.nan,
                }
            )
    iv_frame = pd.DataFrame(iv_rows)
    if len(iv_frame):
        LOGGER.info(
            "敏感性变量单变量 IV: %s",
            {r.variable: round(r.iv, 4) for r in iv_frame.itertuples(index=False)},
        )

    rows = []
    base_auc = np.nan
    for label, extra in sets.items():
        # 基线复用已训练模型，避免重复训练
        if not extra and base_model is not None and base_p_oot is not None:
            m = binary_metrics_oot(y_oot, base_p_oot)
            auc = float(m["auc"])
            base_auc = auc
            rows.append(
                {
                    "feature_set": label,
                    "n_features": len(base_model.features),
                    "added": "（无）",
                    "oot_auc": auc,
                    "oot_ks": float(m["ks"]),
                    "oot_pr_auc": float(m["pr_auc"]),
                    "auc_vs_base": 0.0,
                    "added_vars_gain_pct": 0.0,
                    "best_iteration": base_model.best_iteration,
                    "reused_base_model": True,
                }
            )
            LOGGER.info(
                "消融 %-34s: n=%3d, OOT AUC=%.4f（复用已训练基线）",
                label,
                len(base_model.features),
                auc,
            )
            continue

        if extra:
            missing = [c for c in extra if c not in sens_train.columns]
            if missing:
                LOGGER.warning("缺少敏感性列 %s，跳过 %s", missing, label)
                continue
            xt = pd.concat([X_train, sens_train[extra]], axis=1)
            xv = pd.concat([X_valid, sens_valid[extra]], axis=1)
            xo = pd.concat([X_oot, sens_oot[extra]], axis=1)
        else:
            xt, xv, xo = X_train, X_valid, X_oot

        # 对照列可能含缺失，用训练集中位数填补（只在训练集上统计，不泄漏）
        for col in extra:
            if col in xt.columns and xt[col].isna().any():
                fill = float(xt[col].median())
                xt[col] = xt[col].fillna(fill)
                xv[col] = xv[col].fillna(fill)
                xo[col] = xo[col].fillna(fill)

        model = fit_lightgbm(xt, y_train, xv, y_valid, cfg)
        p_oot = model.predict(xo)
        m = binary_metrics_oot(y_oot, p_oot)
        auc = float(m["auc"])
        if not extra:
            base_auc = auc
        imp = lgbm_importance(model)
        extra_gain = float(
            imp.loc[imp["feature"].isin(extra), "gain_pct"].sum()
        ) if extra else 0.0
        rows.append(
            {
                "feature_set": label,
                "n_features": len(model.features),
                "added": ", ".join(extra) if extra else "（无）",
                "oot_auc": auc,
                "oot_ks": float(m["ks"]),
                "oot_pr_auc": float(m["pr_auc"]),
                "auc_vs_base": auc - base_auc if np.isfinite(base_auc) else 0.0,
                "added_vars_gain_pct": extra_gain,
                "best_iteration": model.best_iteration,
                "reused_base_model": False,
            }
        )
        LOGGER.info(
            "消融 %-34s: n=%3d, OOT AUC=%.4f, Δbase=%+.4f, 新增变量增益占比=%.4f",
            label,
            len(model.features),
            auc,
            auc - base_auc if np.isfinite(base_auc) else 0.0,
            extra_gain,
        )

    out = pd.DataFrame(rows)
    if len(out) and len(iv_frame):
        out.attrs["iv"] = iv_frame
    return out if len(out) else pd.DataFrame()


def binary_metrics_oot(y, p) -> dict:
    """消融实验用的精简指标，避免与 evaluation 模块循环导入。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    from .evaluation import ks_statistic

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    auc = float(roc_auc_score(y, p))
    ks, _ = ks_statistic(y, p)
    return {"auc": auc, "ks": ks, "pr_auc": float(average_precision_score(y, p))}
