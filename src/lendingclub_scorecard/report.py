"""自包含中文 HTML 报告（图片 base64 内嵌，无外部依赖）。"""

from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import LEAKAGE_CHECKLIST
from .data import feature_label
from .monitoring import (
    alert_flow,
    baseline_table,
    hypothetical_cases,
    monitoring_metrics_table,
    psi_threshold_table,
)
from .scorecard import Scorecard

_CSS = """
:root{--navy:#17324d;--ink:#243746;--muted:#6b7c8c;--line:#dce5ec;--teal:#1f8a8a;--orange:#e07a4f;--gold:#d9a441;--bg:#f4f7fa;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei","PingFang SC","Noto Sans SC",system-ui,sans-serif;line-height:1.72;font-size:14px}
.page{max-width:1180px;margin:0 auto;padding:24px 20px 60px}
header.hero{background:linear-gradient(135deg,#17324d 0%,#24546f 55%,#1f8a8a 100%);color:#fff;border-radius:22px;padding:38px 42px;box-shadow:0 18px 44px rgba(23,50,77,.18)}
header.hero .eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,.72)}
header.hero h1{margin:8px 0 8px;font-size:33px;line-height:1.3;letter-spacing:.5px}
header.hero p{margin:0;color:rgba(255,255,255,.86);font-size:14.5px;max-width:930px}
.badges{margin-top:16px;display:flex;flex-wrap:wrap;gap:8px}
.badge{display:inline-block;padding:5px 12px;border:1px solid rgba(255,255,255,.32);border-radius:999px;font-size:11.5px;color:rgba(255,255,255,.92)}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-top:18px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:14px 15px;box-shadow:0 8px 22px rgba(23,50,77,.05)}
.kpi .label{font-size:11px;color:var(--muted);letter-spacing:.4px}
.kpi .value{font-size:24px;font-weight:800;color:var(--navy);line-height:1.25;margin-top:2px}
.kpi .sub{font-size:10.5px;color:#9aa9b6}
.toc{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px 22px;margin-top:18px}
.toc h2{margin:0 0 10px;font-size:15px;color:var(--navy)}
.toc ol{margin:0;padding-left:20px;columns:2;column-gap:32px}
.toc li{margin:3px 0;font-size:13px}
.toc a{color:var(--ink);text-decoration:none}
.toc a:hover{color:var(--teal)}
.section{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:26px 30px;margin-top:18px;box-shadow:0 8px 24px rgba(23,50,77,.045)}
.section-head{display:flex;align-items:baseline;gap:12px;margin-bottom:16px;border-bottom:1px solid #edf3f7;padding-bottom:12px}
.section-head .no{font-size:11px;letter-spacing:1.5px;color:var(--teal);font-weight:700}
.section-head h2{margin:0;font-size:21px;color:var(--navy)}
.section h3{font-size:15.5px;color:var(--navy);margin:22px 0 8px}
.section p{margin:8px 0}
.muted{color:var(--muted)}
.small{font-size:12px}
.lead{font-size:14.5px;color:#3c5060}
.note{background:#f3faf9;border-left:4px solid var(--teal);border-radius:0 10px 10px 0;padding:12px 16px;margin:14px 0;color:#3c5060}
.warn{background:#fff8f4;border-left:4px solid var(--orange);border-radius:0 10px 10px 0;padding:12px 16px;margin:14px 0;color:#5c463d}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;margin:12px 0 18px}
table.data{width:100%;border-collapse:collapse;font-size:12.5px;background:#fff}
table.data thead th{background:#f3f8fa;color:var(--navy);text-align:left;font-weight:600;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
table.data td{padding:8px 10px;border-bottom:1px solid #eef4f8;vertical-align:top}
table.data tbody tr:nth-child(even){background:#fbfdfe}
table.data tbody tr:hover{background:#f2faf9}
figure{margin:16px 0 20px;background:#fff;border:1px solid var(--line);border-radius:14px;padding:10px 10px 6px}
figure img{width:100%;height:auto;display:block;border-radius:8px}
figcaption{font-size:11.5px;color:var(--muted);text-align:center;padding:7px 4px 3px}
.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:14px 0}
.flow .step{background:#f7fafc;border:1px solid var(--line);border-radius:12px;padding:10px 9px;text-align:center}
.flow .step .n{display:inline-flex;width:22px;height:22px;border-radius:50%;background:var(--navy);color:#fff;align-items:center;justify-content:center;font-size:11px;margin-bottom:4px}
.flow .step .t{font-size:11.5px;color:var(--navy);font-weight:600;line-height:1.35}
.flow .step .d{font-size:10px;color:var(--muted);line-height:1.35;margin-top:3px}
.formula{background:#0f2435;color:#d9edf7;border-radius:12px;padding:14px 18px;font-family:Consolas,"Courier New",monospace;font-size:12.5px;overflow-x:auto;margin:12px 0}
.two-col{display:grid;grid-template-columns:1.05fr .95fr;gap:18px}
.tag{display:inline-block;font-size:10.5px;padding:2px 8px;border-radius:999px;background:#eaf6f4;color:#20786f;margin-left:4px}
.tag.warn{background:#fdf0eb;color:#b44b34}
ul.clean{margin:8px 0 8px 18px;padding:0}
ul.clean li{margin:4px 0}
ol.clean{margin:8px 0 8px 20px;padding:0}
ol.clean li{margin:6px 0}
footer{margin-top:22px;text-align:center;color:#93a3b1;font-size:11.5px;line-height:1.8}
code{background:#f2f7f9;color:var(--navy);padding:1px 5px;border-radius:5px;font-family:Consolas,"Courier New",monospace;font-size:12px}
pre.code{background:#0f2435;color:#d9edf7;border-radius:12px;padding:14px 18px;overflow-x:auto;font-size:12px;line-height:1.65}
pre.code code{background:none;color:inherit;padding:0}
a{color:#20786f}
@media (max-width:1000px){.kpis{grid-template-columns:repeat(3,1fr)}.flow{grid-template-columns:repeat(3,1fr)}.two-col{grid-template-columns:1fr}}
@media (max-width:620px){header.hero{padding:26px 22px}header.hero h1{font-size:23px}.section{padding:20px 18px}.kpis{grid-template-columns:1fr 1fr}.toc ol{columns:1}.flow{grid-template-columns:1fr 1fr}}
@media print{body{background:#fff}.section,.toc,.kpi{box-shadow:none;break-inside:avoid}header.hero{box-shadow:none}}
"""


def _fmt_int(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{int(value):,}"
    except Exception:  # noqa: BLE001
        return str(value)


def _fmt_pct(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.2%}"
    except Exception:  # noqa: BLE001
        return str(value)


def _fmt_float(value, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except Exception:  # noqa: BLE001
        return str(value)


def _fmt_p(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        value = float(value)
    except Exception:  # noqa: BLE001
        return str(value)
    if value < 1e-6:
        return "<0.000001"
    return f"{value:.4f}"


def render_table(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    headers: list[str] | None = None,
    formats: dict | None = None,
    max_rows: int | None = None,
) -> str:
    if df is None or len(df) == 0:
        return '<p class="muted">暂无数据。</p>'
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    if columns is not None:
        data = data[columns]
    if formats:
        for col, func in formats.items():
            if col in data.columns:
                data[col] = data[col].map(func)
    if headers is not None:
        data.columns = headers
    return '<div class="table-wrap">' + data.to_html(
        index=False, classes="data", border=0, escape=True, na_rep="—"
    ) + "</div>"


def _image(path, alt: str, caption: str) -> str:
    if path is None:
        return ""
    path = Path(path)
    if not path.exists():
        return f'<p class="muted">[缺少图片：{html.escape(path.name)}]</p>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _section(no: str, title: str, body: str, anchor: str) -> str:
    return (
        f'<section class="section" id="{anchor}">'
        f'<div class="section-head"><span class="no">{no}</span><h2>{title}</h2></div>'
        f"{body}</section>"
    )


def _metrics_rows(metrics: dict, lgbm_metrics: dict) -> pd.DataFrame:
    rows = []
    for model_name, data in (("WOE 评分卡", metrics), ("LightGBM", lgbm_metrics)):
        for dataset, values in data.items():
            rows.append(
                {
                    "模型": model_name,
                    "数据": {"train": "训练 2013-2015", "valid": "验证 2016", "oot": "OOT 2017-2018"}.get(dataset, dataset),
                    "样本数": values["n"],
                    "坏客户率": values["bad_rate"],
                    "AUC": values["auc"],
                    "KS": values["ks"],
                    "Gini": values["gini"],
                    "PR-AUC": values["pr_auc"],
                    "Brier": values["brier"],
                    "LogLoss": values["log_loss"],
                    "平均PD": values.get("mean_pd"),
                }
            )
    return pd.DataFrame(rows)


def build_report(ctx: dict, path: str | Path) -> None:
    cfg = ctx["cfg"]
    metrics = ctx["metrics"]["logit"]
    lgbm_metrics = ctx["metrics"]["lgbm"]
    oot = metrics["oot"]
    lgbm_oot = lgbm_metrics["oot"]
    card: Scorecard = ctx["scorecard"]
    conf = ctx["threshold_confusion"]
    chosen = ctx["chosen_threshold"]
    psi_features = ctx["psi_features"]
    psi_quarter = ctx["psi_quarter"]
    data_quality = ctx["data_quality"]
    vintage = ctx["vintage"]
    figures = ctx["figures"]
    generated_at = ctx.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ------------------------------------------------------------------ 表格
    data_quality_html = render_table(
        data_quality,
        columns=[
            "issue_year",
            "total",
            "resolved",
            "resolved_rate",
            "evaluable",
            "avg_obs_mob",
            "legacy_bad_rate",
            "vintage_mob12_bad_rate",
            "legacy_over_vintage_ratio",
            "current_rate",
        ],
        headers=[
            "放款年",
            "全部贷款",
            "已终态",
            "终态比例",
            "MOB12 可评估",
            "平均已观察月数",
            "旧口径坏账率（仅终态）",
            f"MOB{cfg.performance_mob} vintage 坏账率",
            "旧口径 / vintage",
            "Current 占比",
        ],
        formats={
            "issue_year": lambda x: str(int(x)),
            "total": _fmt_int,
            "resolved": _fmt_int,
            "resolved_rate": _fmt_pct,
            "evaluable": _fmt_int,
            "avg_obs_mob": lambda x: _fmt_float(x, 1),
            "legacy_bad_rate": _fmt_pct,
            "vintage_mob12_bad_rate": _fmt_pct,
            "legacy_over_vintage_ratio": lambda x: _fmt_float(x, 2),
            "current_rate": _fmt_pct,
        },
    )
    vintage_html = render_table(
        vintage,
        columns=["issue_quarter", "total", "resolved", "bad_rate", "current_rate", "late_rate", "avg_score", "avg_pd"],
        headers=["放款季度", "全部贷款", "已终态", "坏客户率", "Current 占比", "Late 占比", "平均评分", "平均 PD"],
        formats={
            "total": _fmt_int,
            "resolved": _fmt_int,
            "bad_rate": _fmt_pct,
            "current_rate": _fmt_pct,
            "late_rate": _fmt_pct,
            "avg_score": lambda x: _fmt_float(x, 1),
            "avg_pd": _fmt_pct,
        },
    )
    leakage_html = render_table(
        pd.DataFrame(
            LEAKAGE_CHECKLIST,
            columns=["字段/变量", "泄露类型", "为什么不能入模", "处理"],
        )
    )
    leakage_checklist_html = "".join(
        f"<li>{item}</li>"
        for item in [
            "目标标签是否直接或间接进入特征（loan_status、贷后状态、催收/和解结果）。",
            "贷后现金流字段是否被误用（total_pymnt、recoveries、out_prncp、last_pymnt_*）。",
            "贷后救助/处置字段是否被误用（hardship_*、settlement_*、debt_settlement_*）。",
            "是否使用了未来才补齐的字段（il_util、all_util、open_rv_12m、inq_last_12m 等）。",
            "是否复用了平台自身评级/定价（grade、sub_grade、int_rate、installment）。",
            "ID、URL、自由文本、高基数邮编/职位是否被误当特征。",
            "时间字段是否只用于切分或派生“放款时点”变量，而不是直接入模。",
            "缺失填补、截断、分箱、WOE 是否只在训练集拟合，避免验证/OOT 信息泄漏。",
            "重复样本是否跨训练/OOT 出现，导致同一借款人多条记录泄漏。",
            "每个字段的“可用时点”是否有数据字典/业务确认，而不是仅凭字段名判断。",
        ]
    )

    iv_html = render_table(
        ctx["iv_table"].head(22),
        columns=["name_cn", "group", "iv", "n_bins", "strength", "method"],
        headers=["变量", "类别", "IV", "分箱数", "预测力", "方法"],
        formats={"iv": lambda x: _fmt_float(x, 4)},
    )
    scorecard_html = render_table(
        card.table(),
        columns=["变量", "中文名", "分箱", "样本数", "坏客户率", "WOE", "系数", "得分"],
        formats={
            "样本数": _fmt_int,
            "坏客户率": _fmt_pct,
            "WOE": lambda x: "—" if pd.isna(x) else _fmt_float(x, 4),
            "系数": lambda x: "—" if pd.isna(x) else _fmt_float(x, 4),
            "得分": _fmt_int,
        },
    )
    coef_html = render_table(
        ctx["coef_table"],
        columns=["name_cn", "coef", "std_err", "z", "p_value", "odds_ratio", "ci_lower", "ci_upper"],
        headers=["变量", "系数 β", "标准误", "z 值", "p 值", "OR", "95% CI 下限", "95% CI 上限"],
        formats={
            "coef": lambda x: _fmt_float(x, 4),
            "std_err": lambda x: _fmt_float(x, 4),
            "z": lambda x: _fmt_float(x, 2),
            "p_value": _fmt_p,
            "odds_ratio": lambda x: _fmt_float(x, 4),
            "ci_lower": lambda x: _fmt_float(x, 4),
            "ci_upper": lambda x: _fmt_float(x, 4),
        },
    )
    vif_html = render_table(
        ctx["vif"],
        columns=["name_cn", "VIF"],
        headers=["变量", "VIF"],
        formats={"VIF": lambda x: _fmt_float(x, 2)},
    )
    vif_initial = ctx.get("vif_initial")
    vif_initial_html = render_table(
        vif_initial.sort_values("VIF", ascending=False).head(18)
        if vif_initial is not None and len(vif_initial)
        else None,
        columns=["feature", "name_cn", "VIF", "dropped_later"],
        headers=["变量", "中文名", "剔除前 VIF", "后续是否被剔除"],
        formats={
            "VIF": lambda x: _fmt_float(x, 2),
            "dropped_later": lambda x: "是" if x else "否（保留入模）",
        },
    )
    metrics_html = render_table(
        _metrics_rows(metrics, lgbm_metrics),
        formats={
            "样本数": _fmt_int,
            "坏客户率": _fmt_pct,
            "AUC": lambda x: _fmt_float(x, 4),
            "KS": lambda x: _fmt_float(x, 4),
            "Gini": lambda x: _fmt_float(x, 4),
            "PR-AUC": lambda x: _fmt_float(x, 4),
            "Brier": lambda x: _fmt_float(x, 4),
            "LogLoss": lambda x: _fmt_float(x, 4),
            "平均PD": _fmt_pct,
        },
    )
    calibration_html = render_table(
        ctx["calibration"]["scorecard_oot"],
        columns=["probability_bin", "count", "bad", "pred_mean", "observed", "gap"],
        headers=["概率分箱", "样本数", "坏样本数", "预测 PD 均值", "实际坏客户率", "偏差"],
        formats={
            "count": _fmt_int,
            "bad": _fmt_int,
            "pred_mean": _fmt_pct,
            "observed": _fmt_pct,
            "gap": lambda x: f"{float(x):+.2%}",
        },
    )
    lift_html = render_table(
        ctx["lift_scorecard"],
        columns=["top_pct", "n", "bad", "capture_rate", "lift", "bad_rate"],
        headers=["风险排序", "样本数", "坏样本数", "坏客户捕获率", "提升度", "该群体坏客户率"],
        formats={
            "top_pct": _fmt_pct,
            "n": _fmt_int,
            "bad": _fmt_int,
            "capture_rate": _fmt_pct,
            "lift": lambda x: f"{float(x):.2f}×",
            "bad_rate": _fmt_pct,
        },
    )
    cal_metrics = ctx["calibration_metrics"]
    cal_rows = []
    for key, label in (("scorecard", "WOE 评分卡"), ("lgbm", "LightGBM")):
        values = cal_metrics[key]
        cal_rows.append(
            {
                "模型": label,
                "平均预测 PD": values["mean_pred"],
                "实际坏账率": values["actual"],
                "校准截距": values["calibration_intercept"],
                "校准斜率": values["calibration_slope"],
                "ECE": values["ece"],
                "MCE": values["mce"],
                "Brier": values["brier"],
            }
        )
    cal_metrics_html = render_table(
        pd.DataFrame(cal_rows),
        formats={
            "平均预测 PD": _fmt_pct,
            "实际坏账率": _fmt_pct,
            "校准截距": lambda x: _fmt_float(x, 4),
            "校准斜率": lambda x: _fmt_float(x, 4),
            "ECE": lambda x: _fmt_float(x, 4),
            "MCE": lambda x: _fmt_float(x, 4),
            "Brier": lambda x: _fmt_float(x, 4),
        },
    )
    psi_features_html = render_table(
        psi_features.head(18),
        columns=["feature", "value", "level"],
        headers=["变量", "PSI", "等级"],
        formats={"feature": feature_label, "value": lambda x: _fmt_float(x, 4)},
    )
    psi_quarter_html = render_table(
        psi_quarter,
        columns=["period", "psi", "level"],
        headers=["放款季度", "Score PSI", "等级"],
        formats={"psi": lambda x: _fmt_float(x, 4)},
    )
    threshold_table = ctx["threshold_scan_valid"].copy()
    threshold_table = threshold_table[
        threshold_table["threshold"].round(2).isin([0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.28, 0.32])
    ]
    threshold_html = render_table(
        threshold_table,
        columns=["threshold", "approval_rate", "approved_bad_rate", "rejected_bad_rate", "bad_capture", "bad_rate_reduction"],
        headers=["PD 阈值", "批准率", "批准坏账率", "拒绝人群坏账率", "坏客户捕获率", "坏账率下降"],
        formats={
            "threshold": lambda x: _fmt_float(x, 2),
            "approval_rate": _fmt_pct,
            "approved_bad_rate": _fmt_pct,
            "rejected_bad_rate": _fmt_pct,
            "bad_capture": _fmt_pct,
            "bad_rate_reduction": _fmt_pct,
        },
    )
    threshold_summary_html = render_table(
        ctx["threshold_summary"],
        columns=["模型", "阈值来源", "阈值", "批准率", "批准坏账率", "坏客户捕获率", "审阅"],
        formats={
            "阈值": lambda x: _fmt_float(x, 2),
            "批准率": _fmt_pct,
            "批准坏账率": _fmt_pct,
            "坏客户捕获率": _fmt_pct,
        },
    )
    conf_html = render_table(
        pd.DataFrame(
            [
                ("PD 阈值", f"{conf['threshold']:.2f}"),
                ("批准人数 / 拒绝人数", f"{conf['approved_n']:,} / {conf['rejected_n']:,}"),
                ("批准率", f"{conf['approval_rate']:.2%}"),
                ("批准好客户 / 坏客户", f"{conf['approved_good']:,} / {conf['approved_bad']:,}"),
                ("拒绝好客户 / 坏客户", f"{conf['rejected_good']:,} / {conf['rejected_bad']:,}"),
                ("批准人群坏账率", f"{conf['approved_bad_rate']:.2%}"),
                ("坏客户捕获率（拒绝）", f"{conf['bad_capture']:.2%}"),
            ],
            columns=["项目", "OOT 2017-2018"],
        )
    )
    grade_html = render_table(
        ctx["grade_table"],
        columns=["grade", "count", "share", "bad", "bad_rate", "avg_score", "min_score", "max_score"],
        headers=["等级", "样本数", "占比", "坏样本数", "坏客户率", "平均分", "最低分", "最高分"],
        formats={
            "count": _fmt_int,
            "share": _fmt_pct,
            "bad": _fmt_int,
            "bad_rate": _fmt_pct,
            "avg_score": lambda x: _fmt_float(x, 1),
            "min_score": _fmt_int,
            "max_score": _fmt_int,
        },
    )
    grade_cal_html = render_table(
        ctx["grade_calibration"],
        columns=["grade", "count", "avg_pred_pd", "actual_bad_rate", "avg_score"],
        headers=["等级", "样本数", "平均预测 PD", "实际坏客户率", "平均分"],
        formats={
            "count": _fmt_int,
            "avg_pred_pd": _fmt_pct,
            "actual_bad_rate": _fmt_pct,
            "avg_score": lambda x: _fmt_float(x, 1),
        },
    )
    lgbm_imp_html = render_table(
        ctx["lgbm_importance"].head(15),
        columns=["name_cn", "gain_pct", "split"],
        headers=["变量", "Gain 占比", "分裂次数"],
        formats={"gain_pct": _fmt_pct, "split": _fmt_int},
    )
    comparison_html = render_table(
        ctx["model_comparison"],
        columns=["model", "auc", "ks", "gini", "pr_auc", "brier", "log_loss"],
        headers=["模型", "AUC", "KS", "Gini", "PR-AUC", "Brier", "LogLoss"],
        formats={
            "auc": lambda x: _fmt_float(x, 4),
            "ks": lambda x: _fmt_float(x, 4),
            "gini": lambda x: _fmt_float(x, 4),
            "pr_auc": lambda x: _fmt_float(x, 4),
            "brier": lambda x: _fmt_float(x, 4),
            "log_loss": lambda x: _fmt_float(x, 4),
        },
    )
    swap = ctx["swap_summary"]
    swap_compare_html = render_table(
        pd.DataFrame(
            [
                ("批准人数", f"{swap['approved_n']:,}", f"{swap['approved_n']:,}"),
                ("批准率", f"{swap['approval_rate']:.2%}", f"{swap['approval_rate']:.2%}"),
                ("批准坏账率", f"{swap['scorecard_bad_rate']:.2%}", f"{swap['lgbm_bad_rate']:.2%}"),
                ("坏客户捕获率", f"{swap['scorecard_bad_capture']:.2%}", f"{swap['lgbm_bad_capture']:.2%}"),
            ],
            columns=["指标", "WOE 评分卡", "LightGBM"],
        )
    )
    swap_flow_html = render_table(
        pd.DataFrame(
            [
                ("两模型批准集合重合率", f"{swap['overlap_rate']:.2%}"),
                ("Jaccard 相似度", f"{swap['jaccard']:.2%}"),
                ("Swap-in：仅 LightGBM 批准（人数）", f"{swap['swap_in_n']:,}"),
                ("Swap-out：仅评分卡批准（人数）", f"{swap['swap_out_n']:,}"),
                ("Swap-in 人群坏账率", f"{swap['swap_in_bad_rate']:.2%}"),
                ("Swap-out 人群坏账率", f"{swap['swap_out_bad_rate']:.2%}"),
                ("净减少坏客户（swap-out bad − swap-in bad）", f"{swap['net_bad_avoided']:,}"),
                ("净增加好客户", f"{swap['net_good_gain']:,}"),
            ],
            columns=["Swap-set 指标", "数值"],
        )
    )
    swap_groups_html = render_table(
        ctx["swap_groups"],
        columns=["group", "count", "bad", "bad_rate"],
        headers=["集合", "人数", "坏客户数", "坏客户率"],
        formats={"count": _fmt_int, "bad": _fmt_int, "bad_rate": _fmt_pct},
    )
    interp = ctx["interpretability_cost"]
    interp_html = render_table(
        pd.DataFrame(
            [
                (
                    "模型变量数",
                    f"{interp['scorecard_features']} 个",
                    f"{interp['lgbm_features']} 个",
                    f"{interp['feature_ratio']:.1f}×",
                ),
                (
                    "单笔评分计算",
                    f"{interp['scorecard_lookups_per_score']:.0f} 次分箱查表+求和",
                    f"{interp['lgbm_trees']:,} 棵树遍历，约 {interp['lgbm_node_comparisons_per_score']:,.0f} 次节点比较",
                    f"{interp['scoring_ops_ratio']:.0f}×",
                ),
                (
                    "模型结构",
                    f"1 个线性方程，{interp['scorecard_bins']} 个分箱",
                    f"{interp['lgbm_trees']:,} 棵树，{interp['lgbm_leaves']:,} 个叶子，最大深度 {interp['lgbm_max_depth']}",
                    "高",
                ),
                (
                    "单笔解释",
                    f"{interp['scorecard_features']} 项整数分相加，严格可加",
                    "需 SHAP/树路径，非严格可加",
                    "需额外解释器",
                ),
                (
                    "全局单调性",
                    "WOE 方向受控",
                    "无全局单调约束",
                    "合规/沟通成本",
                ),
                (
                    "部署与审计",
                    "表格/JSON，可人工复核",
                    "模型文件+特征管道+版本管理",
                    "高",
                ),
                (
                    "模型文件大小",
                    f"scorecard.json {interp['scorecard_json_kb']:.1f} KB",
                    f"lgbm.joblib {interp['lgbm_model_mb']:.2f} MB",
                    f"{interp['model_size_ratio']:.0f}×",
                ),
                (
                    "OOT AUC 增量",
                    "基准",
                    f"+{interp['auc_gain']:.4f}",
                    f"每增加 1 特征 +{interp['auc_gain_per_extra_feature']:.5f}；每 100 棵树 +{interp['auc_gain_per_100_trees']:.4f}",
                ),
                (
                    "特征集中度",
                    f"{interp['scorecard_features']} 个变量均有明确业务含义",
                    f"Top 10 / Top 20 gain 占 {interp['top10_gain']:.1%} / {interp['top20_gain']:.1%}；80% gain 需 {interp['features_for_80pct_gain']} 个特征",
                    "解释面更宽",
                ),
            ],
            columns=["维度", "WOE 评分卡", "LightGBM", "代价/差异"],
        )
    )
    cases_html = "".join(
        f'<div class="note"><h4 style="margin:4px 0 6px;color:#17324d">{case["case"]}</h4>'
        f'<p><strong>触发信号：</strong>{case["signal"]}</p>'
        f'<p><strong>排查顺序：</strong></p><ol class="clean">'
        + "".join(f"<li>{step}</li>" for step in case["steps"])
        + f'</ol><p><strong>处置建议：</strong>{case["decision"]}</p></div>'
        for case in hypothetical_cases()
    )
    psi_threshold_html = render_table(psi_threshold_table(cfg))
    monitoring_metrics_html = render_table(monitoring_metrics_table(cfg))
    baseline_html = render_table(baseline_table(ctx))
    alert_flow_html = "".join(
        f'<li><strong>{stage}</strong>（{owner}）：{action}</li>'
        for stage, action, owner in alert_flow()
    )

    # ------------------------------------------------------------------ 页面
    toc_items = [
        ("overview", "项目概览"),
        ("data", "数据与时间切分"),
        ("model", "特征工程与评分卡"),
        ("oot", "OOT 验证"),
        ("psi", "PSI 与 Vintage"),
        ("threshold", "阈值扫描"),
        ("lgbm", "LightGBM 对照"),
        ("monitoring", "上线后监控方案"),
        ("repro", "复现说明"),
        ("notes", "数据口径与限制"),
        ("limitations", "项目局限与后续计划"),
    ]
    toc_html = "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in toc_items)

    flow_steps = [
        ("01", "定义目标", "终态贷款；坏=核销/违约"),
        ("02", "时间切分", "2013-15 训练 / 16 验证 / 17-18 OOT"),
        ("03", "特征工程", "缺失/截断/one-hot"),
        ("04", "WOE 评分卡", "等频分箱 + 单调 + IV"),
        ("05", "OOT 与监控", "AUC/KS、PSI、Vintage"),
        ("06", "策略落地", "阈值扫描 + 监控方案"),
    ]
    flow_html = "".join(
        f'<div class="step"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for n, t, d in flow_steps
    )
    kpis = [
        ("OOT AUC（评分卡）", f"{oot['auc']:.4f}", "2017–2018 时间外样本"),
        ("OOT KS（评分卡）", f"{oot['ks']:.4f}", "排序能力"),
        ("OOT AUC（LightGBM）", f"{lgbm_oot['auc']:.4f}", "全特征对照"),
        ("Score PSI（train→OOT）", f"{ctx['psi_oot']:.4f}", "阈值 <0.10"),
        ("OOT 批准率", f"{conf['approval_rate']:.1%}", f"阈值 {conf['threshold']:.2f}"),
        ("OOT 批准坏账率", f"{conf['approved_bad_rate']:.1%}", f"目标 ≤{cfg.threshold_target_bad_rate:.0%}"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'
        for label, value, sub in kpis
    )

    selected_text = "、".join(f"<strong>{feature_label(f)}</strong>" for f in ctx["selected_features"])
    bootstrap = ctx["bootstrap"]

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LendingClub 信用风险评分卡 · OOT / PSI / Vintage 报告</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page">
<header class="hero">
  <div class="eyebrow">LENDINGCLUB · CREDIT RISK SCORECARD · OOT MONITORING</div>
  <h1>LendingClub 个人信贷评分卡<br>OOT 验证、PSI、Vintage 与上线后监控</h1>
  <p>基于 LendingClub 公开贷款数据（2012–2018），按 issue_d 做时间切分：2013–2015 训练、2016 验证、2017–2018 OOT。完成 WOE/IV 评分卡、LightGBM 对照、PSI 漂移、Vintage 资产质量、阈值扫描与上线后监控方案。</p>
  <div class="badges">
    <span class="badge">样本：{ctx['base_shape'][0]:,} 条 2012–2018 贷款</span>
    <span class="badge">目标：Charged Off / Default = 1</span>
    <span class="badge">OOT：2017–2018 年放款</span>
    <span class="badge">评分卡：600 基准 · PDO 20 · 好坏比 4:1</span>
    <span class="badge">LightGBM 对照</span>
    <span class="badge">生成时间：{generated_at}</span>
  </div>
</header>

<div class="kpis">{kpi_html}</div>

<nav class="toc"><h2>目录</h2><ol>{toc_html}</ol></nav>

{_section("01", "项目概览", f'''
<p class="lead">本项目把“评分卡开发”与“上线后监控”放在同一条时间轴上：训练集来自 2013–2015 年放款，2016 年用于阈值与超参数选择，2017–2018 年作为完全未参与训练的 OOT 样本。除了传统 AUC/KS，还输出 Score PSI、特征 PSI、Vintage 坏账率、阈值扫描和可执行的监控预警流程。</p>
<div class="flow">{flow_html}</div>
<div class="two-col">
  <div>
    <h3>关键结论</h3>
    <ul class="clean">
      <li>评分卡 OOT AUC <strong>{oot['auc']:.4f}</strong>、KS <strong>{oot['ks']:.4f}</strong>；2017–2018 客群漂移使 OOT 低于 2016 验证集（AUC {metrics['valid']['auc']:.4f}），正好用于演示监控。</li>
      <li>LightGBM OOT AUC <strong>{lgbm_oot['auc']:.4f}</strong>、KS <strong>{lgbm_oot['ks']:.4f}</strong>，比评分卡高约 <strong>{(lgbm_oot['auc']-oot['auc']):.4f}</strong> AUC，但可解释性和部署成本更高。</li>
      <li>Score PSI（训练→OOT）为 <strong>{ctx['psi_oot']:.4f}</strong>，处于稳定区间；但 2017–2018 年 Vintage 的 Current 占比明显上升，绝对坏账率需按账龄解读。</li>
      <li>按验证集目标“批准坏账率 ≤{cfg.threshold_target_bad_rate:.0%}”选择 PD 阈值 <strong>{conf['threshold']:.2f}</strong>，OOT 批准率 <strong>{conf['approval_rate']:.1%}</strong>、批准坏账率 <strong>{conf['approved_bad_rate']:.1%}</strong>、坏客户捕获率 <strong>{conf['bad_capture']:.1%}</strong>。</li>
      <li><strong>阈值跨期表现：</strong>同一阈值在 2016 验证集与 2017–2018 OOT 上分别对应批准坏账率 {chosen['approved_bad_rate']:.1%} 与 {conf['approved_bad_rate']:.1%}；跨期变化说明上线后必须监控“批准坏账率 vs 目标”，而不能假设一套阈值长期不变。</li>
    </ul>
  </div>
  <div class="note">
    <strong>为什么不用 LC 的 grade / int_rate？</strong> LendingClub 的 grade、sub_grade、int_rate 是平台自身评级与定价结果，直接入模相当于复用 LC 的模型，无法评估“自建评分卡”的增量价值。主模型只使用申请时点的申请表、征信局与交易线变量；LightGBM 使用同一套特征，保证对照公平。
  </div>
</div>
''', "overview")}

{_section("02", "数据与时间切分", f'''
<h3>2.1 目标与样本</h3>
<p>目标变量 <code>bad_flag</code>：<code>loan_status</code> 为 <strong>Charged Off / Default</strong> 记为 1，<strong>Fully Paid</strong> 记为 0。只保留已终态贷款；<code>Current</code>、<code>Late</code>、<code>In Grace Period</code> 等未定状态不进入训练与 OOT 评估，避免把“尚未发生”误标为好客户。</p>
{data_quality_html}
{_image(figures.get("03_target_oot"), "OOT 好坏分布", "图 2-1 2017–2018 OOT 好坏客户分布")}
<div class="warn"><strong>2017–2018 OOT 的删失偏差：</strong>2017–2018 年放款 cohort 的 Current 占比约 <strong>{ctx['current_share_oot']:.1%}</strong>，终态样本只是较早结清/较早核销的部分。因此 2017–2018 OOT 更适合比较 <strong>排序能力（AUC/KS）与分布漂移（PSI）</strong>；绝对坏账率应结合同账龄 Vintage 曲线解读。</div>
{_image(figures.get("01_sample_flow"), "样本量与终态比例", "图 2-2 各放款年度的样本量与终态比例")}
<h3>2.2 时间外切分</h3>
<div class="flow">
  <div class="step"><div class="n">T</div><div class="t">训练 2013–2015</div><div class="d">{ctx['train_shape'][0]:,} 条</div></div>
  <div class="step"><div class="n">V</div><div class="t">验证 2016</div><div class="d">{ctx['valid_shape'][0]:,} 条</div></div>
  <div class="step"><div class="n">O</div><div class="t">OOT 2017–2018</div><div class="d">{ctx['oot_shape'][0]:,} 条</div></div>
</div>
<p>训练期坏客户率 <strong>{metrics['train']['bad_rate']:.2%}</strong>，验证期 <strong>{metrics['valid']['bad_rate']:.2%}</strong>，OOT 期 <strong>{oot['bad_rate']:.2%}</strong>。训练/验证/OOT 的风险水平差异、客群结构变化与 OOT 删失，共同构成了一次真实的“模型上线后漂移”演练。</p>
<h3>2.3 泄露变量排查清单</h3>
<p>信用风险建模最常见的错误不是“模型不够复杂”，而是把贷后/未来信息带进特征。下表是本项目在建模前逐项核对过的清单；清单之外的新字段，也应先确认“申请时点是否可得”。</p>
{leakage_html}
<h4>通用排查步骤</h4>
<ol class="clean">{leakage_checklist_html}</ol>
<div class="note">本项目最终只使用申请时点的申请表、征信局与交易线变量；贷后现金流、贷后救助/和解、2016 才补齐的字段、LC 自身评级/定价均已剔除。时间字段 <code>issue_d</code> 只用于切分，<code>earliest_cr_line</code> 只用于派生放款时点的信用历史长度。</div>
''', "data")}

{_section("03", "特征工程与评分卡", f'''
<h3>3.1 特征与缺失</h3>
<p>候选特征覆盖申请金额/收入、征信查询、逾期历史、循环额度使用、账户数量、公共记录、信用历史长度等。对连续变量按训练集 0.1%/99.9% 分位截断；对“无逾期/无记录”类变量用 999 表示缺失并加缺失标记；类别变量做 one-hot。所有拟合只在 2013–2015 训练集上完成。</p>
{_image(figures.get("04_missing_train"), "训练集缺失率", "图 3-1 训练集缺失率最高的字段")}
{_image(figures.get("05_feature_distributions"), "入模变量分布", "图 3-2 部分入模变量的分布")}
<h3>3.2 IV 与 WOE</h3>
<p>连续变量先等频分箱，再合并小箱，并按估计方向做单调合并；二值变量直接两箱。WOE 定义为“坏客户占比 / 好客户占比”的对数，因此 WOE 越高风险越高，逻辑回归系数为正。</p>
{_image(figures.get("06_iv"), "IV 排名", "图 3-3 候选特征 IV 排名（前 22）")}
{iv_html}
{_image(figures.get("07_woe"), "WOE 分箱", "图 3-4 入模变量 WOE 分箱")}
<h3>3.3 入模变量</h3>
<p>按 IV ≥ {cfg.iv_threshold:.2f} 初筛，再按相关性 &lt; {cfg.corr_threshold:.2f} 去重，最后用 p 值与系数方向逐步删减。最终入模 <strong>{len(ctx['selected_features'])}</strong> 个变量：{selected_text}。</p>
{coef_html}
<p class="small muted">VIF 均低于常见阈值 5，入模变量之间不存在明显共线性。</p>
{vif_html}
<h3>3.4 标准评分卡</h3>
<div class="formula">
B = PDO / ln(2) = {card.factor:.4f}<br>
A = 基准分 − B × ln(基准好坏比) = {card.offset:.4f}<br>
Score = A − B × logit(PD)<br>
基础分 = A − B × β<sub>0</sub> = {card.base_points}<br>
某箱得分 = −B × β<sub>i</sub> × WOE<sub>ij</sub>
</div>
<p>评分刻度：基准分 {card.base_score}，PDO={card.pdo}，好坏比 {card.base_odds:g}:1（即基准 PD≈{1/(1+card.base_odds):.0%}）。基础分 {card.base_points} 是当前模型截距对应的分数；实际总分 = 基础分 + 各分箱得分。分数越高，违约概率越低。</p>
{scorecard_html}
''', "model")}

{_section("04", "OOT 验证", f'''
<h3>4.1 区分能力</h3>
<p>评分卡在 2017–2018 OOT 上 AUC 为 <strong>{oot['auc']:.4f}</strong>、KS 为 <strong>{oot['ks']:.4f}</strong>、Gini 为 <strong>{oot['gini']:.4f}</strong>。Bootstrap（{bootstrap.get('rounds', 0)} 次）估计的 OOT AUC 95% 置信区间为 <strong>[{bootstrap.get('lower', float('nan')):.4f}, {bootstrap.get('upper', float('nan')):.4f}]</strong>。OOT 低于 2016 验证集，主要来自 2017–2018 客群与账龄结构变化，而不是训练集过拟合。</p>
{metrics_html}
{_image(figures.get("08_roc_oot"), "OOT ROC", "图 4-1 OOT ROC：评分卡 vs LightGBM")}
{_image(figures.get("09_ks_oot"), "OOT KS", "图 4-2 OOT KS 曲线（评分卡）")}
<h3>4.2 校准与排序</h3>
<p>校准曲线越接近对角线越好；OOT 与验证集的绝对风险水平存在差异，评分卡预测概率可能低估，但排序能力仍需保持有效。提升表展示按风险从高到低取前 5%/10%/20%/30% 时能捕获多少坏客户。</p>
{_image(figures.get("20_calibration_curve"), "OOT 校准诊断", "图 4-3 OOT 校准曲线（95% CI）与分箱校准误差")}
{_image(figures.get("11_lift_oot"), "OOT 提升", "图 4-4 OOT 风险排序提升对比")}
{calibration_html}
{lift_html}
<h3>4.3 校准指标与解读</h3>
<p>仅看 AUC/KS 不能回答“预测 PD 是否可直接用于定价/拨备”。下表补充校准截距、校准斜率、ECE（期望校准误差）、MCE（最大校准误差）和 Brier。校准斜率理想为 1，校准截距理想为 0；ECE/MCE 越小越好。</p>
{cal_metrics_html}
<div class="warn"><strong>校准结论：</strong>评分卡 OOT 平均预测 PD 约 {cal_metrics['scorecard']['mean_pred']:.2%}，实际坏账率 {cal_metrics['scorecard']['actual']:.2%}；校准截距约 {cal_metrics['scorecard']['calibration_intercept']:.2f}、斜率约 {cal_metrics['scorecard']['calibration_slope']:.2f}，说明绝对概率被系统性低估。LightGBM 的平均预测 PD 为 {cal_metrics['lgbm']['mean_pred']:.2%}、校准斜率 {cal_metrics['lgbm']['calibration_slope']:.2f}。若模型要直接输出 PD 用于定价/拨备，应先做截距校准或 isotonic/Platt 校准；若只用于排序/审批阈值，则需用批准坏账率监控替代“PD 绝对值”假设。</div>
<h3>4.4 评分分布与等级</h3>
{_image(figures.get("15_score_distribution"), "OOT 评分分布", "图 4-5 OOT 评分分布")}
{_image(figures.get("16_grade"), "OOT 等级坏账率", "图 4-6 OOT 各评分等级坏客户率")}
{grade_html}
<p>下表同时对比各等级的<strong>平均预测 PD</strong>与<strong>实际坏客户率</strong>。OOT 全样本平均预测 PD 约 {oot['mean_pd']:.2%}，实际坏账率 {oot['bad_rate']:.2%}，说明 2017–2018 年模型存在明显的“绝对概率低估”，但等级排序仍然有效。</p>
{grade_cal_html}
''', "oot")}

{_section("05", "PSI 与 Vintage", f'''
<h3>5.1 Score PSI</h3>
<p>PSI 衡量当前样本与训练集在分数分布上的差异。本项目以 2013–2015 训练集为参照：训练→验证 PSI <strong>{ctx['psi_valid']:.4f}</strong>，训练→OOT PSI <strong>{ctx['psi_oot']:.4f}</strong>，均低于 0.10 的稳定阈值。这说明评分分布整体稳定，但 2017–2018 的客群/账龄变化已经可以在季度维度观察到。</p>
<div class="warn"><strong>PSI 稳定 ≠ 校准稳定：</strong>2017–2018 OOT 平均预测 PD 为 {oot['mean_pd']:.2%}，实际坏账率为 {oot['bad_rate']:.2%}，模型明显低估绝对风险。Score PSI 低只说明“分数分布形状”稳定，不能替代校准曲线、分分数段实际坏账率与批准坏账率监控。</div>
{_image(figures.get("12_psi_quarter"), "季度 PSI", "图 5-1 评分 PSI 按放款季度（参照训练集）")}
{psi_quarter_html}
<h3>5.2 特征 PSI</h3>
<p>特征 PSI 用于定位漂移来源。下表按 OOT PSI 从高到低列出前 18 个入模变量；单特征 PSI &gt; {cfg.psi_amber:.2f} 需要优先排查口径、渠道或客群变化。</p>
{_image(figures.get("13_psi_features"), "特征 PSI", "图 5-2 OOT 特征 PSI 最高的变量")}
{psi_features_html}
<h3>5.3 Vintage 分析</h3>
<p>Vintage 按放款季度拆分 cohort，观察已终态坏账率与 Current/Late 构成。2017–2018 年 Current 占比显著上升，说明近期 cohort 尚未成熟；解读坏账率时必须同时看成熟度，不能把“未到期”当成“低风险”。</p>
{_image(figures.get("02_vintage"), "Vintage", "图 5-3 Vintage：坏账率与贷款状态构成")}
{vintage_html}
<div class="note">监控建议：对坏账率/PSI 的评估应优先使用 <strong>同账龄（months on book）</strong> 口径；本数据没有逐月还款表现快照，因此报告中的 Vintage 坏账率是“已终态样本”的 cohort 坏账率，适合看趋势和相对变化。</div>
''', "psi")}

{_section("06", "阈值扫描", f'''
<h3>6.1 业务阈值如何选</h3>
<p>评分卡输出 PD 后，需要把连续概率转成“批准/拒绝”。本项目在 <strong>2016 验证集</strong>上扫描 PD 阈值（低于阈值批准），以“批准人群坏账率 ≤ {cfg.threshold_target_bad_rate:.0%} 且批准率尽量高”为目标选出阈值，再原样应用到 2017–2018 OOT，避免用 OOT 调参。</p>
{_image(figures.get("14_threshold_scan"), "阈值扫描", "图 6-1 阈值扫描：批准率、批准坏账率、坏客户捕获率")}
{threshold_html}
<h3>6.2 选定阈值与 OOT 效果</h3>
<p>验证集选定 PD 阈值 <strong>{conf['threshold']:.2f}</strong>。在 2016 验证集上该阈值对应批准坏账率 {chosen['approved_bad_rate']:.2%}；原样应用到 2017–2018 OOT 后：批准率 <strong>{conf['approval_rate']:.1%}</strong>，批准人群坏账率 <strong>{conf['approved_bad_rate']:.1%}</strong>，坏客户捕获率 <strong>{conf['bad_capture']:.1%}</strong>。OOT 批准坏账率较验证集变化 <strong>{(conf['approved_bad_rate']-chosen['approved_bad_rate'])*100:+.2f}pp</strong>；固定阈值的跨期表现需要持续监控，不能假设长期有效。</p>
<div class="warn"><strong>阈值诊断：</strong>若只看 OOT 做事后诊断，PD 阈值需调整到约 {ctx['oot_diagnostic']['threshold']:.2f} 才能把批准坏账率控制在约 {ctx['oot_diagnostic']['approved_bad_rate']:.1%}，对应批准率约 {ctx['oot_diagnostic']['approval_rate']:.1%}、坏客户捕获率约 {ctx['oot_diagnostic']['bad_capture']:.1%}。这不是建议用 OOT 调阈值，而是说明上线后应监控“批准坏账率 vs 目标”并触发校准/收紧阈值；具体阈值必须由业务收益与风险容忍度共同决定。</div>
{conf_html}
{threshold_summary_html}
<div class="two-col">
  <div class="note"><strong>策略含义：</strong>阈值越严格，批准率越低、批准坏账率越低、坏客户捕获率越高。实际业务应结合获客成本、资金成本、LGD、收益率与监管要求，把“坏账率目标”换成“风险调整后收益最大”或“限额/定价组合优化”。</div>
  <div class="warn"><strong>不要只用 OOT 选阈值：</strong>OOT 只用于最终评估。若用 OOT 反复调阈值，会高估上线效果；本项目的阈值完全来自 2016 验证集。</div>
</div>
''', "threshold")}

{_section("07", "LightGBM 对照", f'''
<h3>7.1 模型设置</h3>
<p>LightGBM 使用与评分卡相同的预处理特征（数值 + 缺失标记 + one-hot），在 2013–2015 训练、2016 早停（AUC），2017–2018 OOT 只做最终评估。最佳迭代轮数 <strong>{ctx['lgbm_best_iteration']}</strong>。LightGBM 不限制变量数，因此能捕捉非线性与交互；评分卡则保留可解释的 WOE/整数分。</p>
{_image(figures.get("08_roc_oot"), "ROC 对比", "图 7-1 OOT ROC：评分卡 vs LightGBM")}
{_image(figures.get("17_lgbm_importance"), "LightGBM 重要性", "图 7-2 LightGBM 特征重要性（Gain，前 20）")}
{lgbm_imp_html}
<h3>7.2 指标对比</h3>
{comparison_html}
{_image(figures.get("18_model_comparison"), "模型对比", "图 7-3 OOT AUC / KS 对比")}
<h3>7.3 Swap-set 分析</h3>
<p>Swap-set 回答一个业务问题：<strong>在相同批准率/批准人数下，换模型会“换掉”哪些人？</strong>本项目按 OOT 批准人数对齐（取 PD 最低的相同人数），比较评分卡与 LightGBM 的批准集合差异。Swap-in 指仅 LightGBM 批准的人群，Swap-out 指仅评分卡批准的人群；理想情况下 Swap-in 的坏客户率应低于 Swap-out，从而实现“同批准率下少放坏客户”。</p>
{swap_compare_html}
{swap_flow_html}
{swap_groups_html}
{_image(figures.get("19_swap_set"), "Swap-set", "图 7-4 Swap-set：同批准率下两模型批准集合差异")}
<div class="note"><strong>解读：</strong>两模型批准集合重合率约 {swap['overlap_rate']:.1%}；LightGBM 换入 {swap['swap_in_n']:,} 人（坏客户率 {swap['swap_in_bad_rate']:.2%}），换出 {swap['swap_out_n']:,} 人（坏客户率 {swap['swap_out_bad_rate']:.2%}），净减少坏客户约 <strong>{swap['net_bad_avoided']:,}</strong>。这说明 LightGBM 的增量主要来自“在相同批准规模下重新排序”，而不是简单扩大批准。若业务目标是最大化通过率且风险可控，可把 LightGBM 作为挑战者/辅助排序；若目标是可解释审批，则保留评分卡并用 LightGBM 做 Swap-set 监控。</div>
<h3>7.4 LightGBM 可解释性代价量化</h3>
<p>LightGBM 的 AUC 提升不是免费的。下表把“可解释性代价”拆成可量化维度：变量数、单笔评分计算量、模型结构、单笔解释方式、单调性、部署审计、文件大小，以及每单位复杂度换来的 AUC 增量。这样可以在“业务收益 vs 解释/治理成本”之间做显式权衡，而不是只说“LightGBM 更准”。</p>
{interp_html}
<div class="warn"><strong>代价结论：</strong>LightGBM 用 <strong>{interp['feature_ratio']:.1f}×</strong> 的变量数、约 <strong>{interp['scoring_ops_ratio']:.0f}×</strong> 的单笔评分计算量（按节点比较次数计：评分卡 {interp['scorecard_lookups_per_score']:.0f} 次分箱查表 vs LightGBM 约 {interp['lgbm_node_comparisons_per_score']:,.0f} 次节点比较）、约 <strong>{interp['model_size_ratio']:.0f}×</strong> 的模型文件，换来 OOT AUC <strong>+{interp['auc_gain']:.4f}</strong>。其中 <strong>+{interp['auc_gain_model_form']:.4f}</strong> 来自模型形式（同样 12 个变量换成树模型），<strong>+{interp['auc_gain_feature_count']:.4f}</strong> 来自变量数量（12 → {interp['lgbm_features']} 个）。如果监管/业务要求逐笔给出可复核的“原因码”，评分卡的 {interp['scorecard_features']} 项整数分加总仍然是更优选择；LightGBM 更适合做挑战者、辅助排序、或经 SHAP 解释后的二线模型。若要把 LightGBM 上线，必须额外建设解释器、特征管道版本管理和更严格的变更审计。</div>
<div class="note"><strong>结论：</strong>LightGBM 的 OOT AUC 比评分卡高约 <strong>{(lgbm_oot['auc']-oot['auc']):.4f}</strong>，说明非线性/交互确有增量；但评分卡在可解释性、稳定性、部署与监管沟通上更优。建议生产采用“评分卡为主 + LightGBM 挑战者/辅助排序”的双轨模式。</div>
''', "lgbm")}

{_section("08", "上线后监控方案", f'''
<h3>8.1 监控目标</h3>
<p>监控不是“看 AUC 有没有降”，而是分层判断：数据质量 → 分数分布 → 区分能力 → 资产质量。只有确认不是数据/口径问题后，才考虑调阈值、校准或重训。</p>
<h3>8.2 PSI 监控阈值</h3>
{psi_threshold_html}
<h3>8.3 监控指标矩阵</h3>
{monitoring_metrics_html}
<h3>8.4 预警流程</h3>
<ol class="clean">{alert_flow_html}</ol>
<h3>8.5 假设性报警案例（演练）</h3>
<p>以下案例用于校验“报警后到底先查什么”。真实上线时可把案例中的指标替换为当日监控值，按同样顺序执行。</p>
{cases_html}
<h3>8.6 重训触发条件</h3>
<ul class="clean">
  <li>评分 PSI &gt; {cfg.psi_amber:.2f} 连续 2 个月，或单月 &gt; 0.40。</li>
  <li>OOT AUC 较基线下降 &gt; {cfg.auc_drop_red:.2f}，或 KS 下降 &gt; {cfg.ks_drop_red:.2f}。</li>
  <li>同账龄 Vintage 坏账率连续 2 个 cohort 相对恶化 &gt; {cfg.bad_rate_rel_red:.0%}。</li>
  <li>批准率/拒绝率相对基线偏移 &gt; 10pp，且无法用策略调整解释。</li>
  <li>关键特征口径、数据源、进件政策发生重大变化；或模型有效期到期。</li>
</ul>
<h3>8.7 本项目监控基线</h3>
{baseline_html}
<div class="warn">2017–2018 OOT 的终态样本占比约 {1-ctx['current_share_oot']:.1%}，属于“早成熟” cohort。其绝对坏账率会随账龄继续上升，因此更适合看排序能力与 PSI；绝对坏账率请结合同账龄 vintage 曲线。</div>
<p class="small muted">完整可执行版本见 <code>outputs/monitoring_plan.md</code>。</p>
''', "monitoring")}

{_section("09", "复现说明", f'''
<h3>9.1 目录结构</h3>
<pre class="code"><code>lendingclub_scorecard_2018/
├── src/lendingclub_scorecard/   # 数据、分箱、模型、评估、报告、监控
├── scripts/run_pipeline.py      # 一键运行
├── scripts/score_csv.py         # 对新数据批量打分
├── tests/
├── outputs/
│   ├── figures/                 # 20 张报告图
│   ├── tables/                  # IV、分箱、指标、PSI、Vintage、阈值
│   ├── predictions/oot_scores.csv
│   ├── scorecard.csv / .json
│   ├── report.html
│   └── monitoring_plan.md
├── requirements.txt
└── README.md</code></pre>
<h3>9.2 运行命令</h3>
<pre class="code"><code>cd D:\\DATAsupport\\lendingclub_scorecard_2018
.venv\\Scripts\\python.exe scripts\\run_pipeline.py

# 强制重建数据缓存
.venv\\Scripts\\python.exe scripts\\run_pipeline.py --refresh-data</code></pre>
<p class="small muted">本次运行耗时约 {ctx['runtime_seconds']:.1f} 秒。所有随机过程固定 random_state={cfg.random_state}。</p>
''', "repro")}

{_section("10", "数据口径与限制", f'''
<h3>10.1 数据来源</h3>
<p>数据来自 LendingClub 公开贷款数据（LoanStats）与配套 <code>LCDataDictionary.xlsx</code>。样本覆盖 2012–2018 年放款；建模只使用 2013–2018 年已终态贷款，2012 年仅用于 Vintage 客群结构展示。</p>
<h3>10.2 重要口径</h3>
<ul class="clean">
  <li><strong>FICO 字段：</strong>新数据包含 <code>fico_range_low/high</code>（完整），但主模型为保持与上一版特征口径一致，没有把 FICO 入模；FICO 是申请时点可得的合法变量，后续可单独纳入做增量评估。因此本报告 AUC 不应与“含 FICO”的外部模型直接比较。</li>
  <li><strong>不使用 LC 评级/定价：</strong>主模型不使用 grade、sub_grade、int_rate，避免复用 LendingClub 自身的风险判断。</li>
  <li><strong>只使用终态样本：</strong>Current/Late/Grace 不进入训练与 OOT 评估；这会带来样本选择偏差，尤其影响 2018 的绝对坏账率。</li>
  <li><strong>2017–2018 OOT 的删失偏差：</strong>OOT cohort 的 Current 占比约 {ctx['current_share_oot']:.1%}，其中 2018 年约 {ctx['current_share_2018']:.1%}；报告强调排序与漂移，而非绝对校准。</li>
  <li><strong>变量可用性：</strong>2016 才补齐的 il_util、all_util、open_rv_12m、inq_last_12m 等在训练期整列缺失，已剔除，避免“未来字段”泄漏。</li>
</ul>
<h3>10.3 使用限制</h3>
<div class="warn">本项目用于学习、建模复现与监控方案演示，不构成真实授信建议。上线前需补充合规审查、模型验证、压力测试、公平性评估、拒绝推断与人工复核流程；上线后应按本报告监控方案持续跟踪，并定期重训。</div>
''', "notes")}

{_section("11", "项目局限与后续计划", f'''
<p class="lead">这一节主动说明项目“没有做什么”，避免把离线实验结果误读为可直接上线的生产系统。以下局限不影响本项目的教学/复现价值，但会直接影响真实授信决策。</p>
<h3>11.1 拒绝推断（Reject Inference）</h3>
<p>本项目只使用 <strong>已批准且已终态</strong> 的贷款，未使用被拒绝申请（RejectStats），也没有做拒绝推断。由于批准政策本身会筛掉一部分高风险人群，训练样本是“被政策选择过”的子集，模型学到的是“批准人群中的风险排序”，而不是全申请人群的真实违约率。这会导致：</p>
<ul class="clean">
  <li>对“如果批准被拒绝人群”的坏账率估计偏乐观；</li>
  <li>阈值/批准率外推到新客群时可能失真；</li>
  <li>OOT 的绝对坏账率不能直接当作全申请人群的预期损失。</li>
</ul>
<p><strong>后续计划：</strong>引入 RejectStats/拒绝原因；用 fuzzy augmentation、parceling 或 Heckman 两阶段做拒绝推断；对推断前后的通过率与坏账率做敏感性分析。</p>
<h3>11.2 校准（Calibration）</h3>
<p>评分卡在 2017–2018 OOT 的平均预测 PD 为 <strong>{oot['mean_pd']:.2%}</strong>，实际坏账率 <strong>{oot['bad_rate']:.2%}</strong>；校准截距约 <strong>{cal_metrics['scorecard']['calibration_intercept']:.2f}</strong>、斜率约 <strong>{cal_metrics['scorecard']['calibration_slope']:.2f}</strong>，说明绝对概率被系统性低估。当前模型只做了“排序/阈值”评估，没有做上线前的概率校准。</p>
<p><strong>后续计划：</strong>在独立校准集上做截距校准（calibration-in-the-large）、Platt/isotonic 校准；上线后按月监控“平均预测 PD vs 实际坏账率”和分数段校准；若只用于排序，应在制度上禁止直接用 PD 做定价/拨备，必须经过校准层。</p>
<h3>11.3 数据年份与账龄</h3>
<ul class="clean">
  <li>样本覆盖 <strong>2012–2018</strong>，但 2018 年大量贷款尚未成熟，且未纳入 2019 及更近周期；宏观经济、利率、客群结构可能已变化。</li>
  <li>2017–2018 OOT 的 Current 占比约 <strong>{ctx['current_share_oot']:.1%}</strong>，终态样本早于完整账龄，绝对坏账率会随成熟度上升。</li>
  <li>本数据没有逐月还款表现快照，Vintage 只能做“已终态 cohort 坏账率”，无法画真正的 months-on-book 累计违约曲线。</li>
  <li>没有宏观变量（失业率、利率、区域房价等），无法做压力测试与宏观情景分析。</li>
</ul>
<p><strong>后续计划：</strong>扩展至 2019+ 及最新放款；接入逐月表现快照，构建真正的 Vintage 曲线；引入宏观变量与压力测试；按季度做时间外滚动验证（rolling OOT）。</p>
<h3>11.4 其他限制</h3>
<ul class="clean">
  <li><strong>特征限制：</strong>新数据含 FICO，但主模型未使用；未接入外部征信/ bureau/另类数据；无收入验证、设备/行为数据。</li>
  <li><strong>策略限制：</strong>阈值目标是“批准坏账率 ≤15%”，未做 LGD/EAD、资金成本、获客成本、定价联动的风险调整收益优化。</li>
  <li><strong>治理限制：</strong>未做公平性/反歧视评估、压力测试、模型风险评级、独立验证与监管报备。</li>
  <li><strong>工程限制：</strong>未真正部署实时评分服务、特征平台、自动监控告警与模型回滚；当前监控方案是“可执行方案”，不是已接入的生产系统。</li>
  <li><strong>模型限制：</strong>评分卡使用线性 WOE，可能漏掉非线性/交互；LightGBM 虽更强，但未做超参数搜索、稳定性选择和可解释性约束。</li>
</ul>
<h3>11.5 后续计划（按优先级）</h3>
<ol class="clean">
  <li><strong>P0：</strong>补拒绝推断与独立校准层；把“平均预测 PD − 实际坏账率”纳入上线门槛。</li>
  <li><strong>P0：</strong>扩展数据年份与逐月表现，重建真实 Vintage/滚动 OOT；验证 2019+ 的稳定性。</li>
  <li><strong>P1：</strong>把阈值从“固定坏账率目标”升级为“风险调整收益最大”（LGD/EAD、价格、额度、获客成本）。</li>
  <li><strong>P1：</strong>建立评分卡 + LightGBM 的 champion-challenger 与 Swap-set 月度监控；设置自动回滚条件。</li>
  <li><strong>P2：</strong>公平性/压力测试/模型风险治理；接入特征平台与实时评分服务；完善监控告警与审计留痕。</li>
  <li><strong>P2：</strong>探索非线性评分卡（分段/交互）或在可解释约束下蒸馏 LightGBM。</li>
</ol>
<div class="warn"><strong>结论：</strong>当前版本适合作为“评分卡方法论 + OOT/PSI/Vintage/阈值/监控”的完整样板，但不应直接用于真实授信。上线前至少需要：拒绝推断、概率校准、扩展时间外验证、独立模型验证与合规审批。</div>
''', "limitations")}

<footer>
  LendingClub 信用风险评分卡项目 · 自动生成于 {generated_at}<br>
  数据来源：LendingClub 公开 LoanStats 数据 · 仅用于建模演示
</footer>
</div>
</body>
</html>
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


