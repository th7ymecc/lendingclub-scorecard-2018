"""Build the v2 review-response report from generated artifacts.

This report deliberately does not re-fit anything: it reads the artifacts produced
by ``run_pipeline.py`` and presents the MOB12 / lifetime / calibration / ablation
results in one place.  It exists because the older ``report.html`` used the
terminal-only label and therefore overstated the approved bad rate.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TAB = OUT / "tables"
FIG = OUT / "figures"


def pct(x, digits: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return f"{float(x):.{digits}%}"


def num(x, digits: int = 4) -> str:
    if pd.isna(x):
        return "—"
    return f"{float(x):,.{digits}f}"


def table(df: pd.DataFrame, cols=None, headers=None, formats=None) -> str:
    data = df.copy()
    if cols is not None:
        data = data[cols]
    if headers:
        data = data.rename(columns=dict(zip(cols or data.columns, headers)))
    if formats:
        for col, func in formats.items():
            if col in data.columns:
                data[col] = data[col].map(func)
    return data.to_html(index=False, border=0, classes="data", escape=True)


def image(name: str, caption: str) -> str:
    path = FIG / name
    if not path.exists():
        return f"<p class='muted'>[missing figure: {html.escape(name)}]</p>"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f"<figure><img src='data:image/png;base64,{b64}' alt='{html.escape(caption)}'>"
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


meta = json.loads((OUT / "run_metadata.json").read_text(encoding="utf-8"))
dq = pd.read_csv(TAB / "data_quality.csv")
m_short = pd.read_csv(TAB / "metrics_scorecard.csv")
m_life = pd.read_csv(TAB / "metrics_scorecard_lifetime.csv")
cal = pd.read_csv(TAB / "calibrator_comparison_oot.csv")
thr = pd.read_csv(TAB / "threshold_summary.csv")
sens = pd.read_csv(TAB / "threshold_sensitivity.csv")
abl = pd.read_csv(TAB / "sensitivity_ablation.csv")
iv = pd.read_csv(TAB / "sensitivity_iv.csv")
attr = pd.read_csv(TAB / "model_attribution_oot.csv")
vif0 = pd.read_csv(TAB / "vif_initial.csv")
vif1 = pd.read_csv(TAB / "vif.csv")
sel = pd.read_csv(TAB / "selection_log.csv")
term = pd.read_csv(TAB / "emergence_by_term.csv")
psi = pd.read_csv(TAB / "psi_features_oot.csv")
swap = pd.read_csv(TAB / "swap_set_summary.csv").iloc[0]

# Recompute grade bands on lifetime PD, not on the integer score.  The old
# score-based grade cutoffs put 90% of OOT in A because the model's vintage PD
# scale is much lower than the lifetime PD band definitions.
scores = pd.read_csv(
    OUT / "predictions" / "oot_scores.csv",
    usecols=["score", "prob_lifetime", "target_lifetime"],
)
scores["grade"] = pd.cut(
    scores["prob_lifetime"],
    bins=[-np.inf, 0.08, 0.12, 0.18, 0.25, np.inf],
    labels=list("ABCDE"),
)
grade = (
    scores.groupby("grade", observed=True)
    .agg(
        count=("target_lifetime", "size"),
        bad=("target_lifetime", "sum"),
        bad_rate=("target_lifetime", "mean"),
        avg_pred_pd=("prob_lifetime", "mean"),
        avg_score=("score", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    )
    .reset_index()
)
grade["share"] = grade["count"] / len(scores)

# Headline metrics
oot = meta["metrics"]["scorecard"]["oot"]
valid = meta["metrics"]["scorecard"]["valid"]
train = meta["metrics"]["scorecard"]["train"]
lgbm_oot = meta["metrics"]["lgbm"]["oot"]
life = meta["metrics_lifetime"]["oot"]
data = meta["data"]
cal_meta = meta["calibration"]
thr_meta = meta["threshold"]
interp = meta["interpretability_cost"]

headline = f"""
<div class="hero">
  <div class="eyebrow">LendingClub 2012–2018 · v2 修订报告</div>
  <h1>信用评分卡：MOB12 口径、概率校准与敏感性</h1>
  <p>这份报告替代 2026-10-18 的终态口径 v1 报告。v1 的“OOT 批准坏账率 14.39%”来自只保留终态贷款的删失样本，已废弃；本版统一使用固定 12 个月表现窗口（MOB12），再把 PD 换算回终身口径做审批决策。</p>
  <div class="badges">
    <span>训练 2013–2015 · 791,536</span>
    <span>验证 2016 · 434,350</span>
    <span>OOT 2017–2018Q1 · 550,697</span>
    <span>MOB12 bad rate · 5.71%</span>
    <span>终身 PD · 20.30%</span>
    <span>终身批准坏账率 · 13.74%</span>
  </div>
</div>
<div class="warn"><strong>一句话结论：</strong>删失偏差已通过 MOB12 固定窗口消除；校准后终身 PD 为 {cal_meta['oot_mean_pd_lifetime']:.2%}，OOT 终身坏账率为 {cal_meta['oot_observed_lifetime_bad_rate']:.2%}。阈值 0.27 在 OOT 上批准率 {thr_meta['approval_rate_oot']:.2%}、批准人群终身坏账率 {thr_meta['approved_bad_rate_oot']:.2%}，低于 15% 目标，但这一结论只对“终身加权口径”成立，不能与旧版终态 14.39% 直接比较。</div>
"""

response = pd.DataFrame(
    [
        ("① OOT 删失偏差", "已修", "改为 MOB12 固定窗口；OOT 只到 2018-03，2018Q2+ 观察期不足 12 个月被排除；再用期限分段 e 换算终身 PD。", "终身批准坏账率 13.74%，旧 14.39% 基线废弃"),
        ("② 阈值敏感性", "已补", "目标坏账率 12%/15%/18% 三档扫描；阈值统一在终身 PD 上选。", "0.45 / 0.90 / 0.90；15%/18% 目标非绑定，说明目标偏松"),
        ("③ FICO 缺失", "已量化", "同一 LightGBM、同一切分，只加 fico_mid；FICO 单变量 IV=0.100。", "OOT AUC +0.0036（0.6940→0.6976）"),
        ("④ term 业务耦合", "已解释", "term 是产品/期限变量；用 36/60 期分别估 e 与 lifetime bad rate，报告披露耦合。", "60 期终身坏账 25.8% vs 36 期 12.7%"),
        ("⑤ VIF 幸存者偏差", "已补", "剔除前初始集和最终集双口径 VIF；selection_log 记录剔除时 VIF/相关/单变量方向。", "初始最大 VIF 2.94，final 2.38"),
        ("⑥ LightGBM 不公平", "已补", "增加 LightGBM-12（只使用评分卡 14 变量）与 full 对照。", "模型形式 +0.0056；变量数量 +0.0286"),
        ("⑦ 代码瑕疵", "已清", "删除 read-but-unused 的 grade 逻辑；修正 2012–2016 残留 docstring；PSI 明确用 score。", "grade 用于消融，PSI 口径在报告说明"),
        ("⑧ 校准/后续", "部分完成", "Platt/isotonic 已在验证集拟合、OOT 只 transform；拒绝推断/公平性仍未做。", "Platt ECE 1.15%→0.41%"),
    ],
    columns=["评价点", "状态", "本版处理", "关键结果"],
)

scale = pd.DataFrame(
    [
        ("MOB12 vintage（建模标签）", valid["bad_rate"], oot["bad_rate"], oot["auc"], oot["mean_pd"]),
        ("终身（业务决策口径）", np.nan, cal_meta["oot_observed_lifetime_bad_rate"], life["auc"], cal_meta["oot_mean_pd_lifetime"]),
    ],
    columns=["口径", "验证实际坏账率", "OOT 实际坏账率", "OOT AUC", "OOT 平均 PD"],
)

dq_view = dq[
    [
        "issue_year",
        "total",
        "resolved_rate",
        "current_rate",
        "legacy_bad_rate",
        "vintage_mob12_bad_rate",
        "evaluable",
        "evaluable_rate",
    ]
].copy()
dq_view.columns = [
    "放款年",
    "全部贷款",
    "终态率（旧口径）",
    "Current 占比",
    "旧口径坏账率",
    "MOB12 坏账率",
    "MOB12 可评估",
    "可评估率",
]

metric_rows = []
for name, row in [("训练", train), ("验证", valid), ("OOT", oot)]:
    metric_rows.append(
        {
            "口径": "MOB12",
            "数据集": name,
            "AUC": row["auc"],
            "KS": row["ks"],
            "Gini": row["gini"],
            "PR-AUC": row["pr_auc"],
            "Brier": row["brier"],
            "平均 PD": row["mean_pd"],
            "实际坏账率": row["bad_rate"],
        }
    )
for name, row in [("验证", m_life[m_life["数据集"] == "valid"].iloc[0]), ("OOT", m_life[m_life["数据集"] == "oot"].iloc[0])]:
    metric_rows.append(
        {
            "口径": "终身",
            "数据集": name,
            "AUC": row["auc"],
            "KS": row["ks"],
            "Gini": row["gini"],
            "PR-AUC": row["pr_auc"],
            "Brier": row["brier"],
            "平均 PD": row["mean_pd"],
            "实际坏账率": cal_meta["oot_observed_lifetime_bad_rate"] if name == "OOT" else np.nan,
        }
    )
metrics_view = pd.DataFrame(metric_rows)

cal_view = cal.copy()
cal_view["calibrator"] = cal_view["calibrator"].map(
    {"raw": "原始 PD", "platt": "Platt", "isotonic": "Isotonic"}
)

thr_view = thr.copy()
sens_view = sens.copy()
abl_view = abl.copy()
iv_view = iv.copy()
attr_view = attr.copy()

vif0_view = vif0.sort_values("VIF", ascending=False).head(10).copy()
vif1_view = vif1.sort_values("VIF", ascending=False).head(8).copy()
sel_view = sel.copy()

interp_rows = [
    ("模型变量数", f"{interp['scorecard_features']} 个", f"{interp['lgbm_features']} 个", f"{interp['feature_ratio']:.2f}×"),
    ("模型结构", f"1 个线性方程，{interp['scorecard_bins']} 个分箱", f"{interp['lgbm_trees']:,} 棵树，{interp['lgbm_leaves']:,} 个叶子", f"平均深度 {interp['lgbm_avg_depth']:.2f}，最大 {interp['lgbm_max_depth']}"),
    ("单笔计算量", f"{interp['scorecard_lookups_per_score']:.0f} 次查表", f"{interp['lgbm_node_comparisons_per_score']:,.0f} 次节点比较", f"{interp['scoring_ops_ratio']:.0f}×"),
    ("模型文件", f"{interp['scorecard_json_kb']:.1f} KB", f"{interp['lgbm_model_mb']:.2f} MB", f"{interp['model_size_ratio']:.0f}×"),
    ("OOT AUC 增量", "基准", f"+{interp['auc_gain']:.4f}", f"模型形式 +{interp['auc_gain_model_form']:.4f}；变量数量 +{interp['auc_gain_feature_count']:.4f}"),
    ("解释方式", "整数分逐项相加，严格可加", "需 SHAP/树路径，非严格可加", "需额外解释器"),
]
interp_view = pd.DataFrame(interp_rows, columns=["维度", "WOE 评分卡", "LightGBM", "差异"])

monitor = pd.DataFrame(
    [
        ("OOT 窗口", "2017-01 ~ 2018-03（统一 MOB12）"),
        ("OOT 样本", f"{data['oot_rows']:,}"),
        ("MOB12 坏账率", f"{data['bad_rate_oot']:.2%}"),
        ("终身坏账率（加权）", f"{cal_meta['oot_observed_lifetime_bad_rate']:.2%}"),
        ("MOB12 平均预测 PD", f"{oot['mean_pd']:.2%}"),
        ("校准后 MOB12 平均 PD", f"{cal_meta['oot_mean_pd_calibrated']:.2%}"),
        ("终身平均 PD", f"{cal_meta['oot_mean_pd_lifetime']:.2%}"),
        ("选定阈值（终身 PD）", f"{thr_meta['selected']:.2f}"),
        ("OOT 批准率", f"{thr_meta['approval_rate_oot']:.2%}"),
        ("OOT 批准人群终身坏账率", f"{thr_meta['approved_bad_rate_oot']:.2%}"),
        ("OOT 坏客户捕获率", f"{thr_meta['bad_capture_oot']:.2%}"),
        ("Score PSI", f"{meta['psi']['oot']:.4f}"),
        ("OOT Current 占比", f"{data['current_share_oot']:.1%}（2018 年 {data['current_share_2018']:.1%}）"),
    ],
    columns=["监控项", "基线 / 口径"],
)

fig_cal = image("20_calibration_curve.png", "OOT 校准曲线与分箱误差（95% Wilson CI）")
fig_vintage = image("02_vintage.png", "Vintage：MOB12 与旧终态口径对照")
fig_roc = image("08_roc_oot.png", "OOT ROC：评分卡 vs LightGBM")
fig_thr = image("14_threshold_scan.png", "阈值扫描：批准率、批准坏账率、捕获率")
fig_swap = image("19_swap_set.png", "Swap-set：同批准率下两模型批准集合差异")

css = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;color:#1d2b3a;background:#f4f7fb;margin:0;line-height:1.65}
.wrap{max-width:1180px;margin:0 auto;padding:26px 20px 60px}
.hero{background:linear-gradient(135deg,#102a43,#1f4e79);color:#fff;padding:30px 32px;border-radius:14px;margin-bottom:18px}
.hero h1{font-size:28px;margin:8px 0 10px;line-height:1.3}
.hero p{margin:0;color:#dbe9f7;max-width:980px}
.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#9fc3e6}
.badges{margin-top:16px;display:flex;flex-wrap:wrap;gap:8px}
.badges span{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);padding:5px 9px;border-radius:999px;font-size:12px}
h2{font-size:22px;margin:30px 0 10px;color:#102a43;border-bottom:2px solid #d9e5f2;padding-bottom:6px}
h3{font-size:16px;margin:20px 0 8px;color:#173f5f}
p{margin:8px 0}
.muted{color:#64748b;font-size:12.5px}
.warn{background:#fff6e5;border-left:4px solid #d97706;padding:12px 15px;border-radius:8px;margin:14px 0}
.note{background:#eef6ff;border-left:4px solid #2b6cb0;padding:12px 15px;border-radius:8px;margin:14px 0}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}
.kpi{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:23px;color:#173f5f}
.kpi span{font-size:12px;color:#64748b}
table.data{width:100%;border-collapse:collapse;background:#fff;font-size:12.6px;margin:10px 0 16px;border:1px solid #e2e8f0}
table.data th{background:#eaf2fa;color:#173f5f;text-align:left;padding:7px 8px;border-bottom:1px solid #d8e4f0;white-space:nowrap}
table.data td{padding:6px 8px;border-bottom:1px solid #edf2f7;vertical-align:top}
table.data tr:nth-child(even) td{background:#fbfdff}
figure{margin:16px 0;background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:10px}
figure img{width:100%;height:auto;display:block}
figcaption{font-size:12px;color:#64748b;text-align:center;margin-top:6px}
ul,ol{margin:8px 0 8px 22px}
li{margin:4px 0}
.footer{margin-top:32px;color:#64748b;font-size:12px;border-top:1px solid #d9e5f2;padding-top:12px}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}
"""

html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LendingClub 2012–2018 评分卡 v2 · MOB12 / 校准 / 敏感性</title>
<style>{css}</style></head>
<body><div class="wrap">
{headline}
<div class="grid">
  <div class="kpi"><b>{data['train_rows']:,}</b><span>训练样本（MOB12）</span></div>
  <div class="kpi"><b>{oot['auc']:.4f}</b><span>评分卡 OOT AUC</span></div>
  <div class="kpi"><b>{thr_meta['approved_bad_rate_oot']:.2%}</b><span>OOT 批准终身坏账率</span></div>
  <div class="kpi"><b>{cal_meta['oot_mean_pd_lifetime']:.2%}</b><span>OOT 终身平均 PD</span></div>
</div>

<h2>1. 对评价的逐条回应</h2>
<p>这份 v2 不是“再加一段解释”，而是把标签、阈值、校准和对照实验重新对齐。旧报告的终态口径结论已明确作废。</p>
{table(response, headers=None)}

<h2>2. 口径修正：MOB12 固定窗口 + 终身换算</h2>
<p>旧版只保留 <code>Fully Paid / Charged Off / Default</code>，等于让每笔贷款停在自己的终态时点；2018 年进入 OOT 的只是最快结清的那批，晚期核销被系统性排除。本版定义：</p>
<ul>
<li><code>bad_flag_v12 = 1</code>：贷款终态为 Charged Off/Default，且核销发生在 <strong>MOB ≤ 12</strong>；</li>
<li><code>bad_flag_v12 = 0</code>：已观察满 12 个月，且未在 MOB12 前核销；</li>
<li><code>bad_flag_v12 = NaN</code>：观察不足 12 个月，直接剔除，不硬算。</li>
</ul>
<p>OOT 上界因此收到 <strong>2018-03-31</strong>（快照日 2019-03-31 减 12 个月）；2018Q2 之后放款的观察期不足，不能进入 MOB12 OOT。所有 cohort 截到同一时点后，坏账率才可跨期比较。</p>
{table(dq_view, formats={'全部贷款':lambda x:f'{int(x):,}','MOB12 可评估':lambda x:f'{int(x):,}','终态率（旧口径）':pct,'Current 占比':pct,'旧口径坏账率':pct,'MOB12 坏账率':pct,'可评估率':pct})}
<p>旧口径与 MOB12 的比值约 3.0–4.0×，说明“终态坏账率”不是终身坏账率；直接拿 14.39% 当监控基线会同时犯两个错：样本删失 + 口径混用。</p>
{table(term, headers=['期限','分层','成熟样本','终身坏账','MOB12 坏样本','终身坏账率','MOB12 坏账率','暴露比例 e','换算系数','校验终身率','误差'], formats={'成熟样本':lambda x:f'{int(x):,}','终身坏账':lambda x:f'{int(x):,}','MOB12 坏样本':lambda x:f'{int(x):,}','终身坏账率':pct,'MOB12 坏账率':pct,'暴露比例 e':pct,'换算系数':lambda x:f'{x:.2f}','校验终身率':pct,'误差':lambda x:f'{x:+.4f}'})}
<div class="note">终身换算在<strong>概率空间</strong>做除法：<code>p_lifetime = p_vintage / e(term)</code>，不是 logit 平移。36 期 e≈0.339，60 期 e≈0.250；若用总体 e=0.304 统一换算，长期限会被系统性低估。</div>
{fig_vintage}

<h2>3. 模型与 OOT 结果</h2>
<p>主模型仍是 WOE + Logistic，但标签改为 MOB12；评分卡 14 个变量。OOT AUC {oot['auc']:.4f}、KS {oot['ks']:.4f}，低于旧版终态口径的 0.6646 是正常的——标签更难、删失被消除，不应把旧数字当基准。</p>
{table(metrics_view, formats={'AUC':num,'KS':num,'Gini':num,'PR-AUC':num,'Brier':num,'平均 PD':pct,'实际坏账率':pct})}
{table(scale, formats={'验证实际坏账率':pct,'OOT 实际坏账率':pct,'OOT AUC':num,'OOT 平均 PD':pct})}
{fig_roc}
<p>终身口径的 AUC 为 {life['auc']:.4f}，与 MOB12 口径接近；说明期限换算主要改变 PD 水平，没有破坏排序。</p>

<h2>4. 概率校准</h2>
<p>校准器只在 2016 验证集 fit，OOT 只 transform。原始 PD 的 ECE 为 {cal.iloc[0]['ece']:.2%}，Platt 后降到 {cal.iloc[1]['ece']:.2%}（isotonic 为 {cal.iloc[2]['ece']:.2%}），说明“排序可用、绝对概率不可用”的问题被实质修正。</p>
{table(cal_view, headers=['校准器','平均预测 PD','实际坏账率','校准截距','校准斜率','ECE','MCE','Brier'], formats={'平均预测 PD':pct,'实际坏账率':pct,'校准截距':num,'校准斜率':num,'ECE':pct,'MCE':pct,'Brier':num})}
{fig_cal}
<h3>按终身 PD 重划等级</h3>
<p>旧版用整数分硬切 A–E，导致 90% 的 OOT 落在 A，且 A 段实际坏账 8.70% 越过 8% 定义。本版改为<strong>按终身 PD 划等级</strong>：A&lt;8%、B 8–12%、C 12–18%、D 18–25%、E≥25%。这样等级含义与业务 PD 一致，也避免“分数段定义”和“实际坏账率”互相打架。</p>
{table(grade, headers=['等级','样本数','坏样本数','实际坏账率','平均预测 PD','平均分','最低分','最高分','占比'], formats={'样本数':lambda x:f'{int(x):,}','坏样本数':lambda x:f'{x:,.0f}','实际坏账率':pct,'平均预测 PD':pct,'平均分':lambda x:f'{x:.1f}','最低分':lambda x:f'{x:.0f}','最高分':lambda x:f'{x:.0f}','占比':pct})}

<h2>5. 阈值与敏感性</h2>
<p>阈值统一在<strong>终身 PD</strong> 上选：目标批准坏账率 ≤15%，批准率 ≥50%，取满足条件的最大批准率。验证集选出 0.27；OOT 批准率 {thr_meta['approval_rate_oot']:.2%}，批准人群终身坏账率 {thr_meta['approved_bad_rate_oot']:.2%}，捕获率 {thr_meta['bad_capture_oot']:.2%}。</p>
{table(thr, headers=['模型','阈值来源','阈值','批准率','批准坏账率','捕获率','审阅'], formats={'阈值':lambda x:f'{x:.2f}','批准率':pct,'批准坏账率':pct,'捕获率':pct})}
<h3>目标坏账率 12% / 15% / 18% 敏感性</h3>
<p>15% 与 18% 两档都选到 0.90，并不是代码错误，而是目标在这版 PD 分布下<strong>不具约束力</strong>：拒绝 0.2% 的最高 PD 人群就能把批准坏账率压到 13.38%。这反而说明“15% 目标”对当前客群偏松，应结合收益/额度/定价重新设定，而不是把 0.90 当生产阈值。</p>
{table(sens_view, headers=['目标坏账率','最低批准率','阈值（终身 PD）','批准率','批准坏账率','捕获率','拒绝人群坏账率','可行'], formats={'目标坏账率':pct,'最低批准率':pct,'阈值（终身 PD）':lambda x:f'{x:.2f}','批准率':pct,'批准坏账率':pct,'捕获率':pct,'拒绝人群坏账率':pct})}
{fig_thr}
{fig_swap}

<h2>6. 消融：FICO、grade 与“同变量”LightGBM</h2>
<p>评价里说“FICO 是最强单变量、AUC 应该 0.72–0.78”。在本数据 + MOB12 标签下，FICO 的单变量 IV 只有 <strong>{iv.loc[iv.variable=='fico_mid','iv'].iloc[0]:.3f}</strong>，加入 LightGBM 只提升 OOT AUC <strong>+{abl.loc[abl.feature_set=='+fico','auc_vs_base'].iloc[0]:.4f}</strong>；真正强的是 LC 的 <code>grade</code>（IV {iv.loc[iv.variable=='grade_ord','iv'].iloc[0]:.3f}），但它属于平台自身评级，不能作为自建评分卡的特征。</p>
{table(abl_view, headers=['特征集','变量数','新增','OOT AUC','OOT KS','PR-AUC','相对 base','新增变量增益占比','最佳迭代','复用基线'], formats={'OOT AUC':num,'OOT KS':num,'PR-AUC':num,'相对 base':lambda x:f'{x:+.4f}','新增变量增益占比':pct})}
<p>“LightGBM 比评分卡强”也要拆开看：只用评分卡 14 个变量时，LightGBM-12 的 AUC 是 <strong>{attr.iloc[1]['oot_auc']:.4f}</strong>，只比评分卡高 {attr.iloc[1]['auc_vs_previous']:+.4f}；换到 82 变量才到 {attr.iloc[2]['oot_auc']:.4f}，变量数量的贡献是 {attr.iloc[2]['auc_vs_previous']:+.4f}。因此旧版“+0.043 全是非线性”的说法不成立，主要增量来自变量数量，而不是模型形式。</p>
{table(attr_view, headers=['模型','变量数','模型形式','OOT AUC','OOT KS','角色','相对上一行'], formats={'OOT AUC':num,'OOT KS':num,'相对上一行':lambda x:'—' if pd.isna(x) else f'{x:+.4f}'})}
{table(iv_view, headers=['变量','单变量 IV','训练覆盖','OOT 覆盖'], formats={'单变量 IV':num,'训练覆盖':pct,'OOT 覆盖':pct})}

<h2>7. 共线性、term 与可解释性代价</h2>
<p>VIF 不再只算“幸存者”：剔除前初始集最大 {vif0['VIF'].max():.2f}，最终集最大 {vif1['VIF'].max():.2f}。selection_log 记录每个被剔除变量在剔除时点的 VIF、与保留变量的最大相关、单变量方向；负系数不再是“黑箱”。</p>
{table(vif0_view, headers=['阶段','变量','中文名','VIF','后续剔除'], formats={'VIF':lambda x:f'{x:.2f}','后续剔除':lambda x:'是' if x else '否'})}
{table(sel_view, headers=['步骤','变量','中文名','p 值','系数','剔除时 VIF','最大相关','相关变量','单变量相关'], formats={'p 值':lambda x:f'{x:.3g}','系数':num,'剔除时 VIF':lambda x:f'{x:.2f}','最大相关':num,'单变量相关':num})}
<p><strong>term：</strong>60 期与 36 期的终身坏账率分别是 25.8% 与 12.7%，e 为 0.250 与 0.339。term 确实是最强变量之一，但它同时是产品/期限策略变量；生产上必须把 term 固定在申请时点，不能允许“改期限就改评分”的耦合。</p>
{table(interp_view, headers=['维度','WOE 评分卡','LightGBM','差异'])}
<div class="warn"><strong>可解释性代价的修正：</strong>旧版用“树数 / 变量数 = 2034/12”当计算量，口径不成立。本版改为树遍历的<strong>节点比较次数</strong>：单笔约 {interp['lgbm_node_comparisons_per_score']:,.0f} 次，对比评分卡 {interp['scorecard_lookups_per_score']:.0f} 次查表，约 {interp['scoring_ops_ratio']:.0f}×；同时给出文件大小 {interp['model_size_ratio']:.0f}× 与 AUC 增量 +{interp['auc_gain']:.4f}。</div>

<h2>8. 监控基线（替换旧版 14.39%）</h2>
<p>旧版监控基线把终态批准坏账率 14.39% 当成未来每月对比基准，这会系统性低估。本版基线明确分成三层：MOB12（建模标签）、校准后 vintage、终身（业务决策）。</p>
{table(monitor, headers=['监控项','基线 / 口径'])}
<div class="warn">2018 年 Current 占比 {data['current_share_2018']:.1%}；MOB12 口径已经把 OOT 截到 2018-03，避免把“未成熟”当“低风险”。但 2018Q2+ 被排除也意味着 OOT 不覆盖最近放款，监控上线后仍需按季度滚动补充。</div>

<h2>9. 仍然存在的局限</h2>
<ul>
<li><strong>拒绝推断：</strong>仍只使用已批准贷款，未使用 RejectStats；批准政策的选择偏差仍在。</li>
<li><strong>没有真正逐月表现快照：</strong>MOB12 用 <code>last_pymnt_d</code> 近似事件时点，不能画真正的 months-on-book 累计违约曲线；2018Q2+ 无法评估。</li>
<li><strong>term 耦合：</strong>本版披露了期限结构，但没有做“去掉 term 重训”的完整敏感性；生产必须锁定 term 时点。</li>
<li><strong>FICO/grade：</strong>FICO 只有 +0.0036 AUC；grade 很强但属于平台评级，不能入主模型。AUC 0.66 不是“方法差”，也不是“只差 FICO”。</li>
<li><strong>阈值目标：</strong>15%/18% 目标非绑定，说明目标偏松；未做 LGD/EAD、资金成本、获客成本与定价联动的收益优化。</li>
<li><strong>校准：</strong>Platt 在 OOT 上把平均 PD 从 4.56% 抬到 6.12%，略高于实际 5.71%；终身 PD 20.30% 也高于实测 18.18%，存在约 2pp 保守偏差。校准器只在 2016 fit，跨期仍需重估。</li>
<li><strong>治理：</strong>公平性、压力测试、独立验证、生产部署与模型风险评级仍未做。</li>
</ul>

<h2>10. 产物</h2>
<ul>
<li><code>outputs/report_v2.html</code>：本报告；<code>outputs/report.html</code> 为同一份 v2 主报告；<code>outputs/report_v1_terminal.html</code> 为旧版归档。</li>
<li><code>outputs/tables/</code>：calibrator_comparison_oot.csv、threshold_sensitivity.csv、sensitivity_ablation.csv、vif_initial.csv、grade_calibration_lifetime_oot.csv 等。</li>
<li><code>outputs/monitoring_plan.md</code>：按 MOB12/终身口径重写的监控方案。</li>
</ul>
<div class="footer">生成时间：{meta['generated_at']} · 数据快照：2019-03-31 · 运行耗时：{meta['runtime_seconds']:.0f}s · 所有校准器只在验证集 fit，OOT 只 transform。</div>
</div></body></html>
"""

(OUT / "report_v2.html").write_text(html_doc, encoding="utf-8")
(OUT / "report.html").write_text(html_doc, encoding="utf-8")
print(f"wrote {OUT / 'report_v2.html'} ({len(html_doc):,} chars)")
