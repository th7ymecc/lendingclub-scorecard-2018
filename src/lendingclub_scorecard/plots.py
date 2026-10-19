"""报告图表：Vintage、PSI、阈值扫描、ROC/KS、LightGBM 重要性等。"""

from __future__ import annotations

import math
import os
from pathlib import Path

_MPL_CONFIG = Path(__file__).resolve().parents[2] / ".mplconfig"
if (_MPL_CONFIG.parent / "pyproject.toml").exists():
    os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import roc_curve

from .binning import BinningResult
from .config import GRADE_ORDER
from .data import feature_label
from .evaluation import calibration_table, ks_curve_data

NAVY = "#17324d"
TEAL = "#1f8a8a"
ORANGE = "#e07a4f"
GOLD = "#d9a441"
GRAY = "#8d99ae"
RED = "#c44536"
GREEN = "#2a9d8f"
LIGHT = "#eef4f7"

_STYLE = False


def setup_style() -> None:
    global _STYLE
    if _STYLE:
        return
    candidates = ("Microsoft YaHei", "SimHei", "Noto Sans SC", "Source Han Sans CN")
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.sans-serif": [chosen, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#c9d5df",
            "axes.labelcolor": "#243746",
            "axes.titlecolor": NAVY,
            "xtick.color": "#4a5a68",
            "ytick.color": "#4a5a68",
            "grid.color": "#e6edf2",
            "legend.frameon": False,
        }
    )
    _STYLE = True


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _quarter_ticks(ax, labels, max_labels=10):
    step = max(1, math.ceil(len(labels) / max_labels))
    ax.set_xticks(np.arange(len(labels))[::step])
    ax.set_xticklabels(labels[::step], rotation=35, ha="right", fontsize=8)


def plot_sample_flow(base_all: pd.DataFrame, path) -> None:
    setup_style()
    grouped = (
        base_all.groupby("issue_year", observed=True)
        .agg(total=("loan_id", "size"), resolved=("resolved", "sum"))
        .reset_index()
    )
    grouped["resolved_rate"] = grouped["resolved"] / grouped["total"]
    fig, ax1 = plt.subplots(figsize=(8.8, 4.4), constrained_layout=True)
    x = np.arange(len(grouped))
    ax1.bar(x - 0.18, grouped["total"], width=0.36, color=NAVY, alpha=0.85, label="全部贷款")
    ax1.bar(x + 0.18, grouped["resolved"], width=0.36, color=TEAL, alpha=0.9, label="已终态")
    ax1.set_xticks(x)
    ax1.set_xticklabels(grouped["issue_year"].astype(int).astype(str))
    ax1.set_ylabel("贷款笔数")
    ax1.set_title("样本量与终态比例（按 issue_d 年份）")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.5)
    ax1.set_axisbelow(True)
    ax2 = ax1.twinx()
    ax2.plot(x, grouped["resolved_rate"], color=ORANGE, marker="o", linewidth=2, label="终态比例")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_ylabel("终态比例")
    ax2.legend(loc="lower right")
    _save(fig, path)


def plot_vintage(composition: pd.DataFrame, path) -> None:
    setup_style()
    frame = composition.sort_values("issue_quarter").reset_index(drop=True)
    x = np.arange(len(frame))
    labels = frame["issue_quarter"].tolist()
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), constrained_layout=True, sharex=True)

    ax = axes[0]
    colors = [ORANGE if q.startswith("2016") else NAVY for q in labels]
    ax.bar(x, frame["bad_rate"], color=colors, alpha=0.9, width=0.65)
    for i, value in enumerate(frame["bad_rate"]):
        if i % 2 == 0 or i >= len(frame) - 4:
            ax.text(i, value, f"{value:.1%}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("坏客户率（已终态）")
    ax.set_title("Vintage：按放款季度的已终态坏客户率")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    bottom = np.zeros(len(frame))
    for col, color, label in [
        ("resolved_rate", TEAL, "已终态"),
        ("current_rate", GOLD, "Current"),
        ("late_rate", ORANGE, "Late / Grace"),
    ]:
        values = frame[col].to_numpy()
        ax.bar(x, values, bottom=bottom, color=color, alpha=0.9, label=label, width=0.65)
        bottom += values
    ax.set_ylabel("占比")
    ax.set_title("Vintage：贷款状态构成（2016 年 Current 占比明显上升）")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="lower left", ncol=3)
    _quarter_ticks(ax, labels)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, path)


def plot_target_distribution(df: pd.DataFrame, target: str, path) -> None:
    setup_style()
    counts = df[target].value_counts().sort_index()
    labels = ["好客户 (0)", "坏客户 (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    bars = ax.bar(labels, values, color=[GREEN, RED], width=0.55, alpha=0.9)
    total = sum(values)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}\n{value/total:.1%}", ha="center", va="bottom")
    ax.set_ylabel("样本数")
    ax.set_title("OOT 样本好坏分布")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, path)


def plot_missing(df: pd.DataFrame, path) -> None:
    setup_style()
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(18)
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    if miss.empty:
        ax.text(0.5, 0.5, "无缺失值", ha="center", va="center")
        ax.axis("off")
    else:
        labels = [feature_label(c) for c in miss.index]
        ax.barh(labels[::-1], miss.values[::-1], color=TEAL, alpha=0.85)
        for i, value in enumerate(miss.values[::-1]):
            ax.text(value, i, f"{value:.1%}", va="center", fontsize=8)
        ax.set_xlabel("缺失率")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title("训练集缺失率最高的 18 个字段")
        ax.grid(axis="x", alpha=0.5)
        ax.set_axisbelow(True)
    _save(fig, path)


def plot_feature_distributions(df: pd.DataFrame, features: list[str], path) -> None:
    setup_style()
    features = features[:8]
    ncols = 4
    nrows = max(1, math.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.3 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, feature in zip(axes, features):
        series = df[feature].dropna()
        if series.nunique() <= 2:
            counts = series.value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values, color=TEAL, alpha=0.85)
        else:
            lo, hi = series.quantile(0.01), series.quantile(0.99)
            ax.hist(series.clip(lo, hi), bins=30, color=TEAL, alpha=0.82, edgecolor="white", linewidth=0.3)
        ax.set_title(feature_label(feature), fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.5)
        ax.set_axisbelow(True)
    for ax in axes[len(features):]:
        ax.axis("off")
    _save(fig, path)


def plot_iv(iv_table: pd.DataFrame, path, top_n: int = 22) -> None:
    setup_style()
    frame = iv_table.head(top_n).sort_values("iv", ascending=True)
    colors = [TEAL if iv >= 0.02 else GRAY for iv in frame["iv"]]
    fig, ax = plt.subplots(figsize=(8.6, 6.0), constrained_layout=True)
    bars = ax.barh(frame["name_cn"], frame["iv"], color=colors, alpha=0.9)
    for bar, value in zip(bars, frame["iv"]):
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    ax.axvline(0.02, color=ORANGE, linestyle="--", linewidth=1.1, label="IV 阈值 0.02")
    ax.set_xlabel("IV")
    ax.set_title("候选特征 IV 排名（前 22）")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, path)


def plot_woe(binnings: dict[str, BinningResult], selected: list[str], path) -> None:
    setup_style()
    n = min(len(selected), 9)
    selected = selected[:n]
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, selected):
        result = binnings[name]
        bins = result.bins
        colors = [RED if w > 0 else GREEN for w in bins["woe"]]
        ax.bar(np.arange(len(bins)), bins["woe"], color=colors, alpha=0.85)
        ax.axhline(0, color=NAVY, linewidth=0.8)
        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels(bins["label"], rotation=35, ha="right", fontsize=7)
        ax.set_title(f"{result.name_cn}（IV={result.iv:.3f}）", fontsize=10)
        ax.set_ylabel("WOE")
        ax.grid(axis="y", alpha=0.5)
        ax.set_axisbelow(True)
    for ax in axes[n:]:
        ax.axis("off")
    _save(fig, path)


def plot_roc(y, predictions: dict[str, np.ndarray], path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    colors = [NAVY, ORANGE, TEAL]
    for (label, p), color in zip(predictions.items(), colors):
        fpr, tpr, _ = roc_curve(np.asarray(y), np.asarray(p))
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{label}（AUC={auc:.4f}）")
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("假正率 FPR")
    ax.set_ylabel("真正率 TPR")
    ax.set_title("OOT ROC 曲线")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.5)
    _save(fig, path)


def plot_ks(y, p, path) -> None:
    setup_style()
    population, cum_bad, cum_good, ks, idx = ks_curve_data(y, p)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.plot(population, cum_bad, color=RED, linewidth=2.2, label="累计坏客户占比")
    ax.plot(population, cum_good, color=GREEN, linewidth=2.2, label="累计好客户占比")
    ax.vlines(population[idx], cum_good[idx], cum_bad[idx], color=NAVY, linestyle="--")
    ax.annotate(
        f"KS = {ks:.4f}",
        xy=(population[idx], (cum_bad[idx] + cum_good[idx]) / 2),
        xytext=(min(population[idx] + 0.12, 0.72), (cum_bad[idx] + cum_good[idx]) / 2 + 0.06),
        arrowprops=dict(arrowstyle="->", color=NAVY),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("按预测 PD 降序的累计人群占比")
    ax.set_ylabel("累计占比")
    ax.set_title("OOT KS 曲线（评分卡）")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.5)
    _save(fig, path)


def plot_calibration(calibrations: dict[str, pd.DataFrame], path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="完美校准")
    for (label, frame), color in zip(calibrations.items(), [NAVY, ORANGE]):
        ax.plot(frame["pred_mean"], frame["observed"], marker="o", color=color, linewidth=2, label=label)
    ax.set_xlabel("预测 PD（分箱均值）")
    ax.set_ylabel("实际坏客户率")
    ax.set_title("OOT 校准曲线")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend()
    ax.grid(alpha=0.5)
    _save(fig, path)


def _wilson_ci(bad: np.ndarray, count: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    p = bad / np.maximum(count, 1)
    denom = 1 + z * z / np.maximum(count, 1)
    centre = (p + z * z / (2 * np.maximum(count, 1))) / denom
    half = z * np.sqrt(p * (1 - p) / np.maximum(count, 1) + z * z / (4 * np.maximum(count, 1) ** 2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def plot_calibration_diagnostics(
    y,
    p_scorecard,
    p_lgbm,
    cal_metrics: dict,
    path,
    n_bins: int = 10,
) -> None:
    setup_style()
    frames = {
        "WOE 评分卡": calibration_table(y, p_scorecard, n_bins=n_bins),
        "LightGBM": calibration_table(y, p_lgbm, n_bins=n_bins),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="完美校准")
    for (label, frame), color in zip(frames.items(), [NAVY, ORANGE]):
        lo, hi = _wilson_ci(frame["bad"].to_numpy(), frame["count"].to_numpy())
        yerr = np.vstack([frame["observed"].to_numpy() - lo, hi - frame["observed"].to_numpy()])
        ax.errorbar(
            frame["pred_mean"],
            frame["observed"],
            yerr=yerr,
            marker="o",
            markersize=5,
            capsize=3,
            color=color,
            linewidth=1.8,
            label=f"{label}（ECE={cal_metrics['scorecard' if label == 'WOE 评分卡' else 'lgbm']['ece']:.2%}）",
        )
    ax.set_xlabel("预测 PD（分箱均值）")
    ax.set_ylabel("实际坏账率（95% Wilson CI）")
    ax.set_title("OOT 校准曲线")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.5)

    ax = axes[1]
    width = 0.36
    x = np.arange(len(next(iter(frames.values()))))
    for i, ((label, frame), color) in enumerate(zip(frames.items(), [NAVY, ORANGE])):
        ax.bar(x + (i - 0.5) * width, frame["gap"], width=width, color=color, alpha=0.85, label=label)
    ax.axhline(0, color=NAVY, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"B{i+1}" for i in x])
    ax.set_xlabel("按预测 PD 排序的校准分箱")
    ax.set_ylabel("实际 − 预测（pp）")
    ax.set_title("校准误差（正=低估风险，负=高估）")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend()
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle(
        "OOT 校准诊断：评分卡 vs LightGBM（误差棒为 95% Wilson 置信区间）",
        fontsize=11,
        color=NAVY,
    )
    _save(fig, path)


def plot_lift(y, predictions: dict[str, np.ndarray], path) -> None:
    setup_style()
    from .evaluation import lift_table

    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    width = 0.36
    for i, (label, p) in enumerate(predictions.items()):
        lift = lift_table(y, p)
        x = np.arange(len(lift)) + (i - 0.5) * width
        bars = ax.bar(x, lift["capture_rate"], width=width, color=[NAVY, ORANGE][i], alpha=0.9, label=label)
        for bar, row in zip(bars, lift.itertuples(index=False)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{row.capture_rate:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["前 5%", "前 10%", "前 20%", "前 30%"])
    ax.set_ylabel("坏客户捕获率")
    ax.set_title("OOT 风险排序提升对比")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend()
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, path)


def plot_psi_quarter(psi_df: pd.DataFrame, cfg, path) -> None:
    setup_style()
    frame = psi_df.copy()
    x = np.arange(len(frame))
    colors = [
        GREEN if v < cfg.psi_green else GOLD if v < cfg.psi_amber else RED for v in frame["psi"]
    ]
    fig, ax = plt.subplots(figsize=(11.0, 4.6), constrained_layout=True)
    ax.axhspan(0, cfg.psi_green, color=GREEN, alpha=0.08)
    ax.axhspan(cfg.psi_green, cfg.psi_amber, color=GOLD, alpha=0.10)
    ax.axhspan(cfg.psi_amber, max(frame["psi"].max() * 1.2, cfg.psi_amber * 1.4), color=RED, alpha=0.06)
    ax.bar(x, frame["psi"], color=colors, alpha=0.9, width=0.65)
    ax.axhline(cfg.psi_green, color=GREEN, linestyle="--", linewidth=1)
    ax.axhline(cfg.psi_amber, color=RED, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["period"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Score PSI")
    ax.set_title("评分 PSI 按放款季度（参照：2013-2014 训练集）")
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)
    _save(fig, path)


def plot_psi_features(psi_df: pd.DataFrame, cfg, path, top_n: int = 18) -> None:
    setup_style()
    frame = psi_df.head(top_n).sort_values("value", ascending=True)
    colors = [
        GREEN if v < cfg.psi_green else GOLD if v < cfg.psi_amber else RED for v in frame["value"]
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
    labels = [feature_label(f) for f in frame["feature"]]
    ax.barh(labels, frame["value"], color=colors, alpha=0.9)
    for i, value in enumerate(frame["value"]):
        ax.text(value, i, f"{value:.3f}", va="center", fontsize=8)
    ax.axvline(cfg.psi_green, color=GREEN, linestyle="--", linewidth=1)
    ax.axvline(cfg.psi_amber, color=RED, linestyle="--", linewidth=1)
    ax.set_xlabel("PSI（训练集 vs OOT）")
    ax.set_title("OOT 特征 PSI 最高的变量")
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, path)


def plot_threshold_scan(
    scan_valid: pd.DataFrame,
    scan_oot: pd.DataFrame,
    chosen_threshold: float,
    path,
    target_bad_rate: float = 0.15,
) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    panels = [
        ("approval_rate", "批准率", True),
        ("approved_bad_rate", "批准人群坏客户率", True),
        ("bad_capture", "坏客户拒绝捕获率", True),
    ]
    for ax, (col, title, pct) in zip(axes, panels):
        ax.plot(scan_valid["threshold"], scan_valid[col], color=NAVY, linewidth=2, label="验证集 2015")
        ax.plot(scan_oot["threshold"], scan_oot[col], color=ORANGE, linewidth=2, label="OOT 2016")
        ax.axvline(chosen_threshold, color=RED, linestyle="--", linewidth=1.2, label="选定阈值")
        if col == "approved_bad_rate":
            ax.axhline(target_bad_rate, color=GRAY, linestyle=":", linewidth=1, label="目标坏账率")
        ax.set_xlabel("PD 阈值（低于阈值批准）")
        ax.set_title(title)
        if pct:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower left")
    _save(fig, path)


def plot_score_distribution(scored: pd.DataFrame, target: str, grade_cutoffs: dict, path) -> None:
    setup_style()
    good = scored.loc[scored[target] == 0, "score"]
    bad = scored.loc[scored[target] == 1, "score"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    bins = np.arange(min(scored["score"].min(), 450), max(scored["score"].max(), 750) + 4, 5)
    ax.hist(good, bins=bins, density=True, alpha=0.55, color=GREEN, label="好客户")
    ax.hist(bad, bins=bins, density=True, alpha=0.55, color=RED, label="坏客户")
    for grade in ("A", "B", "C", "D"):
        ax.axvline(grade_cutoffs[grade], color=NAVY, linestyle=":", linewidth=1)
    ax.set_xlabel("信用评分")
    ax.set_ylabel("密度")
    ax.set_title("OOT 评分分布（虚线为 A/B/C/D 等级阈值）")
    ax.legend()
    ax.grid(axis="y", alpha=0.45)
    ax.set_axisbelow(True)
    _save(fig, path)


def plot_grade(grade_table: pd.DataFrame, path) -> None:
    setup_style()
    colors = {"A": GREEN, "B": "#457b9d", "C": GOLD, "D": ORANGE, "E": RED}
    frame = grade_table.sort_values("grade")
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    bars = ax.bar(
        frame["grade"].astype(str),
        frame["bad_rate"],
        color=[colors.get(str(g), GRAY) for g in frame["grade"]],
        alpha=0.92,
        width=0.55,
    )
    for bar, row in zip(bars, frame.itertuples(index=False)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row.bad_rate:.1%}\nn={row.count:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("评分等级")
    ax.set_ylabel("坏客户率")
    ax.set_title("OOT 各评分等级坏客户率")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, path)


def plot_lgbm_importance(importance: pd.DataFrame, path, top_n: int = 20) -> None:
    setup_style()
    frame = importance.head(top_n).sort_values("gain", ascending=True)
    fig, ax = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
    ax.barh(frame["name_cn"], frame["gain_pct"], color=TEAL, alpha=0.9)
    for i, value in enumerate(frame["gain_pct"]):
        ax.text(value, i, f"{value:.1%}", va="center", fontsize=8)
    ax.set_xlabel("Gain 占比")
    ax.set_title("LightGBM 特征重要性（Gain，前 20）")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, path)


def plot_swap_set(summary: dict, groups: pd.DataFrame, path) -> None:
    setup_style()
    frame = groups.copy()
    colors = [NAVY, ORANGE, TEAL, GRAY]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)

    ax = axes[0]
    bars = ax.bar(frame["group"], frame["count"], color=colors, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("人数（对数轴）")
    ax.set_title("Swap-set：同批准率下的集合划分")
    ax.tick_params(axis="x", rotation=18, labelsize=8)
    for bar, row in zip(bars, frame.itertuples(index=False)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{row.count:,}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    bars = ax.bar(frame["group"], frame["bad_rate"], color=colors, alpha=0.9)
    ax.set_ylabel("坏客户率")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Swap-set：各组实际坏客户率")
    ax.tick_params(axis="x", rotation=18, labelsize=8)
    for bar, row in zip(bars, frame.itertuples(index=False)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{row.bad_rate:.1%}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"同批准率 {summary['approval_rate']:.2%} · 重合率 {summary['overlap_rate']:.1%} · "
        f"净减少坏客户 = {summary['net_bad_avoided']:,}",
        fontsize=11,
        color=NAVY,
    )
    _save(fig, path)


def plot_model_comparison(metrics: pd.DataFrame, path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.8, 4.3), constrained_layout=True)
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, metrics["auc"], width=width, color=NAVY, alpha=0.9, label="AUC")
    ax.bar(x + width / 2, metrics["ks"], width=width, color=ORANGE, alpha=0.9, label="KS")
    for i, row in enumerate(metrics.itertuples(index=False)):
        ax.text(i - width / 2, row.auc, f"{row.auc:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, row.ks, f"{row.ks:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("指标值")
    ax.set_title("评分卡 vs LightGBM：OOT 区分能力")
    ax.legend()
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, path)
