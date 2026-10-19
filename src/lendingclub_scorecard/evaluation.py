"""OOT 评估、PSI、Vintage、阈值扫描与监控指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y, p) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(np.asarray(y), np.asarray(p))
    gaps = tpr - fpr
    idx = int(np.argmax(gaps))
    return float(gaps[idx]), float(thresholds[idx])


def ks_curve_data(y, p) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p)
    y_sorted = y[order]
    total_bad = int(y.sum())
    total_good = int(len(y) - total_bad)
    cum_bad = np.cumsum(y_sorted) / max(total_bad, 1)
    cum_good = np.cumsum(1 - y_sorted) / max(total_good, 1)
    gaps = cum_bad - cum_good
    idx = int(np.argmax(gaps))
    population = np.arange(1, len(y) + 1) / len(y)
    return population, cum_bad, cum_good, float(gaps[idx]), idx


def binary_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    auc = float(roc_auc_score(y, p))
    ks, ks_threshold = ks_statistic(y, p)
    return {
        "n": int(len(y)),
        "bad": int(y.sum()),
        "bad_rate": float(y.mean()),
        "auc": auc,
        "ks": ks,
        "ks_threshold": ks_threshold,
        "gini": 2.0 * auc - 1.0,
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "mean_pd": float(np.mean(p)),
    }


def bootstrap_auc(y, p, n_boot: int = 200, seed: int = 42) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(float(roc_auc_score(y[idx], p[idx])))
    arr = np.asarray(vals)
    return {
        "mean": float(arr.mean()),
        "lower": float(np.quantile(arr, 0.025)),
        "upper": float(np.quantile(arr, 0.975)),
        "rounds": int(len(arr)),
    }


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y, dtype=int), "p": np.asarray(p, dtype=float)})
    bins = pd.qcut(frame["p"], n_bins, duplicates="drop")
    grouped = frame.groupby(bins, observed=True).agg(
        count=("y", "size"), bad=("y", "sum"), pred_mean=("p", "mean")
    )
    grouped["observed"] = grouped["bad"] / grouped["count"]
    grouped["gap"] = grouped["observed"] - grouped["pred_mean"]
    grouped = grouped.reset_index().rename(columns={"p": "probability_bin"})
    grouped["probability_bin"] = grouped["probability_bin"].astype(str)
    return grouped[["probability_bin", "count", "bad", "pred_mean", "observed", "gap"]]


def calibration_metrics(y, p, n_bins: int = 10) -> dict[str, float]:
    """校准分析：均值、校准截距/斜率、ECE、MCE、Brier。

    校准斜率理想为 1，校准截距理想为 0（在 logit 尺度上）。
    """
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    try:
        X = sm.add_constant(pd.DataFrame({"logit_pd": logit}))
        result = sm.Logit(y, X).fit(disp=0)
        intercept = float(result.params["const"])
        slope = float(result.params["logit_pd"])
    except Exception:  # noqa: BLE001
        intercept = float("nan")
        slope = float("nan")

    cal = calibration_table(y, p, n_bins=n_bins)
    weights = cal["count"].to_numpy() / len(y)
    gaps = cal["gap"].abs().to_numpy()
    return {
        "mean_pred": float(np.mean(p)),
        "actual": float(np.mean(y)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece": float(np.sum(weights * gaps)),
        "mce": float(np.max(gaps)),
        "brier": float(brier_score_loss(y, p)),
    }


def swap_set_analysis(
    y,
    p_a,
    p_b,
    n_approve: int | None = None,
    approval_rate: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame]:
    """同一批准率下比较两个模型的 Swap-set。

    p 越低越安全；按 p 升序取相同人数作为批准集合。
    swap-in  = B 批准、A 拒绝；swap-out = A 批准、B 拒绝。
    net_bad_avoided > 0 表示 B 在交换人群中少批准了坏客户。

    y 可为 0/1 整数标签或连续加权标签（终身口径，取值 0 或 1/e）。加权时
    bad / net_bad_avoided 等字段为加权和而非笔数，解释为"预期终身坏账量"。
    内部一律 float 累加，避免对加权标签做 int 截断造成静默失真。
    """
    y = np.asarray(y, dtype=float)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    n = len(y)
    if n_approve is None:
        n_approve = int(round(approval_rate * n))
    n_approve = min(max(int(n_approve), 1), n)

    a = np.zeros(n, dtype=bool)
    b = np.zeros(n, dtype=bool)
    a[np.argsort(p_a)[:n_approve]] = True
    b[np.argsort(p_b)[:n_approve]] = True

    both = a & b
    scorecard_only = a & ~b
    lgbm_only = b & ~a
    neither = ~a & ~b

    def rate(mask: np.ndarray) -> float:
        return float(y[mask].mean()) if mask.any() else float("nan")

    total_bad = float(y.sum())
    bad_in = float(y[lgbm_only].sum())
    bad_out = float(y[scorecard_only].sum())
    good_in = float(lgbm_only.sum() - bad_in)
    good_out = float(scorecard_only.sum() - bad_out)

    summary = {
        "approval_rate": n_approve / n,
        "approved_n": n_approve,
        "scorecard_bad_rate": rate(a),
        "lgbm_bad_rate": rate(b),
        "scorecard_bad_capture": float(y[~a].sum() / total_bad) if total_bad else 0.0,
        "lgbm_bad_capture": float(y[~b].sum() / total_bad) if total_bad else 0.0,
        "overlap_rate": float(both.sum() / n_approve),
        "jaccard": float(both.sum() / (a | b).sum()),
        "swap_in_n": int(lgbm_only.sum()),
        "swap_out_n": int(scorecard_only.sum()),
        "swap_in_bad_rate": rate(lgbm_only),
        "swap_out_bad_rate": rate(scorecard_only),
        "net_bad_avoided": float(bad_out - bad_in),
        "net_good_gain": float(good_in - good_out),
        "swap_in_bad": bad_in,
        "swap_out_bad": bad_out,
    }
    groups = pd.DataFrame(
        [
            ("两模型均批准", int(both.sum()), float(y[both].sum()), rate(both)),
            ("仅评分卡批准（swap-out）", int(scorecard_only.sum()), bad_out, rate(scorecard_only)),
            ("仅 LightGBM 批准（swap-in）", int(lgbm_only.sum()), bad_in, rate(lgbm_only)),
            ("两模型均拒绝", int(neither.sum()), float(y[neither].sum()), rate(neither)),
        ],
        columns=["group", "count", "bad", "bad_rate"],
    )
    return summary, groups


def lift_table(y, p, fractions=(0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p)
    y_sorted = y[order]
    total_bad = int(y.sum())
    rows = []
    for frac in fractions:
        k = max(int(round(frac * len(y))), 1)
        captured = int(y_sorted[:k].sum())
        rows.append(
            {
                "top_pct": frac,
                "n": k,
                "bad": captured,
                "capture_rate": captured / total_bad if total_bad else 0.0,
                "lift": (captured / total_bad) / frac if total_bad else 0.0,
                "bad_rate": float(y_sorted[:k].mean()),
            }
        )
    return pd.DataFrame(rows)


def psi(expected, actual, n_bins: int = 10, eps: float = 1e-6) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        values = np.unique(np.concatenate([expected, actual]))
        if len(values) < 2:
            return 0.0
        if len(values) == 2:
            edges = np.array([-np.inf, (values[0] + values[1]) / 2.0, np.inf], dtype=float)
        else:
            edges = np.array([-np.inf, *values[1:-1], np.inf], dtype=float)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    expected_bins = pd.cut(expected, bins=edges, include_lowest=True)
    actual_bins = pd.cut(actual, bins=edges, include_lowest=True)
    categories = expected_bins.categories
    expected_pct = (
        pd.Series(expected_bins).value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    )
    actual_pct = (
        pd.Series(actual_bins).value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    )
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def psi_level(value: float) -> str:
    if value < 0.10:
        return "稳定"
    if value < 0.25:
        return "需关注"
    return "显著漂移"


def psi_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        value = psi(reference[feature], current[feature], n_bins=n_bins)
        rows.append(
            {
                "feature": feature,
                "value": value,
                "level": psi_level(value),
            }
        )
    return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)


def score_psi_by_period(
    reference_scores,
    scored: pd.DataFrame,
    score_col: str = "score",
    period_col: str = "issue_quarter",
) -> pd.DataFrame:
    rows = []
    for period, group in scored.groupby(period_col, observed=True):
        value = psi(reference_scores, group[score_col])
        rows.append({"period": str(period), "psi": value, "level": psi_level(value)})
    return pd.DataFrame(rows)


def threshold_scan(
    y,
    p,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """p < threshold 视为批准（低风险）。

    标签 y 支持两种口径：
      * 0/1 整数标签（vintage 口径，MOB12 前是否核销）；
      * 连续加权标签（终身口径，取值 0 或 1/e，e 为该笔贷款的坏账暴露比例）。
    后者是对终身坏账率的无偏估计：E[y_vintage / e] = p_lifetime。

    **内部一律用 float 累加**。早期版本写成 int(...)，对加权标签会静默截断
    （1/0.346=2.89 被截成 2），导致坏账率与捕获率全部失真。对整数标签，float
    化不改变任何结果。
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if thresholds is None:
        thresholds = np.arange(0.02, 0.601, 0.01)
    total_bad = float(y.sum())
    baseline = float(y.mean())
    rows = []
    for threshold in thresholds:
        approved = p < threshold
        n_app = int(approved.sum())
        n_rej = int(len(y) - n_app)
        bad_app = float(y[approved].sum()) if n_app else 0.0
        bad_rej = float(y[~approved].sum()) if n_rej else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "approval_rate": n_app / len(y),
                "approved_n": n_app,
                "rejected_n": n_rej,
                "approved_bad": bad_app,
                "rejected_bad": bad_rej,
                "approved_bad_rate": bad_app / n_app if n_app else np.nan,
                "rejected_bad_rate": bad_rej / n_rej if n_rej else np.nan,
                "bad_capture": bad_rej / total_bad if total_bad else 0.0,
                "bad_rate_reduction": baseline - (bad_app / n_app if n_app else np.nan),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(
    scan: pd.DataFrame,
    target_bad_rate: float = 0.15,
    min_approval: float = 0.50,
) -> pd.Series:
    eligible = scan[
        (scan["approved_bad_rate"] <= target_bad_rate)
        & (scan["approval_rate"] >= min_approval)
    ]
    if len(eligible):
        return eligible.sort_values("approval_rate", ascending=False).iloc[0]
    # 回退：在满足最低批准率的前提下，选择批准坏账率最低的阈值
    eligible = scan[scan["approval_rate"] >= min_approval]
    if len(eligible):
        return eligible.sort_values("approved_bad_rate").iloc[0]
    return scan.iloc[-1]


def threshold_confusion(y, p, threshold: float) -> dict[str, float]:
    """给定阈值下的批准/拒绝构成。

    y 可为 0/1 整数标签（vintage 口径）或连续加权标签（终身口径，取值 0 或 1/e）。
    加权标签下 approved_bad_rate 是终身坏账率的无偏估计，可直接与业务目标比较；
    approved_bad / rejected_bad 为加权和而非笔数。

    内部一律 float 累加，避免对加权标签做 int 截断造成静默失真。
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    approved = p < threshold
    n_app = int(approved.sum())
    n_rej = int((~approved).sum())
    bad_app = float(y[approved].sum()) if n_app else 0.0
    bad_rej = float(y[~approved].sum()) if n_rej else 0.0
    total_bad = float(y.sum())
    return {
        "threshold": float(threshold),
        "approved_n": n_app,
        "rejected_n": n_rej,
        # 加权标签下 approved_good 是"批准笔数 − 加权坏账量"，不是严格的整数笔数，
        # 仅用于与 approved_bad 一起核对批准人群构成。
        "approved_good": float(n_app - bad_app),
        "approved_bad": bad_app,
        "rejected_good": float(n_rej - bad_rej) if n_rej else 0.0,
        "rejected_bad": bad_rej,
        "approval_rate": float(approved.mean()),
        "approved_bad_rate": bad_app / n_app if n_app else np.nan,
        "bad_capture": bad_rej / total_bad if total_bad else 0.0,
    }


def approval_at_rate(
    y, p, target_approval: float, thresholds: np.ndarray | None = None
) -> pd.Series:
    """找到使批准率最接近 target_approval 的阈值，并返回该阈值指标。

    thresholds 需与 p 的取值范围匹配：vintage 口径 PD 落在 0~0.6，默认网格够用；
    终身口径 PD 可高至 0.9，必须传入更宽的网格，否则匹配到的批准率会偏离目标。
    """
    scan = threshold_scan(y, p, thresholds=thresholds)
    idx = (scan["approval_rate"] - target_approval).abs().idxmin()
    return scan.loc[idx]


def grade_table(y, score, grade) -> pd.DataFrame:
    from .config import GRADE_ORDER

    frame = pd.DataFrame(
        {
            "y": np.asarray(y, dtype=int),
            "score": np.asarray(score, dtype=float),
            "grade": np.asarray(grade, dtype=str),
        }
    )
    grouped = frame.groupby("grade", observed=True).agg(
        count=("y", "size"),
        bad=("y", "sum"),
        bad_rate=("y", "mean"),
        avg_score=("score", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    ).reset_index()
    grouped["share"] = grouped["count"] / len(frame)
    grouped["grade"] = pd.Categorical(grouped["grade"], categories=GRADE_ORDER, ordered=True)
    return grouped.sort_values("grade").reset_index(drop=True)
