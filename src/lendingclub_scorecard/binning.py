"""有监督分箱、WOE 与 IV。

约定：1=坏客户，0=好客户；WOE=ln(坏客户占比/好客户占比)，WOE 越高风险越高。
连续变量使用等频分箱，再合并小箱并按方向做单调性合并；二值变量直接两箱。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FEATURE_BY_NAME, FEATURE_NAMES, NINF, PINF, FeatureSpec, PipelineConfig
from .data import feature_label

SMOOTH = 0.5


def _fmt_number(value: float) -> str:
    if math.isinf(value):
        return "-∞" if value < 0 else "+∞"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.5f}".rstrip("0").rstrip(".")


def bin_label(lower: float, upper: float) -> str:
    if math.isinf(lower) and math.isinf(upper):
        return "全部"
    if math.isinf(lower):
        return f"≤ {_fmt_number(upper)}"
    if math.isinf(upper):
        return f"> {_fmt_number(lower)}"
    return f"({_fmt_number(lower)}, {_fmt_number(upper)}]"


def assign_bin_ids(values, edges: list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    inner = np.asarray(edges[1:-1], dtype=float)
    if inner.size == 0:
        return np.zeros(len(arr), dtype=int)
    ids = np.searchsorted(inner, arr, side="left")
    ids = np.where(np.isnan(arr), 0, ids)
    return ids.astype(int)


def _bin_stats(values: np.ndarray, y: np.ndarray, edges: list[float]) -> pd.DataFrame:
    n_bins = len(edges) - 1
    ids = assign_bin_ids(values, edges)
    frame = pd.DataFrame({"bin_id": ids, "y": np.asarray(y, dtype=int)})
    agg = frame.groupby("bin_id")["y"].agg(count="count", bad="sum")
    agg = agg.reindex(range(n_bins), fill_value=0)
    agg["bad"] = agg["bad"].astype(int)
    agg["good"] = agg["count"] - agg["bad"]
    agg["bad_rate"] = np.where(agg["count"] > 0, agg["bad"] / agg["count"], 0.0)

    total_bad = int(agg["bad"].sum())
    total_good = int(agg["good"].sum())
    if total_bad == 0 or total_good == 0:
        raise ValueError("目标变量必须同时包含好客户和坏客户")

    bad_pct = (agg["bad"] + SMOOTH) / (total_bad + SMOOTH * n_bins)
    good_pct = (agg["good"] + SMOOTH) / (total_good + SMOOTH * n_bins)
    agg["woe"] = np.log(bad_pct / good_pct)
    agg["iv"] = (bad_pct - good_pct) * agg["woe"]
    agg["lower"] = [float(edges[i]) for i in range(n_bins)]
    agg["upper"] = [float(edges[i + 1]) for i in range(n_bins)]
    agg["label"] = [bin_label(lo, hi) for lo, hi in zip(agg["lower"], agg["upper"])]
    agg = agg.reset_index()
    return agg[
        ["bin_id", "lower", "upper", "label", "count", "bad", "good", "bad_rate", "woe", "iv"]
    ]


def _merge_edges(edges: list[float], left_bin: int) -> list[float]:
    return edges[: left_bin + 1] + edges[left_bin + 2 :]


def _quantile_edges(values: np.ndarray, max_bins: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return [NINF, PINF]
    quantiles = np.unique(np.quantile(values, np.linspace(0.0, 1.0, max_bins + 1)))
    lo, hi = float(values.min()), float(values.max())
    inner = [float(q) for q in quantiles if lo < q < hi]
    return [NINF, *inner, PINF]


def _small_bin_merge(
    edges: list[float], values: np.ndarray, y: np.ndarray, min_count: int
) -> list[float]:
    while len(edges) > 3:
        stats = _bin_stats(values, y, edges)
        counts = stats["count"].to_numpy()
        if counts.min() >= min_count:
            break
        idx = int(np.argmin(counts))
        n_bins = len(edges) - 1
        if idx == 0:
            merge_at = 0
        elif idx == n_bins - 1:
            merge_at = n_bins - 2
        else:
            rates = stats["bad_rate"].to_numpy()
            left_gap = abs(rates[idx] - rates[idx - 1])
            right_gap = abs(rates[idx] - rates[idx + 1])
            merge_at = idx - 1 if left_gap <= right_gap else idx
        edges = _merge_edges(edges, int(merge_at))
    return edges


def _monotonic_merge(
    edges: list[float], values: np.ndarray, y: np.ndarray, direction: int
) -> list[float]:
    while len(edges) > 3:
        stats = _bin_stats(values, y, edges)
        rates = stats["bad_rate"].to_numpy()
        diffs = direction * np.diff(rates)
        if np.all(diffs >= -1e-12):
            break
        candidates = np.where(diffs < 0)[0]
        if len(candidates) == 0:
            break
        counts = stats["count"].to_numpy()
        merge_at = min(
            candidates,
            key=lambda k: (abs(diffs[k]), -(counts[k] + counts[k + 1])),
        )
        edges = _merge_edges(edges, int(merge_at))
    return edges


def _auto_direction(
    edges: list[float], values: np.ndarray, y: np.ndarray
) -> int:
    """用初始分箱的箱均值和坏客户率估计单调方向。"""
    stats = _bin_stats(values, y, edges)
    if len(stats) < 3:
        return 0
    mean_x = []
    for row in stats.itertuples(index=False):
        mask = (values > row.lower) & (values <= row.upper)
        if mask.sum() == 0:
            mean_x.append(np.nan)
        else:
            mean_x.append(float(np.nanmean(values[mask])))
    valid = ~np.isnan(mean_x)
    if valid.sum() < 3:
        return 0
    rho = pd.Series(np.asarray(mean_x)[valid]).corr(
        pd.Series(stats["bad_rate"].to_numpy()[valid]), method="spearman"
    )
    if pd.isna(rho) or abs(rho) < 0.10:
        return 0
    return int(np.sign(rho))


@dataclass
class BinningResult:
    feature: str
    name_cn: str
    group: str
    edges: list[float]
    bins: pd.DataFrame
    iv: float
    direction: int
    method: str

    @property
    def woe_values(self) -> list[float]:
        return self.bins["woe"].astype(float).tolist()

    def transform(self, series: pd.Series) -> pd.Series:
        ids = assign_bin_ids(series, self.edges)
        woe = np.asarray(self.woe_values, dtype=float)
        return pd.Series(woe[ids], index=series.index, name=self.feature)

    def to_dict(self) -> dict:
        def edge_json(v: float) -> float | None:
            return None if math.isinf(v) else float(v)

        return {
            "feature": self.feature,
            "name_cn": self.name_cn,
            "group": self.group,
            "edges": [edge_json(v) for v in self.edges],
            "iv": float(self.iv),
            "direction": int(self.direction),
            "method": self.method,
            "bins": [
                {
                    "bin_id": int(row.bin_id),
                    "label": str(row.label),
                    "lower": edge_json(float(row.lower)),
                    "upper": edge_json(float(row.upper)),
                    "count": int(row.count),
                    "bad": int(row.bad),
                    "good": int(row.good),
                    "bad_rate": float(row.bad_rate),
                    "woe": float(row.woe),
                    "iv": float(row.iv),
                }
                for row in self.bins.itertuples(index=False)
            ],
        }


def fit_binning(
    values: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    spec: FeatureSpec,
    cfg: PipelineConfig,
) -> BinningResult:
    x = np.asarray(values, dtype=float)
    target = np.asarray(y, dtype=int)
    unique = np.unique(x[~np.isnan(x)])

    if unique.size <= 2:
        mid = float(np.mean(unique)) if unique.size else 0.0
        edges = [NINF, mid, PINF]
        direction = 0
        method = "二值分箱"
    else:
        edges = _quantile_edges(x, cfg.max_bins)
        method = "等频分箱"
        direction = spec.direction if spec.direction != 0 else _auto_direction(edges, x, target)
        if direction == 0:
            method += "（不强制单调）"

    min_count = max(int(cfg.min_bin_pct * len(x)), 200)
    edges = _small_bin_merge(edges, x, target, min_count)
    if direction != 0:
        edges = _monotonic_merge(edges, x, target, direction)
        method += "+单调合并"
    stats = _bin_stats(x, target, edges)
    if unique.size <= 2 and set(unique.tolist()).issubset({0.0, 1.0}) and len(stats) == 2:
        stats.loc[stats["bin_id"] == 0, "label"] = "否（0）"
        stats.loc[stats["bin_id"] == 1, "label"] = "是（1）"

    while (stats["count"] == 0).any() and len(edges) > 3:
        idx = int(np.argmin(stats["count"].to_numpy()))
        edges = _merge_edges(edges, min(max(idx, 0), len(edges) - 3))
        stats = _bin_stats(x, target, edges)

    return BinningResult(
        feature=spec.name,
        name_cn=spec.name_cn,
        group=spec.group,
        edges=edges,
        bins=stats,
        iv=float(stats["iv"].sum()),
        direction=direction,
        method=method,
    )


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "几乎无预测力"
    if iv < 0.10:
        return "弱"
    if iv < 0.30:
        return "中等"
    if iv < 0.50:
        return "强"
    return "很强（需复核）"


def fit_all_binnings(
    X: pd.DataFrame, y: pd.Series, cfg: PipelineConfig
) -> tuple[dict[str, BinningResult], pd.DataFrame]:
    results: dict[str, BinningResult] = {}
    rows = []
    for col in X.columns:
        spec = FEATURE_BY_NAME.get(
            col,
            FeatureSpec(
                name=col,
                name_cn=feature_label(col),
                description="",
                group="衍生",
                kind="binary",
                direction=0,
            ),
        )
        series = X[col]
        if series.nunique(dropna=False) <= 1:
            continue
        result = fit_binning(series, y, spec, cfg)
        results[col] = result
        rows.append(
            {
                "feature": col,
                "name_cn": result.name_cn,
                "group": result.group,
                "iv": result.iv,
                "n_bins": len(result.bins),
                "strength": iv_strength(result.iv),
                "method": result.method,
                "direction": result.direction,
            }
        )
    iv_table = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    return results, iv_table


def woe_frame(
    X: pd.DataFrame, binnings: dict[str, BinningResult], features: list[str]
) -> pd.DataFrame:
    return pd.DataFrame(
        {feature: binnings[feature].transform(X[feature]) for feature in features},
        index=X.index,
    )


def binning_details(
    binnings: dict[str, BinningResult], selected: list[str] | None = None
) -> pd.DataFrame:
    frames = []
    for feature, result in binnings.items():
        if selected is not None and feature not in selected:
            continue
        frame = result.bins.copy()
        frame.insert(0, "feature", feature)
        frame.insert(1, "name_cn", result.name_cn)
        frame.insert(2, "group", result.group)
        frame["iv_total"] = result.iv
        frame["method"] = result.method
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
