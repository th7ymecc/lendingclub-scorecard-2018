"""概率校准与 MOB 口径换算。

本模块解决两个旧版遗留的根本问题：

1. **模型 PD 未经校准**。旧版直接用逻辑回归输出作为 PD，OOT 上校准斜率 0.819、
   截距 0.036，系统性偏低。本版在验证集上拟合 Platt 与 isotonic 校准器，OOT 只做
   transform，保证不泄漏。

2. **MOB 对齐后口径变化**。标签改为"MOB12 前是否核销"后，PD 的绝对值变小（约为
   终身坏账率的 30%），业务无法直接使用。需要换算回终身口径。

换算的数学关系
--------------
设 L = 终身是否核销，B = MOB12 前是否核销。B=1 蕴含 L=1（MOB12 前核销的贷款终身
必然是核销），因此：

    P(B=1) = P(L=1) × P(B=1 | L=1) = p_终身 × e

其中 e = 坏样本在 MOB12 前的暴露比例。于是：

    p_终身 = p_vintage / e

这是**概率空间的除法**，不是 logit 平移。用成熟 cohort 实测验证：MOB12 vintage
坏账率 4.84% / 暴露比例 30.40% = 15.92%，与实测终身坏账率 15.91% 吻合。
（对比：logit 平移会得到 14.31%，系统性低估约 1.6pp，故不采用。）

**e 必须分段估计**。高风险客群的坏样本暴露更快（e 更大），低风险客群更慢（e 更小）。
若统一用总体 e=0.304，高风险段换算后会超过 1 而被 clip，造成失真。因此按 grade 分段
实测各自的 e，再做换算。分段变量用 grade 而非模型分数，因为 grade 不参与建模，用它
分段换算的是"标签成熟度"而非"预测能力"，不引入循环论证。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .config import TARGET, TARGET_VINTAGE, PipelineConfig

LOGGER = logging.getLogger(__name__)

EPS = 1e-6

# 总体分段（不分风险层）的标识
ALL_SEGMENT = "__ALL__"


# ============================================================ 校准器
class PlattCalibrator:
    """Platt scaling：在 logit 空间拟合一元逻辑回归。

    只用验证集 fit，OOT 与生产均只 transform。
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
        self.a_: float = float("nan")
        self.b_: float = float("nan")

    def fit(self, p, y) -> "PlattCalibrator":
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        y = np.asarray(y, dtype=int)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        self._model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self._model.fit(logit, y)
        self.a_ = float(self._model.coef_[0][0])
        self.b_ = float(self._model.intercept_[0])
        LOGGER.info("Platt 校准拟合: p_lt = sigmoid(%.4f × logit(p) + %.4f)", self.a_, self.b_)
        return self

    def transform(self, p) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("PlattCalibrator 尚未 fit")
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        return np.asarray(self._model.predict_proba(logit)[:, 1], dtype=float)


class IsotonicCalibrator:
    """保序回归校准：非参数，能修正非线性失真但易过拟合小样本。"""

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None

    def fit(self, p, y) -> "IsotonicCalibrator":
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=int)
        self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._model.fit(p, y)
        LOGGER.info("Isotonic 校准拟合完成（%s 个样本）", f"{len(y):,}")
        return self

    def transform(self, p) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("IsotonicCalibrator 尚未 fit")
        return np.clip(np.asarray(self._model.transform(np.asarray(p, dtype=float)), dtype=float), EPS, 1 - EPS)


def calibrate_frame(p, y, platt: PlattCalibrator, iso: IsotonicCalibrator) -> pd.DataFrame:
    """输出原始 PD 与两种校准后 PD 的对照表，便于选择。"""
    from .evaluation import calibration_metrics

    rows = []
    for name, pred in (
        ("raw", np.asarray(p, dtype=float)),
        ("platt", platt.transform(p)),
        ("isotonic", iso.transform(p)),
    ):
        m = calibration_metrics(y, pred)
        rows.append({"calibrator": name, **m})
    return pd.DataFrame(rows)


# ============================================================ 暴露比例与终身换算
def risk_layer_emergence(
    base_all: pd.DataFrame,
    cfg: PipelineConfig,
    by: str = "grade",
) -> pd.DataFrame:
    """诊断表：按风险分层实测暴露比例 e，论证"为何不按风险分段换算"。

    这不是换算入口（换算入口是 term_emergence + apply_lifetime_conversion），而是
    设计依据。实测结论：e 的分层敏感性几乎全部来自贷款期限而非风险等级——同一
    期限内各风险层 e 差异较小，而 36 期与 60 期相差约 35%。因此终身换算按期限
    分段即可，无需 term × risk 二维细分，避免小样本分段的方差问题。

    只用完全成熟的 cohort（2013-2014 放款、obs_mob>=60），此时终身坏账率已收敛，
    "MOB12 前核销数 / 终身核销数" 才是真实的暴露比例。

    返回每层的 e、终身坏账率、MOB12 vintage 坏账率，以及换算校验列。
    """
    mob = cfg.performance_mob
    mature = base_all.loc[
        (base_all["issue_dt"] < "2015-01-01")
        & (base_all["obs_mob"] >= 60)
        & base_all[TARGET].notna()
    ].copy()
    if mature.empty:
        LOGGER.warning("无成熟 cohort，风险分层暴露比例无法估计")
        return pd.DataFrame()
    if by not in mature.columns:
        LOGGER.warning("分层字段 %s 不存在，跳过风险分层诊断", by)
        return pd.DataFrame()

    mature["event_mob"] = mature["event_mob"].fillna(mature["term_num"])
    mature["bad_by_mob"] = (
        (mature[TARGET] == 1) & (mature["event_mob"] <= mob)
    ).astype(int)
    mature["_seg"] = mature[by].astype(str)

    rows = []
    for seg, g in mature.groupby("_seg", observed=True):
        n = len(g)
        n_bad_life = int((g[TARGET] == 1).sum())
        n_bad_mob = int(g["bad_by_mob"].sum())
        e = n_bad_mob / n_bad_life if n_bad_life else np.nan
        p_lt = n_bad_life / n if n else np.nan
        p_v = n_bad_mob / n if n else np.nan
        rows.append(
            {
                "segment": str(seg),
                "segment_by": by,
                "n_mature": n,
                "lifetime_bad": n_bad_life,
                "bad_by_mob": n_bad_mob,
                "lifetime_bad_rate": p_lt,
                f"vintage_mob{mob}_bad_rate": p_v,
                "emergence_frac": e,
                # 校验：vintage / e 应还原 lifetime
                "check_lifetime": p_v / e if e else np.nan,
                "check_error": (p_v / e - p_lt) if e else np.nan,
            }
        )
    out = pd.DataFrame(rows)

    # 总体行，供报告展示与缺失分段兜底
    n = len(mature)
    n_bad_life = int((mature[TARGET] == 1).sum())
    n_bad_mob = int(mature["bad_by_mob"].sum())
    e_all = n_bad_mob / n_bad_life if n_bad_life else np.nan
    total = {
        "segment": ALL_SEGMENT,
        "segment_by": by,
        "n_mature": n,
        "lifetime_bad": n_bad_life,
        "bad_by_mob": n_bad_mob,
        "lifetime_bad_rate": n_bad_life / n,
        f"vintage_mob{mob}_bad_rate": n_bad_mob / n,
        "emergence_frac": e_all,
        "check_lifetime": (n_bad_mob / n) / e_all if e_all else np.nan,
        "check_error": ((n_bad_mob / n) / e_all - n_bad_life / n) if e_all else np.nan,
    }
    out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
    seg_only = out.loc[out["segment"] != ALL_SEGMENT, "emergence_frac"].dropna()
    LOGGER.info(
        "风险分层暴露比例（MOB%d, by %s）: 总体 e=%.4f，分层极差 %.4f，变异系数 %.4f",
        mob,
        by,
        e_all,
        float(seg_only.max() - seg_only.min()) if len(seg_only) else np.nan,
        float(seg_only.std() / seg_only.mean()) if len(seg_only) else np.nan,
    )
    return out


def risk_layer_within_term(
    base_all: pd.DataFrame,
    cfg: PipelineConfig,
    by: str = "grade",
) -> pd.DataFrame:
    """诊断表：在同一期限内观察各风险层的 e，验证期限单维分段是否充分。

    若同一期限内各风险层 e 的离散度明显小于跨期限差异，说明按期限单维分段已足够，
    无需 term × risk 二维细分（二维会把每格样本切小，e 的估计方差反而上升）。
    """
    mob = cfg.performance_mob
    if by not in base_all.columns:
        return pd.DataFrame()
    mature = base_all.loc[
        (base_all["issue_dt"] < "2015-01-01")
        & (base_all["obs_mob"] >= 60)
        & base_all[TARGET].notna()
        & base_all[by].notna()
    ].copy()
    if mature.empty:
        return pd.DataFrame()

    mature["event_mob"] = mature["event_mob"].fillna(mature["term_num"])
    mature["bad_by_mob"] = (
        (mature[TARGET] == 1) & (mature["event_mob"] <= mob)
    ).astype(int)
    mature["_term"] = mature["term_num"].astype(int).astype(str)

    rows = []
    for (term, seg), g in mature.groupby(["_term", by], observed=True):
        n_life = int((g[TARGET] == 1).sum())
        n_mob = int(g["bad_by_mob"].sum())
        rows.append(
            {
                "term": term,
                by: str(seg),
                "n_mature": len(g),
                "lifetime_bad": n_life,
                "emergence_frac": n_mob / n_life if n_life else np.nan,
                # 样本过小的格子 e 估计不稳，标记但仍保留供查看
                "low_n": bool(n_life < 200),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    stable = out.loc[~out["low_n"], "emergence_frac"].dropna()
    cross_term = (
        out.loc[~out["low_n"]].groupby("term")["emergence_frac"].mean()
    )
    if len(stable) and len(cross_term) > 1:
        within_range = float(
            out.loc[~out["low_n"]].groupby("term")["emergence_frac"].agg(
                lambda s: s.max() - s.min()
            ).mean()
        )
        between_range = float(cross_term.max() - cross_term.min())
        LOGGER.info(
            "期限内 e 极差 %.4f vs 跨期限 e 极差 %.4f -> %s",
            within_range,
            between_range,
            "期限分段充分" if between_range > within_range else "需考虑二维分段",
        )
    return out


def to_lifetime_pd(
    p_vintage: np.ndarray,
    emergence: pd.Series | np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """把 MOB 对齐口径的 PD 换算到终身口径。

    p_终身 = p_vintage / e，超出 1 的部分 clip 并计入 clipped 掩码。
    被 clip 的样本说明该段 e 偏低或模型 PD 偏高，换算结果为**下限估计**。

    返回 (p_lifetime, clipped_mask)。
    """
    p_v = np.asarray(p_vintage, dtype=float)
    e = np.asarray(emergence, dtype=float)
    e = np.where((e <= 0) | ~np.isfinite(e), np.nan, e)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_lt = p_v / e
    clipped = np.isfinite(p_lt) & (p_lt > 1.0)
    p_lt = np.clip(np.nan_to_num(p_lt, nan=0.0), 0.0, 1.0)
    return p_lt, clipped


def term_emergence(
    base_all: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """按贷款期限分段实测暴露比例 e。

    **为什么按 term 分段**：实测发现 e 的分层敏感性几乎全部来自期限而非风险等级。
    36 期贷款 e=0.346，60 期 e=0.257，相差 35%；而同一期限内各风险等级的 e 差异
    要小得多。原因是 60 期贷款在 MOB12 时远未到期，坏样本暴露更慢。

    **这个差异有实质业务后果**：60 期与 36 期的终身坏账率是 2.02 倍关系
    （25.7% vs 12.7%），但在 MOB12 vintage 口径下只剩 1.50 倍（6.6% vs 4.4%）。
    即固定表现窗口会系统性压缩长期限贷款的风险差异，而 term 恰恰是本模型最强的
    入模变量。不做期限分段换算，长期限贷款的风险会被显著低估。

    返回每个期限的 e、终身坏账率、vintage 坏账率与换算校验列。
    """
    mob = cfg.performance_mob
    mature = base_all.loc[
        (base_all["issue_dt"] < "2015-01-01")
        & (base_all["obs_mob"] >= 60)
        & base_all[TARGET].notna()
    ].copy()
    if mature.empty:
        LOGGER.warning("无成熟 cohort，期限暴露比例无法估计")
        return pd.DataFrame()

    mature["event_mob"] = mature["event_mob"].fillna(mature["term_num"])
    mature["bad_by_mob"] = (
        (mature[TARGET] == 1) & (mature["event_mob"] <= mob)
    ).astype(int)
    mature["_term"] = mature["term_num"].astype(int).astype(str)

    rows = []
    for term, g in mature.groupby("_term", observed=True):
        n = len(g)
        n_life = int((g[TARGET] == 1).sum())
        n_mob = int(g["bad_by_mob"].sum())
        e = n_mob / n_life if n_life else np.nan
        lt = n_life / n if n else np.nan
        v = n_mob / n if n else np.nan
        rows.append(
            {
                "segment": str(term),
                "segment_by": "term_num",
                "n_mature": n,
                "lifetime_bad": n_life,
                "bad_by_mob": n_mob,
                "lifetime_bad_rate": lt,
                f"vintage_mob{mob}_bad_rate": v,
                "emergence_frac": e,
                "to_lifetime_factor": 1.0 / e if e else np.nan,
                "check_lifetime": v / e if e else np.nan,
                "check_error": (v / e - lt) if e else np.nan,
            }
        )
    out = pd.DataFrame(rows)

    # 总体 e（按期限结构加权），供缺失期限兜底与跨年对比
    n = len(mature)
    n_life = int((mature[TARGET] == 1).sum())
    n_mob = int(mature["bad_by_mob"].sum())
    e_all = n_mob / n_life if n_life else np.nan
    out = pd.concat(
        [
            out,
            pd.DataFrame(
                [
                    {
                        "segment": ALL_SEGMENT,
                        "segment_by": "term_num",
                        "n_mature": n,
                        "lifetime_bad": n_life,
                        "bad_by_mob": n_mob,
                        "lifetime_bad_rate": n_life / n,
                        f"vintage_mob{mob}_bad_rate": n_mob / n,
                        "emergence_frac": e_all,
                        "to_lifetime_factor": 1.0 / e_all if e_all else np.nan,
                        "check_lifetime": (n_mob / n) / e_all if e_all else np.nan,
                        "check_error": ((n_mob / n) / e_all - n_life / n) if e_all else np.nan,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    LOGGER.info(
        "期限暴露比例（MOB%d）: %s；总体 e=%.4f",
        mob,
        {r.segment: round(r.emergence_frac, 4) for r in out.itertuples(index=False)},
        e_all,
    )
    return out


def emergence_by_term_structure(
    base_all: pd.DataFrame,
    term_table: pd.DataFrame,
) -> pd.DataFrame:
    """按各放款期的期限结构加权，给出该期的等效暴露比例 e。

    期限结构会随时间漂移（60 期占比 2012 年 18.6%、2015 年 32.8%），因此不同
    cohort 的等效 e 不同。等效 e 稳定意味着跨期比较可靠；若波动大，则跨期坏账率
    差异可能来自期限结构而非真实风险变化，需要在监控中区分。
    """
    if term_table.empty:
        return pd.DataFrame()
    e_map = dict(zip(term_table["segment"].astype(str), term_table["emergence_frac"]))
    e_all = float(e_map.get(ALL_SEGMENT, np.nan))

    rows = []
    for period, g in base_all.groupby("issue_year", observed=True):
        weights = g["term_num"].astype(int).astype(str).value_counts(normalize=True)
        e_eff = sum(float(w) * float(e_map.get(t, e_all)) for t, w in weights.items())
        rows.append(
            {
                "issue_year": int(period),
                "n": len(g),
                **{f"term{t}_share": float(w) for t, w in weights.items()},
                "e_effective": e_eff,
                "to_lifetime_factor": 1.0 / e_eff if e_eff else np.nan,
            }
        )
    return pd.DataFrame(rows)


def lifetime_conversion_summary(
    vintage_rate: float,
    term_table: pd.DataFrame,
) -> dict[str, float]:
    """把一个 vintage 坏账率按期限结构换算到终身口径，并给出区间。

    区间取各期限换算结果的极值，反映"若期限结构与成熟 cohort 不同"带来的不确定性。
    """
    if term_table.empty:
        return {"vintage": vintage_rate, "lifetime": np.nan, "lower": np.nan, "upper": np.nan}
    seg = term_table.loc[term_table["segment"] != ALL_SEGMENT]
    overall = float(
        term_table.loc[term_table["segment"] == ALL_SEGMENT, "emergence_frac"].iloc[0]
    )
    per_term = (vintage_rate / seg["emergence_frac"].astype(float)).to_numpy()
    return {
        "vintage": vintage_rate,
        "lifetime_overall_e": vintage_rate / overall if overall else np.nan,
        "lifetime_lower": float(np.nanmin(per_term)) if len(per_term) else np.nan,
        "lifetime_upper": float(np.nanmax(per_term)) if len(per_term) else np.nan,
    }


# ============================================================ 阈值敏感性
def threshold_sensitivity(
    y,
    p_lifetime,
    target_bad_rates: tuple[float, ...],
    min_approval: float = 0.50,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """在终身口径 PD 上做阈值敏感性分析。

    旧版只报告"目标坏账率 15% 时选中的阈值"，无法看出该选择对坏账率估计误差的
    敏感度。这里对多个目标坏账率分别选阈值，输出阈值、批准率、坏账率与捕获率，
    让决策者看到阈值随目标漂移的幅度。

    注意：阈值扫描用的是校准并换算到终身口径的 PD，因此目标坏账率可以直接按业务
    语义（终身坏账率）设定，而不是 MOB12 口径的数值。
    """
    from .evaluation import choose_threshold, threshold_scan

    y = np.asarray(y, dtype=int)
    p = np.asarray(p_lifetime, dtype=float)
    if grid is None:
        # 终身口径 PD 分布更宽，网格需覆盖到 0.9
        grid = np.arange(0.01, 0.901, 0.005)
    scan = threshold_scan(y, p, thresholds=grid)

    rows = []
    for target in target_bad_rates:
        chosen = choose_threshold(scan, target_bad_rate=target, min_approval=min_approval)
        rows.append(
            {
                "target_bad_rate": target,
                "min_approval": min_approval,
                "threshold_pd": float(chosen["threshold"]),
                "approval_rate": float(chosen["approval_rate"]),
                "approved_bad_rate": float(chosen["approved_bad_rate"]),
                "bad_capture": float(chosen["bad_capture"]),
                "rejected_bad_rate": float(chosen["rejected_bad_rate"]),
                "feasible": bool(
                    chosen["approved_bad_rate"] <= target + 1e-9
                    and chosen["approval_rate"] >= min_approval - 1e-9
                ),
            }
        )
    return pd.DataFrame(rows)


def threshold_stability(
    y,
    p_lifetime,
    base_target: float = 0.15,
    perturbation: float = 0.02,
    min_approval: float = 0.50,
) -> pd.DataFrame:
    """检验阈值选择规则对坏账率估计误差的敏感度。

    旧版 choose_threshold 的规则是"满足坏账率<=目标且批准率>=下限的阈值里批准率
    最大者"，这个规则天然落在约束边界上——坏账率估计偏差 1~2pp 就会让选中阈值跳档。
    这里对目标坏账率施加 ±perturbation 的扰动，观察阈值与批准率的漂移幅度。
    """
    targets = (
        base_target - perturbation,
        base_target,
        base_target + perturbation,
    )
    out = threshold_sensitivity(y, p_lifetime, tuple(targets), min_approval)
    base = out.loc[out["target_bad_rate"] == base_target]
    if len(base):
        base_thr = float(base["threshold_pd"].iloc[0])
        base_appr = float(base["approval_rate"].iloc[0])
        out["threshold_shift"] = out["threshold_pd"] - base_thr
        out["approval_shift"] = out["approval_rate"] - base_appr
    return out


def apply_lifetime_conversion(
    scored: pd.DataFrame,
    term_table: pd.DataFrame,
    pd_col: str = "prob_calibrated",
    term_col: str = "term_num",
    out_col: str = "prob_lifetime",
) -> pd.DataFrame:
    """逐笔把 MOB 对齐口径 PD 换算到终身口径，按每笔贷款的期限查 e。

    这是本项目终身换算的标准入口。**必须逐笔按期限换算**：36 期与 60 期的 e 相差
    35%，用总体 e 统一换算会让 60 期贷款的终身风险被低估约 18%、36 期被高估约 11%。

    输入 scored 需含 pd_col 与 term_col；输出新增 out_col、emergence_frac、
    lifetime_clipped 三列。
    """
    if term_table.empty:
        LOGGER.warning("无期限暴露比例表，终身换算退化为原值返回")
        out = scored.copy()
        out[out_col] = out[pd_col]
        out["emergence_frac"] = np.nan
        out["lifetime_clipped"] = False
        return out

    e_map = dict(zip(term_table["segment"].astype(str), term_table["emergence_frac"]))
    e_all = float(e_map.get(ALL_SEGMENT, np.nan))

    out = scored.copy()
    keys = out[term_col].astype(float).round().astype("Int64").astype(str)
    e = keys.map(e_map).astype(float).fillna(e_all).to_numpy()
    p_lt, clipped = to_lifetime_pd(out[pd_col].to_numpy(dtype=float), e)
    out[out_col] = p_lt
    out["emergence_frac"] = e
    out["lifetime_clipped"] = clipped
    LOGGER.info(
        "终身换算完成: %s 笔，clip %s 笔（%.4f%%），平均 e=%.4f",
        f"{len(out):,}",
        f"{int(clipped.sum()):,}",
        100.0 * float(clipped.mean()),
        float(np.nanmean(e)),
    )
    return out


def grade_lifetime_calibration(
    scored: pd.DataFrame,
    term_table: pd.DataFrame,
    by: str = "grade",
    pd_col: str = "prob_calibrated",
    y_col: str = "target",
) -> pd.DataFrame:
    """分段校准表：把每段的 vintage PD 按期限换算到终身口径，再与实测对比。

    旧版 grade_calibration_oot 直接拿 MOB 不对齐的实测坏账率与模型 PD 比，两者
    处在不同表现窗口上，得出的 1.4x"低估倍数"其实是口径差异而非校准问题。
    本表统一换算到终身口径后再比，ratio_lifetime 才是真实的校准误差。

    scored 需已含 term_col（供逐笔期限换算），或已含 emergence_frac / prob_lifetime。
    """
    if term_table.empty:
        return pd.DataFrame()

    work = scored
    if "prob_lifetime" not in work.columns:
        work = apply_lifetime_conversion(work, term_table, pd_col=pd_col)

    rows = []
    for seg, g in work.groupby(by, observed=True):
        p_v = g[pd_col].to_numpy(dtype=float)
        p_lt = g["prob_lifetime"].to_numpy(dtype=float)
        observed_v = float(g[y_col].mean())
        # 实测终身坏账率：用该段实际的平均 e 换算，保持与预测侧同一口径
        e_seg = float(g["emergence_frac"].mean()) if "emergence_frac" in g else np.nan
        observed_lt = observed_v / e_seg if e_seg else np.nan
        mean_p_lt = float(np.mean(p_lt)) if len(p_lt) else np.nan
        rows.append(
            {
                by: seg,
                "count": len(g),
                "emergence_frac": e_seg,
                "pred_pd_vintage": float(np.mean(p_v)) if len(p_v) else np.nan,
                "pred_pd_lifetime": mean_p_lt,
                "observed_vintage_bad_rate": observed_v,
                "observed_lifetime_bad_rate": observed_lt,
                "ratio_lifetime": observed_lt / mean_p_lt if mean_p_lt else np.nan,
                "clipped_n": int(g["lifetime_clipped"].sum()) if "lifetime_clipped" in g else 0,
            }
        )
    return pd.DataFrame(rows)
