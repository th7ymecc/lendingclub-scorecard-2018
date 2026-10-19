"""LendingClub 数据读取、清洗、时间切分与特征预处理。"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import (
    BAD_STATUSES,
    CATEGORICAL_FEATURES,
    CN_NAMES,
    FEATURE_NAMES,
    ID_COLUMN,
    MOB_DATE_COLUMNS,
    PERFORMANCE_MOB,
    RAW_COLUMNS,
    SENSITIVITY_COLUMNS,
    SENTINEL_FEATURES,
    TARGET,
    TARGET_VINTAGE,
    TERMINAL_STATUSES,
    PipelineConfig,
)

LOGGER = logging.getLogger(__name__)


def _parse_emp_length(value) -> int:
    if pd.isna(value):
        return -1
    text = str(value).strip().lower()
    if text in {"< 1 year", "<1 year"}:
        return 0
    if text in {"10+ years", "10+ year"}:
        return 10
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else -1


def _build_chunk(
    chunk: pd.DataFrame,
    cfg: PipelineConfig,
    snapshot: pd.Timestamp | None = None,
    mob_window: float | None = None,
) -> pd.DataFrame:
    """清洗单个数据块，并计算 MOB 对齐口径所需字段。

    snapshot   数据快照日，用于计算每笔贷款的已观察月数 obs_mob。
    mob_window 统一表现窗口（月），用于生成 vintage 标签。
    """
    snapshot = snapshot if snapshot is not None else pd.Timestamp(cfg.snapshot_date)
    mob_window = float(mob_window if mob_window is not None else cfg.performance_mob)
    chunk = chunk.copy()
    chunk["issue_dt"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
    start = pd.Timestamp(cfg.sample_start)
    end = pd.Timestamp(cfg.sample_end)
    chunk = chunk.loc[chunk["issue_dt"].between(start, end)].copy()
    if chunk.empty:
        return chunk

    chunk["earliest_cr_line_dt"] = pd.to_datetime(
        chunk["earliest_cr_line"], format="%b-%Y", errors="coerce"
    )
    chunk["credit_hist_months"] = (
        (chunk["issue_dt"] - chunk["earliest_cr_line_dt"]).dt.days / 30.44
    )
    chunk.loc[chunk["credit_hist_months"] < 0, "credit_hist_months"] = np.nan

    chunk["term_num"] = (
        chunk["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    )
    chunk["emp_length"] = chunk["emp_length"].map(_parse_emp_length).astype(int)

    income = chunk["annual_inc"].replace(0, np.nan)
    total_acc = chunk["total_acc"].replace(0, np.nan)
    chunk["loan_income_ratio"] = chunk["loan_amnt"] / income
    chunk["revol_bal_income_ratio"] = chunk["revol_bal"] / income
    chunk["total_bal_income_ratio"] = chunk["total_bal_ex_mort"] / income
    chunk["open_acc_ratio"] = chunk["open_acc"] / total_acc

    chunk["bad_flag"] = np.where(
        chunk["loan_status"].isin(BAD_STATUSES),
        1,
        np.where(chunk["loan_status"] == "Fully Paid", 0, np.nan),
    )
    chunk["resolved"] = chunk["loan_status"].isin(TERMINAL_STATUSES)

    # ---------------- MOB 对齐口径 ----------------
    # 事件时点：终态贷款的实际结清/核销月份；未终态贷款无事件时点。
    chunk["last_pymnt_dt"] = pd.to_datetime(
        chunk["last_pymnt_d"], format="%b-%Y", errors="coerce"
    )
    chunk["event_mob"] = (chunk["last_pymnt_dt"] - chunk["issue_dt"]).dt.days / 30.44
    # 已观察月数：相对数据快照日，决定这笔贷款的标签最多能观察到第几个月。
    chunk["obs_mob"] = (snapshot - chunk["issue_dt"]).dt.days / 30.44

    is_bad = chunk["loan_status"].isin(BAD_STATUSES)
    # 事件时点缺失时用 term 兜底：终态贷款最晚在到期月结清。
    event_mob = chunk["event_mob"].fillna(chunk["term_num"])
    bad_in_window = is_bad & (event_mob <= mob_window)

    # vintage 标签：窗口内是否核销。obs_mob < mob_window 的样本成熟度不足，
    # 无法判断窗口内是否核销，置为 NaN 并从建模样本中剔除。
    chunk[TARGET_VINTAGE] = np.where(
        chunk["obs_mob"] >= mob_window,
        np.where(bad_in_window, 1.0, 0.0),
        np.nan,
    )
    chunk["evaluable"] = chunk[TARGET_VINTAGE].notna()

    chunk["issue_year"] = chunk["issue_dt"].dt.year
    chunk["issue_quarter"] = (
        chunk["issue_dt"].dt.year.astype(str)
        + "Q"
        + chunk["issue_dt"].dt.quarter.astype(str)
    )
    return chunk


def build_base(cfg: PipelineConfig) -> pd.DataFrame:
    """分块读取原始 CSV，保留建模所需列并计算 MOB 对齐口径字段。

    样本范围为 cfg.sample_start ~ cfg.sample_end（当前为 2012-01-01 ~ 2018-12-31），
    其中 2012 年仅供 Vintage 客群结构展示，建模切分从 2013 年开始。
    """
    path = Path(cfg.data_dir) / cfg.raw_file
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    usecols = list(dict.fromkeys(RAW_COLUMNS))
    LOGGER.info("读取 LendingClub CSV（分块），保留列 %s 个", len(usecols))
    snapshot = pd.Timestamp(cfg.snapshot_date)
    mob_window = float(cfg.performance_mob)
    parts: list[pd.DataFrame] = []
    total = 0
    for i, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=250_000, low_memory=False)
    ):
        filtered = _build_chunk(chunk, cfg, snapshot=snapshot, mob_window=mob_window)
        total += len(filtered)
        if len(filtered):
            parts.append(filtered)
        LOGGER.info("分块 %s 完成，累计样本 %s", i + 1, f"{total:,}")

    if not parts:
        raise RuntimeError(
            f"没有读取到 {cfg.sample_start} ~ {cfg.sample_end} 的样本，"
            "请检查 issue_d 与数据文件"
        )

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("issue_dt").reset_index(drop=True)
    df.insert(0, ID_COLUMN, np.arange(1, len(df) + 1))

    # 明显不合理的记录不进入建模
    valid = (
        df["loan_amnt"].gt(0)
        & df["annual_inc"].gt(0)
        & df["issue_dt"].notna()
    )
    removed = int((~valid).sum())
    if removed:
        LOGGER.info("剔除贷款金额/收入/日期异常记录 %s 条", f"{removed:,}")
        df = df.loc[valid].reset_index(drop=True)
        df[ID_COLUMN] = np.arange(1, len(df) + 1)

    return df


def load_base(cfg: PipelineConfig, refresh: bool = False) -> pd.DataFrame:
    """优先读取缓存；缓存可通过 --refresh-data 强制重建。"""
    cache = Path(cfg.output_dir) / "data" / "base_all.pkl"
    if cache.exists() and not refresh:
        LOGGER.info("读取数据缓存: %s", cache)
        return pd.read_pickle(cache)

    df = build_base(cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    LOGGER.info("数据缓存已写入: %s（%s 行）", cache, f"{len(df):,}")
    return df


def _require_mob_columns(df: pd.DataFrame) -> None:
    """校验 MOB 对齐口径所需的列是否存在。

    这些列由 _build_chunk 生成。若缺失，通常说明读到了旧版数据缓存
    （outputs/data/base_all.pkl），此时应加 --refresh-data 重建，而不是让
    下游抛出一个难以定位的 KeyError。
    """
    required = ("evaluable", "obs_mob", "event_mob", "term_num", TARGET_VINTAGE)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"缺少 MOB 对齐口径所需列 {missing}。"
            "这通常是因为读取了旧版数据缓存，请用 --refresh-data 重建 base_all.pkl"
        )


def time_split(
    df: pd.DataFrame, cfg: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 issue_d 做时间切分，只保留 MOB 对齐口径下可评估的样本。

    与旧版的根本区别：
      旧版筛选 `resolved == True`，等于让每笔贷款停在自己的终态时点，2013 年样本
      平均观察到 47 个月、2018Q4 只有 2 个月，坏账率不可比且方向随成熟度非单调变化。
      本版筛选 `evaluable == True`（obs_mob >= 表现窗口），所有样本统一截到 MOB12，
      标签口径完全一致，坏账率可跨期公平比较。

    代价是绝对数值变小（MOB12 vintage 坏账率约为终身坏账率的 30%），业务决策需
    换算回终身口径。换算在**概率空间**做除法：p_终身 = p_vintage / e，其中 e 是
    坏样本在 MOB12 前的暴露比例，且必须按贷款期限分段取值（36 期与 60 期相差
    约 35%）。详见 calibration.py，不要用 logit 平移代替。
    """
    _require_mob_columns(df)
    evaluable = df.loc[df["evaluable"]].copy()
    dropped = int(len(df) - len(evaluable))
    LOGGER.info(
        "MOB%d 对齐切分: 总样本 %s，成熟度不足剔除 %s（%.1f%%），可评估 %s",
        cfg.performance_mob,
        f"{len(df):,}",
        f"{dropped:,}",
        100.0 * dropped / max(len(df), 1),
        f"{len(evaluable):,}",
    )

    train = evaluable.loc[
        evaluable["issue_dt"].between(pd.Timestamp(cfg.train_start), pd.Timestamp(cfg.train_end))
    ].copy()
    valid = evaluable.loc[
        evaluable["issue_dt"].between(pd.Timestamp(cfg.valid_start), pd.Timestamp(cfg.valid_end))
    ].copy()
    oot = evaluable.loc[
        evaluable["issue_dt"].between(pd.Timestamp(cfg.oot_start), pd.Timestamp(cfg.oot_end))
    ].copy()
    for name, part in (("train", train), ("valid", valid), ("oot", oot)):
        if part.empty:
            raise RuntimeError(f"{name} 切分结果为空，请检查时间配置与 MOB 口径")
        LOGGER.info(
            "%s: %s 行，MOB%d vintage 坏账率 %.2f%%，时间 %s ~ %s",
            name,
            f"{len(part):,}",
            cfg.performance_mob,
            part[TARGET_VINTAGE].mean() * 100,
            part["issue_dt"].min().strftime("%Y-%m"),
            part["issue_dt"].max().strftime("%Y-%m"),
        )
    return (
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        oot.reset_index(drop=True),
    )


def vintage_tables(base_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vintage 客群结构与坏账率。

    返回两张表：
      composition      各放款季度的客群结构，含三种口径的坏账率对照列
      resolved_vintage 仅已终态样本的季度汇总（旧口径，保留供对比）
    """
    _require_mob_columns(base_all)
    all_df = base_all.copy()
    all_df["status_group"] = np.select(
        [
            all_df["loan_status"].isin(BAD_STATUSES),
            all_df["loan_status"] == "Fully Paid",
            all_df["loan_status"].isin(["Current", "Issued"]),
            all_df["loan_status"].str.startswith("Late", na=False)
            | all_df["loan_status"].eq("In Grace Period"),
        ],
        ["bad", "paid", "current", "late"],
        default="other",
    )
    composition = (
        all_df.groupby("issue_quarter", observed=True)
        .agg(
            total=("loan_id", "size"),
            resolved=("resolved", "sum"),
            bad=("bad_flag", lambda s: int((s == 1).sum())),
            current=("status_group", lambda s: int((s == "current").sum())),
            late=("status_group", lambda s: int((s == "late").sum())),
        )
        .reset_index()
    )
    composition["resolved_rate"] = composition["resolved"] / composition["total"]
    composition["current_rate"] = composition["current"] / composition["total"]
    composition["late_rate"] = composition["late"] / composition["total"]
    composition["bad_rate"] = composition["bad"] / composition["resolved"].replace(0, np.nan)

    # MOB 对齐口径：每个季度可评估样本的 vintage 坏账率（分母为可评估全 cohort）
    evaluable = all_df.loc[all_df["evaluable"]].copy()
    vintage_mob = (
        evaluable.groupby("issue_quarter", observed=True)
        .agg(
            evaluable=("loan_id", "size"),
            bad_v=(TARGET_VINTAGE, "sum"),
            bad_rate_v=(TARGET_VINTAGE, "mean"),
        )
        .reset_index()
    )
    composition = composition.merge(vintage_mob, on="issue_quarter", how="left")

    resolved = all_df.loc[all_df["resolved"]].copy()
    resolved_vintage = (
        resolved.groupby("issue_quarter", observed=True)
        .agg(
            resolved=("loan_id", "size"),
            bad=("bad_flag", "sum"),
            bad_rate=("bad_flag", "mean"),
            avg_loan_amnt=("loan_amnt", "mean"),
        )
        .reset_index()
    )
    return composition, resolved_vintage


def data_quality_summary(base_all: pd.DataFrame, mob: int = PERFORMANCE_MOB) -> pd.DataFrame:
    """各放款年的样本量、成熟度与两种口径下的坏账率对比。

    同时给出旧版"只留终态样本"口径与新版"MOB 对齐 vintage"口径，两者差异即
    删失偏差的直接量化。旧口径的 bias 方向随成熟度非单调变化：中等成熟度时
    高估坏账率（核销比正常结清更早暴露），极低成熟度时低估坏账率。
    """
    rows = []
    for year, group in base_all.groupby("issue_year"):
        resolved = group.loc[group["resolved"]]
        evaluable = group.loc[group["evaluable"]]
        legacy_rate = resolved[TARGET].mean() if len(resolved) else np.nan
        vintage_rate = evaluable[TARGET_VINTAGE].mean() if len(evaluable) else np.nan
        # 旧口径高估倍数：>1 表示旧口径偏高。vintage 口径需换算到终身才可比，
        # 故此处另给一列"换算到终身的旧口径倍数"作参考。
        rows.append(
            {
                "issue_year": int(year),
                "total": len(group),
                "resolved": len(resolved),
                "resolved_rate": len(resolved) / len(group),
                "evaluable": len(evaluable),
                "evaluable_rate": len(evaluable) / len(group),
                "avg_obs_mob": group["obs_mob"].mean(),
                "avg_event_mob": group.loc[group["event_mob"].notna(), "event_mob"].mean(),
                "legacy_bad": int(resolved[TARGET].sum()),
                "legacy_bad_rate": legacy_rate,
                f"vintage_mob{mob}_bad_rate": vintage_rate,
                "legacy_over_vintage_ratio": legacy_rate / vintage_rate if vintage_rate else np.nan,
                "current_rate": (group["loan_status"].isin(["Current", "Issued"])).mean(),
            }
        )
    return pd.DataFrame(rows)


def emergence_curve(base_all: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """坏账暴露曲线：用完全成熟的 cohort 实测坏样本随 MOB 的暴露比例。

    这是把 MOB 对齐口径换算到终身口径的唯一可靠依据。做法：
      1. 只取 2013-2014 放款且已观察 >= 60 个月的样本（终身坏账率已收敛）；
      2. 统计这些样本中，坏样本在 MOB <= k 前暴露的累计比例；
      3. 该比例即换算系数的倒数：终身坏账率 ≈ vintage_mob_k 坏账率 / 暴露比例。

    实测验证：MOB12 vintage 坏账率 4.84% / 暴露比例 30.40% = 15.92%，
    与成熟 cohort 实测终身坏账率 15.91% 吻合，说明换算关系成立。
    """
    snapshot = pd.Timestamp(cfg.snapshot_date)
    mature = base_all.loc[
        (base_all["issue_dt"] < "2015-01-01")
        & (base_all["obs_mob"] >= 60)
        & base_all[TARGET].notna()
    ].copy()
    if mature.empty:
        LOGGER.warning("无成熟 cohort，坏账暴露曲线无法计算")
        return pd.DataFrame()

    lifetime_rate = float(mature[TARGET].mean())
    n_total = len(mature)
    bad_mask = mature[TARGET] == 1
    event = mature["event_mob"].fillna(mature["term_num"])
    n_bad_total = int(bad_mask.sum())

    rows = []
    for k in (6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60):
        bad_exposed = int((bad_mask & (event <= k)).sum())
        good_settled = int(((~bad_mask) & (event <= k)).sum())
        n_settled = bad_exposed + good_settled
        # vintage 口径坏账率：分母为全 cohort（含窗口内仍未终态的样本）
        vintage_rate = bad_exposed / n_total if n_total else np.nan
        emergence = bad_exposed / n_bad_total if n_bad_total else np.nan
        rows.append(
            {
                "mob": k,
                "cohort": "2013-2014（obs_mob>=60）",
                "n_cohort": n_total,
                "lifetime_bad_rate": lifetime_rate,
                "bad_exposed_n": bad_exposed,
                "bad_emergence_frac": emergence,
                "good_settled_n": good_settled,
                "settled_rate": n_settled / n_total if n_total else np.nan,
                "vintage_bad_rate": vintage_rate,
                # 换算系数：vintage -> 终身。等于 1 / 暴露比例。
                "to_lifetime_factor": 1.0 / emergence if emergence else np.nan,
                # 校验列：vintage / 暴露比例 应逼近终身坏账率
                "check_lifetime_estimate": vintage_rate / emergence if emergence else np.nan,
            }
        )
    curve = pd.DataFrame(rows)
    LOGGER.info(
        "坏账暴露曲线: 成熟 cohort n=%s，终身坏账率 %.4f，MOB%d 暴露 %.4f",
        f"{n_total:,}",
        lifetime_rate,
        cfg.performance_mob,
        float(curve.loc[curve["mob"] == cfg.performance_mob, "bad_emergence_frac"].iloc[0]),
    )
    return curve


class Preprocessor:
    """数值截断/缺失填补/缺失标记/类别 one-hot。"""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.medians_: pd.Series | None = None
        self.caps_: dict[str, tuple[float, float]] = {}
        self.missing_flags_: list[str] = []
        self.cat_levels_: dict[str, list[str]] = {}
        self.cat_bases_: dict[str, str] = {}
        self.feature_names_: list[str] = []
        self.dummy_specs_: list[tuple[str, str]] = []

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        work = df.copy()
        numeric = [f for f in FEATURE_NAMES if f in work.columns]

        # 缺失标记：在填补之前判断
        for col in numeric:
            if col in work.columns and work[col].isna().mean() > self.cfg.missing_flag_threshold:
                self.missing_flags_.append(col)

        # 哨兵值填补
        for col in SENTINEL_FEATURES:
            work[col] = work[col].fillna(999)
        for col in numeric:
            if col in SENTINEL_FEATURES:
                continue
            median = work[col].median()
            self.medians_ = self.medians_ if self.medians_ is not None else pd.Series(dtype=float)
            self.medians_[col] = median
            work[col] = work[col].fillna(median)

        # 截断极端值（哨兵列不做截断，保留 999 缺失箱）
        for col in numeric:
            if col in SENTINEL_FEATURES:
                continue
            lo = float(work[col].quantile(self.cfg.cap_lower_q))
            hi = float(work[col].quantile(self.cfg.cap_upper_q))
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                self.caps_[col] = (lo, hi)
                work[col] = work[col].clip(lo, hi)

        self.medians_ = self.medians_.fillna(0.0) if self.medians_ is not None else pd.Series(dtype=float)

        # 类别编码：保留高频水平，其余归入 OTHER
        for col in CATEGORICAL_FEATURES:
            series = work[col].astype(str).str.strip().fillna("Unknown")
            value_counts = series.value_counts(normalize=True)
            levels = [str(v) for v in value_counts[value_counts >= 0.01].index]
            if "Unknown" not in levels:
                levels.append("Unknown")
            if "OTHER" not in levels:
                levels.append("OTHER")
            base = levels[0]
            self.cat_bases_[col] = base
            self.cat_levels_[col] = [level for level in levels if level != base]

        self.feature_names_ = list(numeric)
        for col in self.missing_flags_:
            self.feature_names_.append(f"{col}_missing")
        for col in CATEGORICAL_FEATURES:
            for level in self.cat_levels_[col]:
                self.dummy_specs_.append((col, level))
                self.feature_names_.append(f"{col}={level}")

        # 防泄漏断言：任何 MOB 相关字段（决定标签观察到第几个月）与敏感性对照列
        # 都不得进入特征矩阵。MOB 字段是贷后信息，若入模等于用未来信息预测；
        # 敏感性对照列（FICO/grade）必须只在消融实验中出现，否则"不用 FICO"的
        # 口径声明就不成立。
        forbidden = set(MOB_DATE_COLUMNS) | set(SENSITIVITY_COLUMNS) | {
            "last_pymnt_dt",
            "event_mob",
            "obs_mob",
            TARGET,
            TARGET_VINTAGE,
            "loan_status",
            "resolved",
            "evaluable",
            "grade",
        }
        leaked = [f for f in self.feature_names_ if f in forbidden]
        if leaked:
            raise RuntimeError(f"检测到 MOB/标签/敏感性字段泄漏进特征矩阵: {leaked}")

        LOGGER.info(
            "预处理拟合完成: 数值 %s，缺失标记 %s，类别哑变量 %s，总特征 %s",
            len(numeric),
            len(self.missing_flags_),
            len(self.dummy_specs_),
            len(self.feature_names_),
        )
        return self

    def _map_categories(self, series: pd.Series, col: str) -> pd.Series:
        levels = set(self.cat_levels_[col])
        base = self.cat_bases_[col]
        text = series.astype(str).str.strip().fillna("Unknown")
        return text.where(text.isin(levels) | (text == base), "OTHER")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("Preprocessor 尚未 fit")
        work = df.copy()
        raw = work.copy()
        out = pd.DataFrame(index=work.index)

        # 数值
        for col in [f for f in FEATURE_NAMES if f in work.columns]:
            if col in SENTINEL_FEATURES:
                out[col] = work[col].fillna(999).astype(float)
            else:
                out[col] = work[col].fillna(self.medians_.get(col, 0.0)).astype(float)
                if col in self.caps_:
                    lo, hi = self.caps_[col]
                    out[col] = out[col].clip(lo, hi)

        # 缺失标记
        for col in self.missing_flags_:
            out[f"{col}_missing"] = raw[col].isna().astype(int)

        # 类别哑变量
        for col, level in self.dummy_specs_:
            mapped = self._map_categories(work[col], col)
            out[f"{col}={level}"] = (mapped == level).astype(int)

        # 保证列顺序和缺失列
        for col in self.feature_names_:
            if col not in out.columns:
                out[col] = 0
        return out[self.feature_names_]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        return joblib.load(path)


def feature_label(name: str) -> str:
    if name in CN_NAMES:
        return CN_NAMES[name]
    if name.endswith("_missing"):
        base = name[: -len("_missing")]
        return f"缺失标记：{CN_NAMES.get(base, base)}"
    if "=" in name:
        col, level = name.split("=", 1)
        col_cn = {
            "term": "期限",
            "home_ownership": "房屋所有权",
            "verification_status": "收入核验",
            "purpose": "贷款用途",
            "initial_list_status": "初始挂牌状态",
            "application_type": "申请类型",
        }.get(col, col)
        return f"{col_cn}={level}"
    return name
