import unittest

import numpy as np
import pandas as pd

from lendingclub_scorecard.config import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    PERFORMANCE_MOB,
    SNAPSHOT_DATE,
    TARGET,
    TARGET_VINTAGE,
    PipelineConfig,
)
from lendingclub_scorecard.data import Preprocessor, time_split, vintage_tables


def make_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(index=range(n))
    for col in FEATURE_NAMES:
        if col in {"emp_length"}:
            frame[col] = rng.integers(-1, 11, n)
        elif col in {
            "mths_since_last_delinq",
            "mths_since_last_record",
            "mths_since_last_major_derog",
            "mths_since_recent_revol_delinq",
            "mths_since_recent_bc_dlq",
            "mths_since_recent_inq",
        }:
            frame[col] = rng.uniform(0, 100, n)
            frame.loc[rng.choice(n, 40, replace=False), col] = np.nan
        else:
            frame[col] = rng.uniform(0, 100, n)
    frame["term"] = rng.choice(["36 months", "60 months"], n)
    frame["home_ownership"] = rng.choice(["MORTGAGE", "RENT", "OWN"], n)
    frame["verification_status"] = rng.choice(["Verified", "Not Verified", "Source Verified"], n)
    frame["purpose"] = rng.choice(["debt_consolidation", "credit_card", "home_improvement", "other"], n)
    frame["initial_list_status"] = rng.choice(["w", "f"], n)
    frame["application_type"] = rng.choice(["Individual", "Joint App"], n)
    frame["loan_amnt"] = rng.uniform(1000, 35000, n)
    frame["annual_inc"] = rng.uniform(20000, 200000, n)
    frame[TARGET] = rng.binomial(1, 0.2, n)
    frame["loan_status"] = np.where(frame[TARGET] == 1, "Charged Off", "Fully Paid")
    frame["resolved"] = True
    dates = pd.to_datetime(
        rng.choice(pd.date_range("2013-01-01", "2018-12-31", freq="D"), n)
    )
    frame["issue_dt"] = dates
    frame["issue_year"] = dates.year
    frame["issue_quarter"] = (
        dates.year.astype(str) + "Q" + dates.quarter.astype(str)
    )
    frame["loan_id"] = np.arange(1, n + 1)

    # ---- MOB 对齐口径所需列（与 _build_chunk 的数据契约保持一致）----
    frame["term_num"] = frame["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    snapshot = pd.Timestamp(SNAPSHOT_DATE)
    frame["obs_mob"] = (snapshot - frame["issue_dt"]).dt.days / 30.44
    # 事件 MOB：终态样本的结清/核销时点，合成数据用一个落在 [0, obs_mob] 的值
    frame["event_mob"] = np.minimum(
        rng.uniform(1, 40, n), frame["obs_mob"]
    ).clip(lower=0)
    frame["bad_by_mob"] = ((frame[TARGET] == 1) & (frame["event_mob"] <= PERFORMANCE_MOB)).astype(int)
    # vintage 标签：obs_mob >= 窗口才可评估，否则 NaN
    frame[TARGET_VINTAGE] = np.where(
        frame["obs_mob"] >= PERFORMANCE_MOB,
        frame["bad_by_mob"].astype(float),
        np.nan,
    )
    frame["evaluable"] = frame[TARGET_VINTAGE].notna()
    return frame


class DataTest(unittest.TestCase):
    def test_preprocessor(self) -> None:
        cfg = PipelineConfig()
        frame = make_frame()
        train = frame[frame["issue_year"] <= 2014]
        pre = Preprocessor(cfg).fit(train)
        transformed = pre.transform(frame)
        self.assertEqual(transformed.shape[0], len(frame))
        self.assertFalse(transformed.isna().any().any())
        self.assertEqual(len(pre.feature_names_), transformed.shape[1])

    def test_time_split(self) -> None:
        cfg = PipelineConfig()
        frame = make_frame(800)
        train, valid, oot = time_split(frame, cfg)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(valid), 0)
        self.assertGreater(len(oot), 0)
        self.assertTrue((train["issue_year"] <= 2015).all())
        self.assertTrue((valid["issue_year"] == 2016).all())
        self.assertTrue((oot["issue_year"] >= 2017).all())
        # 只保留可评估样本：evaluable 必须全为 True，vintage 标签非空
        for part in (train, valid, oot):
            self.assertTrue(part["evaluable"].all())
            self.assertTrue(part[TARGET_VINTAGE].notna().all())
        # OOT 上界不超过配置的 oot_end（快照日 − 表现窗口）
        self.assertLessEqual(oot["issue_dt"].max(), pd.Timestamp(cfg.oot_end))

    def test_time_split_drops_immature(self) -> None:
        """成熟度不足（obs_mob < 窗口）的样本必须被排除，不得进入建模样本。"""
        cfg = PipelineConfig()
        frame = make_frame(800)
        immature = frame.loc[~frame["evaluable"]]
        self.assertGreater(len(immature), 0, "合成数据应包含成熟度不足的样本")
        train, valid, oot = time_split(frame, cfg)
        kept_ids = set(
            pd.concat([train, valid, oot])["loan_id"]
        )
        self.assertFalse(
            set(immature["loan_id"]) & kept_ids,
            "成熟度不足样本不应进入任何切分",
        )

    def test_require_mob_columns_raises_on_stale_cache(self) -> None:
        """缺少 MOB 列（旧版缓存）时应给出清晰报错，而非裸 KeyError。"""
        frame = make_frame().drop(columns=["evaluable", "obs_mob"])
        with self.assertRaises(KeyError) as ctx:
            time_split(frame, PipelineConfig())
        self.assertIn("refresh-data", str(ctx.exception))

    def test_vintage_tables(self) -> None:
        frame = make_frame()
        comp, resolved = vintage_tables(frame)
        self.assertEqual(int(comp["total"].sum()), len(frame))
        self.assertEqual(int(resolved["resolved"].sum()), len(frame))
        # 新增的 MOB 对齐口径列必须存在
        for col in ("evaluable", "bad_v", "bad_rate_v"):
            self.assertIn(col, comp.columns)


if __name__ == "__main__":
    unittest.main()

