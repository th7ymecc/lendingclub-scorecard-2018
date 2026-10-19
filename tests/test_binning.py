import unittest

import numpy as np
import pandas as pd

from lendingclub_scorecard.binning import (
    assign_bin_ids,
    bin_label,
    fit_binning,
)
from lendingclub_scorecard.config import FeatureSpec, PipelineConfig


class BinningTest(unittest.TestCase):
    def test_assign_boundaries(self) -> None:
        edges = [-np.inf, 0, 1, 2, np.inf]
        values = np.array([-1, 0, 0.5, 1, 1.5, 2, 3])
        self.assertEqual(assign_bin_ids(values, edges).tolist(), [0, 0, 1, 1, 2, 2, 3])

    def test_labels(self) -> None:
        self.assertEqual(bin_label(-np.inf, 0), "≤ 0")
        self.assertEqual(bin_label(0, 1), "(0, 1]")
        self.assertEqual(bin_label(2, np.inf), "> 2")

    def test_continuous_monotonic(self) -> None:
        rng = np.random.default_rng(1)
        x = pd.Series(rng.gamma(2.0, 1.0, 20000))
        p = 1 / (1 + np.exp(-(-2.0 + 1.5 * np.log1p(x))))
        y = pd.Series(rng.binomial(1, p))
        spec = FeatureSpec("loan_income_ratio", "贷款收入比", "", "衍生变量")
        result = fit_binning(x, y, spec, PipelineConfig())
        self.assertGreater(result.iv, 0)
        self.assertEqual(int(result.bins["count"].sum()), len(x))
        self.assertTrue(np.all(np.diff(result.bins["bad_rate"]) >= -1e-12))

    def test_binary_feature(self) -> None:
        x = pd.Series([0] * 800 + [1] * 200)
        y = pd.Series([0] * 700 + [1] * 100 + [0] * 80 + [1] * 120)
        spec = FeatureSpec("term=60 months", "期限=60个月", "", "衍生", kind="binary")
        result = fit_binning(x, y, spec, PipelineConfig())
        self.assertEqual(len(result.bins), 2)
        self.assertGreater(result.bins["bad_rate"].iloc[1], result.bins["bad_rate"].iloc[0])


if __name__ == "__main__":
    unittest.main()
