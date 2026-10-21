import unittest

import numpy as np

from lendingclub_scorecard.calibration import threshold_sensitivity, threshold_stability


class WeightedThresholdTest(unittest.TestCase):
    """终身加权标签是 0/1/e，阈值函数必须保持 float 权重。"""

    def setUp(self) -> None:
        e = 0.339
        # 100 个坏样本，权重 1/e；900 个好样本。加权坏账率 = 100*(1/e)/1000。
        self.y = np.array([1.0 / e] * 100 + [0.0] * 900)
        self.p = np.array([0.6] * 100 + [0.1] * 900)
        self.expected_bad_rate = 100.0 * (1.0 / e) / 1000.0

    def test_sensitivity_keeps_weighted_labels(self) -> None:
        out = threshold_sensitivity(
            self.y,
            self.p,
            target_bad_rates=(0.5,),
            min_approval=0.0,
            grid=np.array([0.5, 0.9]),
        )
        row = out.iloc[0]
        self.assertAlmostEqual(float(row["threshold_pd"]), 0.9)
        # int 截断会得到 0.20，正确值是 0.294985...
        self.assertAlmostEqual(float(row["approved_bad_rate"]), self.expected_bad_rate, places=10)
        self.assertGreater(float(row["approved_bad_rate"]), 0.25)

    def test_stability_keeps_weighted_labels(self) -> None:
        out = threshold_stability(
            self.y,
            self.p,
            base_target=0.5,
            perturbation=0.4,
            min_approval=0.0,
        )
        base = out.loc[np.isclose(out["target_bad_rate"], 0.5)].iloc[0]
        self.assertAlmostEqual(float(base["approved_bad_rate"]), self.expected_bad_rate, places=10)
        self.assertAlmostEqual(float(base["threshold_shift"]), 0.0)
        self.assertAlmostEqual(float(base["approval_shift"]), 0.0)


if __name__ == "__main__":
    unittest.main()
