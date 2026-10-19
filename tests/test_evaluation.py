import unittest

import numpy as np
import pandas as pd

from lendingclub_scorecard.evaluation import (
    binary_metrics,
    calibration_metrics,
    choose_threshold,
    grade_table,
    psi,
    psi_level,
    swap_set_analysis,
    threshold_confusion,
    threshold_scan,
)


class EvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(2)
        self.y = rng.binomial(1, 0.2, 5000)
        self.p = np.where(self.y == 1, rng.uniform(0.35, 0.9, 5000), rng.uniform(0.01, 0.5, 5000))

    def test_metrics(self) -> None:
        metrics = binary_metrics(self.y, self.p)
        self.assertGreater(metrics["auc"], 0.8)
        self.assertGreater(metrics["ks"], 0)
        self.assertAlmostEqual(metrics["mean_pd"], float(np.mean(self.p)))

    def test_psi_binary(self) -> None:
        a = np.array([0] * 900 + [1] * 100)
        b = np.array([0] * 800 + [1] * 200)
        value = psi(a, b)
        self.assertGreater(value, 0.05)
        self.assertEqual(psi(a, a), 0.0)

    def test_psi_levels(self) -> None:
        self.assertEqual(psi_level(0.05), "稳定")
        self.assertEqual(psi_level(0.15), "需关注")
        self.assertEqual(psi_level(0.30), "显著漂移")

    def test_threshold_scan_and_choice(self) -> None:
        scan = threshold_scan(self.y, self.p)
        self.assertGreater(len(scan), 10)
        chosen = choose_threshold(scan, target_bad_rate=0.3, min_approval=0.5)
        self.assertLessEqual(float(chosen["approved_bad_rate"]), 0.3)
        conf = threshold_confusion(self.y, self.p, float(chosen["threshold"]))
        self.assertEqual(conf["approved_n"] + conf["rejected_n"], len(self.y))

    def test_grade_table(self) -> None:
        grade = pd.Series(["A"] * 1000 + ["B"] * 1000 + ["C"] * 1000 + ["D"] * 1000 + ["E"] * 1000)
        score = pd.Series(np.linspace(700, 500, 5000))
        table = grade_table(self.y, score, grade)
        self.assertEqual(len(table), 5)
        self.assertEqual(int(table["count"].sum()), 5000)

    def test_calibration_metrics(self) -> None:
        metrics = calibration_metrics(self.y, self.p)
        self.assertIn("calibration_slope", metrics)
        self.assertIn("ece", metrics)
        self.assertGreaterEqual(metrics["ece"], 0)
        self.assertGreaterEqual(metrics["mce"], metrics["ece"])
        self.assertAlmostEqual(metrics["actual"], float(np.mean(self.y)))

    def test_swap_set(self) -> None:
        p2 = np.clip(self.p + np.random.default_rng(7).normal(0, 0.05, len(self.p)), 0.001, 0.999)
        summary, groups = swap_set_analysis(self.y, self.p, p2, n_approve=3000)
        self.assertEqual(summary["swap_in_n"], summary["swap_out_n"])
        self.assertEqual(int(groups["count"].sum()), len(self.y))
        self.assertAlmostEqual(
            summary["net_bad_avoided"],
            summary["swap_out_bad"] - summary["swap_in_bad"],
        )
        self.assertGreaterEqual(summary["overlap_rate"], 0)
        self.assertLessEqual(summary["overlap_rate"], 1)


if __name__ == "__main__":
    unittest.main()
