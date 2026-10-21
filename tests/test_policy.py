import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lendingclub_scorecard.cli import _term_num
from lendingclub_scorecard.policy import DecisionPolicy


def make_policy() -> DecisionPolicy:
    return DecisionPolicy(
        platt_a=1.0,
        platt_b=0.0,
        emergence_by_term={"36": 0.5, "60": 0.25, "__ALL__": 0.5},
        emergence_default=0.5,
        threshold_lifetime=0.27,
        grade_pd_cutoffs=(0.08, 0.12, 0.18, 0.25),
    )


class DecisionPolicyTest(unittest.TestCase):
    def test_calibration_identity(self) -> None:
        policy = make_policy()
        p = np.array([0.01, 0.2, 0.7, 0.99])
        np.testing.assert_allclose(policy.calibrate(p), p, rtol=0, atol=1e-10)

    def test_lifetime_conversion_and_clip(self) -> None:
        policy = make_policy()
        p_lt, e, clipped = policy.to_lifetime([0.1, 0.1, 0.6], [36, 60, 60])
        np.testing.assert_allclose(e, [0.5, 0.25, 0.25])
        np.testing.assert_allclose(p_lt, [0.2, 0.4, 1.0])
        self.assertEqual(clipped.tolist(), [False, False, True])

    def test_grade_boundaries(self) -> None:
        policy = make_policy()
        grades = policy.grade_of([0.0799, 0.08, 0.12, 0.18, 0.25])
        self.assertEqual(grades.tolist(), ["A", "B", "C", "D", "E"])

    def test_approval_uses_lifetime_threshold(self) -> None:
        policy = make_policy()
        self.assertEqual(
            policy.approve([0.1, 0.2699, 0.27, 0.4]).tolist(),
            [True, True, False, False],
        )

    def test_decide_columns(self) -> None:
        policy = make_policy()
        out = policy.decide([0.2, 0.3], [36, 60])
        self.assertEqual(
            list(out.columns),
            [
                "prob_mob12_raw",
                "prob_mob12_calibrated",
                "prob_lifetime",
                "grade",
                "approve",
                "term_num",
                "emergence_frac",
                "lifetime_clipped",
            ],
        )
        np.testing.assert_allclose(out["prob_lifetime"], [0.4, 1.0])

    def test_roundtrip(self) -> None:
        policy = make_policy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision_policy.json"
            policy.save(path)
            loaded = DecisionPolicy.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(loaded, policy)

    def test_term_num_parses_months(self) -> None:
        parsed = _term_num(pd.Series(["36 months", "60 months", None])).tolist()
        self.assertEqual(parsed[:2], [36.0, 60.0])
        self.assertTrue(np.isnan(parsed[2]))

    def test_from_metadata(self) -> None:
        meta = {
            "calibration": {"platt_a": 1.0, "platt_b": 0.0},
            "emergence_by_term": {"36": 0.5, "60": 0.25, "__ALL__": 0.5},
            "threshold": {"selected": 0.27, "target_bad_rate": 0.15},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_metadata.json"
            path.write_text(json.dumps(meta), encoding="utf-8")
            policy = DecisionPolicy.from_metadata(tmp)
        self.assertEqual(policy.threshold_lifetime, 0.27)
        self.assertEqual(policy.emergence_by_term["60"], 0.25)


if __name__ == "__main__":
    unittest.main()
