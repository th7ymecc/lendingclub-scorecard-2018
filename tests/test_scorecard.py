import unittest

import numpy as np
import pandas as pd

from lendingclub_scorecard.config import PipelineConfig
from lendingclub_scorecard.scorecard import Scorecard, ScorecardBin, ScorecardFeature


def make_card() -> Scorecard:
    cfg = PipelineConfig()
    feature = ScorecardFeature(
        name="x",
        name_cn="测试变量",
        coef=0.5,
        edges=[-np.inf, 0.0, np.inf],
        bins=[
            ScorecardBin("≤ 0", -np.inf, 0.0, woe=-0.4, points=6),
            ScorecardBin("> 0", 0.0, np.inf, woe=0.6, points=-9),
        ],
    )
    return Scorecard(
        base_score=600,
        pdo=20,
        base_odds=4.0,
        factor=cfg.factor,
        offset=cfg.offset,
        base_points=700,
        intercept=-1.0,
        features=[feature],
        grade_cutoffs=cfg.grade_cutoffs,
    )


class ScorecardTest(unittest.TestCase):
    def test_score_frame(self) -> None:
        card = make_card()
        scored = card.score_frame(pd.DataFrame({"x": [-1.0, 1.0]}))
        self.assertEqual(scored["score"].tolist(), [706, 691])
        self.assertEqual(scored["grade"].tolist(), ["A", "A"])

    def test_prob_from_score_base_odds(self) -> None:
        card = make_card()
        # base_odds=4 -> 基准分 600 对应 PD=20%
        self.assertAlmostEqual(float(card.prob_from_score(600)), 0.20, places=6)

    def test_predict_pd(self) -> None:
        card = make_card()
        frame = pd.DataFrame({"x": [-1.0, 1.0]})
        p = card.predict_pd(frame)
        expected = 1 / (1 + np.exp(-(-1.0 + 0.5 * np.array([-0.4, 0.6]))))
        np.testing.assert_allclose(p, expected, rtol=1e-10)

    def test_roundtrip(self) -> None:
        card = make_card()
        restored = Scorecard.from_dict(card.to_dict())
        scored = restored.score_frame(pd.DataFrame({"x": [-1.0, 1.0]}))
        self.assertEqual(scored["score"].tolist(), [706, 691])

    def test_grade_boundaries(self) -> None:
        card = make_card()
        cuts = card.grade_cutoffs
        scores = np.array([cuts["A"] + 1, cuts["A"], cuts["A"] - 1, cuts["B"], cuts["C"], cuts["D"], cuts["D"] - 1])
        grades = card.grade_of(scores)
        self.assertEqual(grades.tolist(), ["A", "A", "B", "B", "C", "D", "E"])


if __name__ == "__main__":
    unittest.main()
