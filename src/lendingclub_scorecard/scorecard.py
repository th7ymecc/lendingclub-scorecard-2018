"""标准信用评分卡：WOE 逻辑回归 → 整数分。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .binning import BinningResult, assign_bin_ids
from .config import GRADE_LABELS, GRADE_ORDER, PipelineConfig
from .model import LogitModel


@dataclass
class ScorecardBin:
    label: str
    lower: float
    upper: float
    woe: float
    points: int
    count: int = 0
    bad_rate: float = 0.0
    iv: float = 0.0


@dataclass
class ScorecardFeature:
    name: str
    name_cn: str
    coef: float
    edges: list[float]
    bins: list[ScorecardBin]

    def to_dict(self) -> dict:
        def edge_json(v: float) -> float | None:
            return None if math.isinf(v) else float(v)

        return {
            "name": self.name,
            "name_cn": self.name_cn,
            "coef": float(self.coef),
            "edges": [edge_json(v) for v in self.edges],
            "bins": [
                {
                    "label": b.label,
                    "lower": edge_json(b.lower),
                    "upper": edge_json(b.upper),
                    "woe": float(b.woe),
                    "points": int(b.points),
                    "count": int(b.count),
                    "bad_rate": float(b.bad_rate),
                    "iv": float(b.iv),
                }
                for b in self.bins
            ],
        }


@dataclass
class Scorecard:
    base_score: int
    pdo: int
    base_odds: float
    factor: float
    offset: float
    base_points: int
    intercept: float
    features: list[ScorecardFeature]
    grade_cutoffs: dict[str, int]

    def __post_init__(self) -> None:
        self._point_arrays = {
            f.name: np.asarray([b.points for b in f.bins], dtype=float) for f in self.features
        }
        self._woe_arrays = {
            f.name: np.asarray([b.woe for b in f.bins], dtype=float) for f in self.features
        }
        self._label_arrays = {
            f.name: np.asarray([b.label for b in f.bins], dtype=object) for f in self.features
        }

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        total = np.full(n, float(self.base_points), dtype=float)
        out = pd.DataFrame(index=df.index)
        for feature in self.features:
            ids = assign_bin_ids(df[feature.name], feature.edges)
            points = self._point_arrays[feature.name][ids]
            out[f"bin_{feature.name}"] = self._label_arrays[feature.name][ids]
            out[f"points_{feature.name}"] = points.astype(int)
            total += points
        score = np.rint(total).astype(int)
        out["score"] = score
        out["prob_from_score"] = self.prob_from_score(score)
        out["grade"] = self.grade_of(score)
        return out

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        logit = np.full(len(df), float(self.intercept), dtype=float)
        for feature in self.features:
            ids = assign_bin_ids(df[feature.name], feature.edges)
            logit += feature.coef * self._woe_arrays[feature.name][ids]
        return 1.0 / (1.0 + np.exp(-logit))

    def prob_from_score(self, score) -> np.ndarray:
        score_arr = np.asarray(score, dtype=float)
        return 1.0 / (1.0 + np.exp((score_arr - self.offset) / self.factor))

    def grade_of(self, score) -> np.ndarray:
        score_arr = np.asarray(score, dtype=float)
        cuts = self.grade_cutoffs
        conditions = [
            score_arr >= cuts["A"],
            score_arr >= cuts["B"],
            score_arr >= cuts["C"],
            score_arr >= cuts["D"],
        ]
        return np.select(conditions, ["A", "B", "C", "D"], default="E")

    def table(self) -> pd.DataFrame:
        rows = []
        for feature in self.features:
            for b in feature.bins:
                rows.append(
                    {
                        "变量": feature.name,
                        "中文名": feature.name_cn,
                        "分箱": b.label,
                        "样本数": b.count,
                        "坏客户率": b.bad_rate,
                        "WOE": b.woe,
                        "系数": feature.coef,
                        "得分": b.points,
                    }
                )
        rows.append(
            {
                "变量": "Base",
                "中文名": "基础分",
                "分箱": "—",
                "样本数": np.nan,
                "坏客户率": np.nan,
                "WOE": np.nan,
                "系数": np.nan,
                "得分": self.base_points,
            }
        )
        frame = pd.DataFrame(rows)
        frame["样本数"] = frame["样本数"].astype("Int64")
        return frame

    def to_dict(self) -> dict:
        return {
            "base_score": int(self.base_score),
            "pdo": int(self.pdo),
            "base_odds": float(self.base_odds),
            "factor": float(self.factor),
            "offset": float(self.offset),
            "base_points": int(self.base_points),
            "intercept": float(self.intercept),
            "grade_cutoffs": {k: int(v) for k, v in self.grade_cutoffs.items()},
            "grade_labels": GRADE_LABELS,
            "features": [f.to_dict() for f in self.features],
        }

    def save(self, json_path: str | Path, csv_path: str | Path | None = None) -> None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if csv_path is not None:
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.table().to_csv(csv_path, index=False, encoding="utf-8-sig")

    @classmethod
    def load(cls, path: str | Path) -> "Scorecard":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict) -> "Scorecard":
        def edge_value(v, sign):
            return math.inf * sign if v is None else float(v)

        features = []
        for item in data["features"]:
            edges = [edge_value(v, -1 if i == 0 else 1) for i, v in enumerate(item["edges"])]
            bins = [
                ScorecardBin(
                    label=str(b["label"]),
                    lower=edge_value(b.get("lower"), -1),
                    upper=edge_value(b.get("upper"), 1),
                    woe=float(b["woe"]),
                    points=int(b["points"]),
                    count=int(b.get("count", 0)),
                    bad_rate=float(b.get("bad_rate", 0.0)),
                    iv=float(b.get("iv", 0.0)),
                )
                for b in item["bins"]
            ]
            features.append(
                ScorecardFeature(
                    name=item["name"],
                    name_cn=item.get("name_cn", item["name"]),
                    coef=float(item["coef"]),
                    edges=edges,
                    bins=bins,
                )
            )
        return cls(
            base_score=int(data["base_score"]),
            pdo=int(data["pdo"]),
            base_odds=float(data["base_odds"]),
            factor=float(data["factor"]),
            offset=float(data["offset"]),
            base_points=int(data["base_points"]),
            intercept=float(data["intercept"]),
            features=features,
            grade_cutoffs={k: int(v) for k, v in data["grade_cutoffs"].items()},
        )


def build_scorecard(
    model: LogitModel,
    binnings: dict[str, BinningResult],
    cfg: PipelineConfig,
) -> Scorecard:
    factor = cfg.factor
    offset = cfg.offset
    base_points = int(round(offset - factor * model.intercept))
    features = []
    for name in model.features:
        result = binnings[name]
        coef = float(model.result.params[name])
        bins = []
        for row in result.bins.itertuples(index=False):
            bins.append(
                ScorecardBin(
                    label=str(row.label),
                    lower=float(row.lower),
                    upper=float(row.upper),
                    woe=float(row.woe),
                    points=int(round(-factor * coef * float(row.woe))),
                    count=int(row.count),
                    bad_rate=float(row.bad_rate),
                    iv=float(row.iv),
                )
            )
        features.append(
            ScorecardFeature(
                name=name,
                name_cn=result.name_cn,
                coef=coef,
                edges=list(result.edges),
                bins=bins,
            )
        )
    return Scorecard(
        base_score=cfg.base_score,
        pdo=cfg.pdo,
        base_odds=cfg.base_odds,
        factor=factor,
        offset=offset,
        base_points=base_points,
        intercept=model.intercept,
        features=features,
        grade_cutoffs=cfg.grade_cutoffs,
    )
