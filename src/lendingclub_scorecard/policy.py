"""打分落地的决策策略：MOB12 PD → 校准 → 终身 PD → 等级/审批。

训练与评估产物里的 `prob_lifetime` 是业务决策口径；仅输出未校准、MOB12 尺度的
`predict_pd` 会让 0.27 阈值和按 PD 划分的等级都无法直接使用。本模块把这三步
固化成可保存、可加载、可测试的 policy 文件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GRADE_PD_CUTOFFS

EPS = 1e-6
ALL_SEGMENT = "__ALL__"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


@dataclass(frozen=True)
class DecisionPolicy:
    """生产打分使用的校准、终身换算与审批策略。"""

    platt_a: float
    platt_b: float
    emergence_by_term: dict[str, float]
    emergence_default: float
    threshold_lifetime: float
    grade_pd_cutoffs: tuple[float, ...] = GRADE_PD_CUTOFFS
    target_bad_rate: float = 0.15
    version: str = "v3.0-p0"

    def __post_init__(self) -> None:
        cuts = tuple(float(c) for c in self.grade_pd_cutoffs)
        if len(cuts) != 4 or any(a >= b for a, b in zip(cuts, cuts[1:])):
            raise ValueError(f"grade_pd_cutoffs 必须是 4 个递增切点: {cuts}")
        object.__setattr__(self, "grade_pd_cutoffs", cuts)
        if not 0.0 < float(self.threshold_lifetime) < 1.0:
            raise ValueError("threshold_lifetime 必须在 (0, 1) 内")
        if not 0.0 < float(self.emergence_default) <= 1.0:
            raise ValueError("emergence_default 必须在 (0, 1] 内")
        bad_e = [k for k, v in self.emergence_by_term.items() if not 0.0 < float(v) <= 1.0]
        if bad_e:
            raise ValueError(f"暴露比例 e 必须在 (0, 1] 内: {bad_e}")

    def calibrate(self, p_raw) -> np.ndarray:
        """Platt scaling：sigmoid(a × logit(p) + b)。"""
        return _sigmoid(self.platt_a * _logit(p_raw) + self.platt_b)

    def emergence(self, term_num) -> np.ndarray:
        """按期限查暴露比例 e；未知期限回退到整体 e。"""
        if term_num is None:
            raise ValueError("term_num 不能为空；终身 PD 必须按期限换算")
        keys = (
            pd.to_numeric(pd.Series(term_num), errors="coerce")
            .round()
            .astype("Int64")
            .astype(str)
        )
        return (
            keys.map(self.emergence_by_term)
            .astype(float)
            .fillna(float(self.emergence_default))
            .to_numpy(dtype=float)
        )

    def to_lifetime(self, p_calibrated, term_num) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = np.asarray(p_calibrated, dtype=float)
        e = self.emergence(term_num)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = p / e
        clipped = np.isfinite(ratio) & (ratio > 1.0)
        p_lt = np.clip(np.nan_to_num(ratio, nan=0.0), 0.0, 1.0)
        return p_lt, e, clipped

    def grade_of(self, p_lifetime) -> np.ndarray:
        """按终身 PD 划 A–E；右开区间，和报告里的 8/12/18/25% 边界一致。"""
        bins = [-np.inf, *self.grade_pd_cutoffs, np.inf]
        return (
            pd.cut(
                pd.Series(np.asarray(p_lifetime, dtype=float)),
                bins=bins,
                labels=list("ABCDE"),
                right=False,
            )
            .astype(str)
            .to_numpy()
        )

    def approve(self, p_lifetime) -> np.ndarray:
        return np.asarray(p_lifetime, dtype=float) < float(self.threshold_lifetime)

    def decide(self, p_raw, term_num) -> pd.DataFrame:
        p_cal = self.calibrate(p_raw)
        p_lt, e, clipped = self.to_lifetime(p_cal, term_num)
        return pd.DataFrame(
            {
                "prob_mob12_raw": np.asarray(p_raw, dtype=float),
                "prob_mob12_calibrated": p_cal,
                "prob_lifetime": p_lt,
                "grade": self.grade_of(p_lt),
                "approve": self.approve(p_lt),
                "term_num": pd.to_numeric(pd.Series(term_num), errors="coerce").to_numpy(),
                "emergence_frac": e,
                "lifetime_clipped": clipped,
            }
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "calibration": {
                "method": "platt",
                "a": float(self.platt_a),
                "b": float(self.platt_b),
            },
            "emergence_by_term": {
                str(k): float(v) for k, v in sorted(self.emergence_by_term.items())
            },
            "emergence_default": float(self.emergence_default),
            "threshold_lifetime": float(self.threshold_lifetime),
            "target_bad_rate": float(self.target_bad_rate),
            "grade_pd_cutoffs": [float(c) for c in self.grade_pd_cutoffs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionPolicy":
        cal = data["calibration"]
        return cls(
            platt_a=float(cal["a"]),
            platt_b=float(cal["b"]),
            emergence_by_term={
                str(k): float(v) for k, v in data["emergence_by_term"].items()
            },
            emergence_default=float(data["emergence_default"]),
            threshold_lifetime=float(data["threshold_lifetime"]),
            grade_pd_cutoffs=tuple(float(c) for c in data["grade_pd_cutoffs"]),
            target_bad_rate=float(data.get("target_bad_rate", 0.15)),
            version=str(data.get("version", "v3.0-p0")),
        )

    @classmethod
    def from_metadata(cls, model_dir: str | Path) -> "DecisionPolicy":
        meta = json.loads(
            (Path(model_dir) / "run_metadata.json").read_text(encoding="utf-8")
        )
        cal = meta["calibration"]
        emergence = {
            str(k): float(v) for k, v in meta["emergence_by_term"].items()
        }
        return cls(
            platt_a=float(cal["platt_a"]),
            platt_b=float(cal["platt_b"]),
            emergence_by_term=emergence,
            emergence_default=float(emergence.get(ALL_SEGMENT, np.nan)),
            threshold_lifetime=float(meta["threshold"]["selected"]),
            grade_pd_cutoffs=tuple(
                float(c) for c in meta.get("grade_pd_cutoffs", GRADE_PD_CUTOFFS)
            ),
            target_bad_rate=float(meta["threshold"].get("target_bad_rate", 0.15)),
        )

    @classmethod
    def from_artifacts(cls, model_dir: str | Path) -> "DecisionPolicy":
        path = Path(model_dir) / "decision_policy.json"
        if path.exists():
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        return cls.from_metadata(model_dir)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
