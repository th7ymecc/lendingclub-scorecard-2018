"""用 v2 的 OOT 打分重算 v3 的阈值/校准表，不重新训练模型。

用途：修复 threshold_sensitivity 的加权标签 int 截断后，无需重跑 26 分钟 pipeline，
即可把 v2 模型产物更新为 v3 报告所需的表格与元数据。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lendingclub_scorecard.calibration import threshold_sensitivity, threshold_stability  # noqa: E402
from lendingclub_scorecard.evaluation import threshold_confusion  # noqa: E402


def main() -> int:
    out = ROOT / "outputs"
    scores = pd.read_csv(
        out / "predictions" / "oot_scores.csv",
        usecols=["target", "emergence_frac", "prob_lifetime"],
    )
    y_lt = scores["target"].to_numpy(dtype=float) / scores["emergence_frac"].to_numpy(dtype=float)
    p_lt = scores["prob_lifetime"].to_numpy(dtype=float)

    grid = np.arange(0.02, 0.901, 0.01)
    sens = threshold_sensitivity(
        y_lt,
        p_lt,
        target_bad_rates=(0.12, 0.15, 0.18),
        min_approval=0.50,
        grid=grid,
    )
    stab = threshold_stability(
        y_lt,
        p_lt,
        base_target=0.15,
        perturbation=0.02,
        min_approval=0.50,
    )
    sens.to_csv(out / "tables" / "threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    stab.to_csv(out / "tables" / "threshold_stability.csv", index=False, encoding="utf-8-sig")

    meta_path = out / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    observed = float(np.mean(y_lt))
    selected = float(meta["threshold"]["selected"])
    conf = threshold_confusion(y_lt, p_lt, selected)
    rejected_rate = (
        float(conf["rejected_bad"] / conf["rejected_n"]) if conf["rejected_n"] else float("nan")
    )

    meta["calibration"]["oot_observed_lifetime_bad_rate"] = observed
    meta["threshold_sensitivity"] = sens.to_dict(orient="records")
    meta["threshold_stability"] = stab.to_dict(orient="records")
    meta["threshold"]["rejected_bad_rate_oot"] = rejected_rate
    meta["threshold"]["lifetime_bad_rate_oot"] = observed
    meta["grade_pd_cutoffs"] = [0.08, 0.12, 0.18, 0.25]
    meta["decision_policy_file"] = "decision_policy.json"
    meta["revision"] = {
        "version": "v3.0-p0",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "基于 v2 模型/OOT 产物重算；未重新训练。修复加权标签 int 截断并统一终身口径。",
    }
    meta["scorecard"]["grade_pd_cutoffs"] = [0.08, 0.12, 0.18, 0.25]
    for row in meta.get("model_attribution", []):
        row["model"] = (
            row["model"]
            .replace("12 变量", "14 变量")
            .replace("同 12 变量", "同 14 变量")
        )
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    # 校准/口径对照表：只改终身行的实际值，MOB12 行保持原样。
    cal_path = out / "tables" / "calibration_by_scale.csv"
    cal = pd.read_csv(cal_path)
    mask = cal["pd_scale"] == "lifetime"
    cal.loc[mask, "actual"] = observed
    cal.loc[mask, "scale_gap"] = observed - float(cal.loc[mask, "mean_pred"].iloc[0])
    cal.loc[mask, "ratio"] = observed / float(cal.loc[mask, "mean_pred"].iloc[0])
    cal.loc[mask, "label_scale"] = "lifetime（E[y/e] 加权估计）"
    cal.to_csv(cal_path, index=False, encoding="utf-8-sig")

    pd_path = out / "tables" / "pd_scale_comparison.csv"
    pd_scale = pd.read_csv(pd_path)
    mask = pd_scale["口径"].str.startswith("终身")
    pd_scale.loc[mask, "OOT实测坏账率"] = observed
    pd_scale.loc[mask, "说明"] = "按期限分段换算，E[y/e] 加权估计，可直接与业务坏账率目标对比"
    pd_scale.to_csv(pd_path, index=False, encoding="utf-8-sig")

    for name in ("model_attribution_oot.csv", "model_comparison_oot.csv"):
        path = out / "tables" / name
        frame = pd.read_csv(path)
        for col in ("model",):
            if col in frame.columns:
                frame[col] = (
                    frame[col].astype(str)
                    .str.replace("12 变量", "14 变量", regex=False)
                    .str.replace("同 12 变量", "同 14 变量", regex=False)
                    .str.replace("LightGBM-12", "LightGBM-同变量", regex=False)
                )
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"weighted lifetime bad rate: {observed:.6f}")
    print(f"selected threshold {selected:.4f}: approval {conf['approval_rate']:.4%}, "
          f"approved bad {conf['approved_bad_rate']:.4%}, rejected bad {rejected_rate:.4%}, "
          f"capture {conf['bad_capture']:.4%}")
    print(sens.to_string(index=False))
    print(stab.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
