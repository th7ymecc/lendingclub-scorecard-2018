"""命令行入口。"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURE_NAMES, ID_COLUMN, PipelineConfig
from .data import Preprocessor
from .pipeline import configure_logging, run_pipeline
from .scorecard import Scorecard

LOGGER = logging.getLogger(__name__)


def _prepare_scoring_frame(df: pd.DataFrame) -> pd.DataFrame:
    """为新数据补齐衍生变量；要求提供原始申请字段。"""
    out = df.copy()
    if "issue_d" in out.columns:
        out["issue_dt"] = pd.to_datetime(out["issue_d"], format="%b-%Y", errors="coerce")
    if "earliest_cr_line" in out.columns and "issue_dt" in out.columns:
        earliest = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y", errors="coerce")
        out["credit_hist_months"] = (out["issue_dt"] - earliest).dt.days / 30.44
    if "emp_length" in out.columns:
        def parse(value):
            if pd.isna(value):
                return -1
            text = str(value).strip().lower()
            if "<" in text:
                return 0
            if text.startswith("10+"):
                return 10
            match = re.search(r"(\d+)", text)
            return int(match.group(1)) if match else -1

        out["emp_length"] = out["emp_length"].map(parse)
    income = out.get("annual_inc", pd.Series(np.nan, index=out.index)).replace(0, np.nan)
    if "loan_amnt" in out.columns:
        out["loan_income_ratio"] = out["loan_amnt"] / income
    if "revol_bal" in out.columns:
        out["revol_bal_income_ratio"] = out["revol_bal"] / income
    if "total_bal_ex_mort" in out.columns:
        out["total_bal_income_ratio"] = out["total_bal_ex_mort"] / income
    if "open_acc" in out.columns and "total_acc" in out.columns:
        out["open_acc_ratio"] = out["open_acc"] / out["total_acc"].replace(0, np.nan)
    missing = [c for c in FEATURE_NAMES if c not in out.columns]
    if missing:
        raise ValueError(f"输入缺少字段: {missing}")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LendingClub 信用风险评分卡")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="训练、OOT 评估并生成报告")
    run.add_argument("--data-dir", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--no-report", action="store_true")
    run.add_argument("--refresh-data", action="store_true", help="强制重建数据缓存")

    score = sub.add_parser("score", help="对新 CSV 批量打分")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--model-dir", type=Path, default=Path("outputs"))
    return parser


def _cmd_run(args) -> int:
    cfg = PipelineConfig()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    cfg.make_report = not args.no_report
    run_pipeline(cfg, refresh_data=args.refresh_data)
    return 0


def _cmd_score(args) -> int:
    model_dir = Path(args.model_dir)
    pre = Preprocessor.load(model_dir / "models" / "preprocessor.joblib")
    card = Scorecard.load(model_dir / "scorecard.json")
    df = pd.read_csv(args.input)
    scored_input = _prepare_scoring_frame(df)
    X = pre.transform(scored_input)
    scored = card.score_frame(X)
    scored["Probability"] = card.predict_pd(X)
    output = pd.DataFrame(
        {
            "Id": df[ID_COLUMN].to_numpy() if ID_COLUMN in df.columns else np.arange(1, len(df) + 1),
            "Probability": scored["Probability"].to_numpy(),
            "score": scored["score"].to_numpy(),
            "grade": scored["grade"].to_numpy(),
        }
    )
    output.to_csv(args.output, index=False, float_format="%.9f")
    LOGGER.info("打分完成: %s 行 -> %s", f"{len(df):,}", args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "score":
        return _cmd_score(args)
    return 1
