"""对新申请人 CSV 批量打分。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lendingclub_scorecard.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["score", *sys.argv[1:]]))
