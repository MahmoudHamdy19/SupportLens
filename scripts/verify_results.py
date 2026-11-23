"""Recompute every reported number and fail when a result file or README table differs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.cli import verify_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(verify_main())
