"""Evaluate the four systems on data/ground_truth/test.jsonl (thesis 3.3.2 to 3.3.4). Strict by default."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.cli import run_evaluation_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_evaluation_main())
