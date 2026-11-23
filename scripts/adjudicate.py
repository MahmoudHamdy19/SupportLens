"""Build data/ground_truth/ground_truth.jsonl from the annotators and the supervisor's decisions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.cli import adjudicate_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(adjudicate_main())
