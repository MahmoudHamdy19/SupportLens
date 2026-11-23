"""Stratified 70/15/15 split of the ground truth (seed 42)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.cli import splits_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(splits_main())
