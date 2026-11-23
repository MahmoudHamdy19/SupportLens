"""Collect the PyPI dataset into data/processed/pypi_dataset.jsonl (resumable). Run with --help."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.cli import collect_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(collect_main())
