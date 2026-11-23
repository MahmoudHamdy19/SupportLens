"""Compute Cohen's kappa and the disagreement list from the two annotator exports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.cli import check_agreement_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(check_agreement_main())
