"""Draw the stratified random sample of 300 policies (thesis 3.3.1)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.cli import sample_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(sample_main())
