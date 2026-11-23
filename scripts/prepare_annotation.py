"""Write blind Label Studio task files for the two annotators."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline.cli import prepare_annotation_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(prepare_annotation_main())
