"""Expert rubric: --prepare writes blind rating sheets, --ingest reads the completed ones."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.cli import expert_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(expert_main())
