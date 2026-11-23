"""Fill the RESULT markers in README.md and docs from the result JSON files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.cli import render_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(render_main())
