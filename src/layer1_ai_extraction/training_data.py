"""Access to the annotated splits used by both trainers (INTERFACES.md section 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "ground_truth" / "train.jsonl"
VAL_PATH = PROJECT_ROOT / "data" / "ground_truth" / "val.jsonl"


def require_training_files(train_path: Path, val_path: Path) -> None:
    """Refuse to run without the annotated splits. The trainers never create or edit them."""
    missing = [str(path) for path in (train_path, val_path) if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Training data not found: " + ", ".join(missing) + ". These splits come from the annotation and "
            "split workflow (INTERFACES.md section 5); the trainers do not create or edit them."
        )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON ({exc})") from exc
    return records
