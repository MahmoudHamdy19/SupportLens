"""
Layer 4: Explanation and Output Layer
Component: Result storage (JSON)
================================
Stores each final record as a JSON file under data/reports/ and appends one line per
report to reports_index.jsonl. The index is append-only and written only by this module:
identifiers and timestamps are generated at save time, never typed by hand.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
INDEX_NAME = "reports_index.jsonl"

_REPORT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class StoredReport(NamedTuple):
    """Identity of a saved report."""

    report_id: str
    path: Path
    created_at: str


def _json_default(value: Any) -> Any:
    """Serialise the few non-JSON types that appear in layer traces."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


class ResultStorage:
    """Persists final records as JSON and maintains the append-only report index."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.output_dir / INDEX_NAME

    def save_report(self, report: Any) -> StoredReport:
        """Write `report` to a new uniquely named JSON file and append its index entry."""
        record = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        created = datetime.now(timezone.utc)
        package = str(record.get("package_name") or "report")
        safe_package = re.sub(r"[^A-Za-z0-9_.-]", "_", package).lstrip("._-") or "report"
        report_id = f"{safe_package}_{created:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:12]}"
        created_at = created.isoformat()

        stored: Dict[str, Any] = {"report_id": report_id, "created_at": created_at}
        stored.update({key: value for key, value in record.items() if key not in stored})
        text = json.dumps(stored, indent=2, ensure_ascii=False, default=_json_default)

        path = self.output_dir / f"{report_id}.json"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        os.replace(temporary, path)

        assessment = record.get("assessment") or {}
        entry = {
            "report_id": report_id,
            "created_at": created_at,
            "package_name": record.get("package_name"),
            "verdict": record.get("verdict"),
            "rules_triggered": record.get("rules_triggered") or assessment.get("rules_triggered") or [],
            "report_file": path.name,
        }
        with open(self.index_file, "a", encoding="utf-8", newline="\n") as index:
            index.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")
        return StoredReport(report_id, path, created_at)

    def list_reports(self, package_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the index entries in the order they were written, optionally for one package."""
        if not self.index_file.exists():
            return []
        entries: List[Dict[str, Any]] = []
        with open(self.index_file, "r", encoding="utf-8") as index:
            for number, line in enumerate(index, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{self.index_file.name} line {number} is not valid JSON") from error
                if package_name is None or entry.get("package_name") == package_name:
                    entries.append(entry)
        return entries

    def load_report(self, report_id: str) -> Dict[str, Any]:
        """Load a stored report by its identifier."""
        if not _REPORT_ID_PATTERN.match(report_id) or ".." in report_id:
            raise ValueError(f"Invalid report id: {report_id!r}")
        path = self.output_dir / f"{report_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"No stored report with id {report_id}")
        return json.loads(path.read_text(encoding="utf-8"))
