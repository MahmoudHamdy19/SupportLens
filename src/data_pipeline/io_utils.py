"""File helpers: strict JSON(L) reading, atomic writing and hashing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PathLike = Union[str, Path]


class DataError(RuntimeError):
    """An input file is missing, malformed or inconsistent with the protocol."""


def configure_console() -> None:
    """Make console output safe for non-ASCII paths (the project path contains Arabic letters)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="backslashreplace")
            except (ValueError, OSError):
                pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    """Serialise numpy scalars and arrays without importing numpy at module load."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def dumps_json(obj: Any, *, indent: Union[int, None] = 2) -> str:
    """Serialise to JSON; NaN and infinity are rejected so a results file is always valid JSON."""
    return json.dumps(obj, ensure_ascii=False, indent=indent, allow_nan=False, default=_json_default)


def read_json(path: PathLike) -> Any:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"File not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}: invalid JSON ({exc})") from exc


def read_jsonl(path: PathLike) -> List[dict]:
    """Read a JSON-lines file; a malformed line is an error, never skipped silently."""
    path = Path(path)
    if not path.is_file():
        raise DataError(f"File not found: {path}")
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{number}: invalid JSON ({exc})") from exc
            if not isinstance(obj, dict):
                raise DataError(f"{path}:{number}: expected a JSON object per line")
            records.append(obj)
    return records


def append_jsonl(path: PathLike, record: Mapping[str, Any]) -> None:
    """Append one record and force it to disk, so an interrupted run never loses finished work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n"
    with open(path, "ab") as handle:
        handle.write(line.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_many(files: Mapping[PathLike, Union[str, bytes]]) -> None:
    """Write several files so that either all of them appear or none does.

    Every file is first written to a temporary sibling; only when all temporaries
    exist are they renamed over their targets.
    """
    pending: dict = {}
    try:
        for target, content in files.items():
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            data = content.encode("utf-8") if isinstance(content, str) else content
            temporary.write_bytes(data)
            pending[target] = temporary
        for target, temporary in list(pending.items()):
            os.replace(temporary, target)
            del pending[target]
    finally:
        for temporary in pending.values():
            try:
                temporary.unlink()
            except OSError:
                pass


def write_json(path: PathLike, obj: Any) -> None:
    atomic_write_many({path: dumps_json(obj) + "\n"})


def jsonl_text(records: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(r, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n" for r in records
    )


def write_jsonl(path: PathLike, records: Iterable[Mapping[str, Any]]) -> None:
    atomic_write_many({path: jsonl_text(records)})


def normalise_package_name(name: str) -> str:
    """PEP 503 normalisation, used to deduplicate candidate package names."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def word_count(text: str) -> int:
    return len(text.split())
