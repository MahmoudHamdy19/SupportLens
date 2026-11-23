"""Stratified 70/15/15 split on the final verdict (thesis Section 3.3.1, seed 42).

Each verdict class is split separately so that the class proportions of the training,
validation and test sets match the whole ground truth as closely as the class sizes allow.

Explicit rules for small classes (n is the number of records in a class):

* n >= 3: validation and test receive ``max(1, round_half_up(0.15 n))`` records each, the
  rest go to training, so every class is present in every split;
* n == 2: one record to training and one to test (no validation example), because the test
  set is the one that must contain every class that exists;
* n == 1: the record goes to training; the class cannot be evaluated and is flagged;
* n == 0: the class is absent and is reported as such.

Every deviation from the nominal ratios is written to ``split_report.json``.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .constants import SEED, SPLIT_RATIOS, VERDICTS
from .io_utils import DataError, sha256_file, utc_now_iso


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def class_allocation(n: int, ratios: Mapping[str, float] = SPLIT_RATIOS) -> Tuple[int, int, int, str]:
    """Return ``(train, val, test, rule)`` for a class with ``n`` records."""
    if n == 0:
        return 0, 0, 0, "absent"
    if n == 1:
        return 1, 0, 0, "n=1: training only"
    if n == 2:
        return 1, 0, 1, "n=2: one training and one test record, no validation record"
    val = max(1, round_half_up(ratios["val"] * n))
    test = max(1, round_half_up(ratios["test"] * n))
    train = n - val - test
    forced = round_half_up(ratios["val"] * n) < 1 or round_half_up(ratios["test"] * n) < 1
    rule = "minimum of one record each in validation and test" if forced else "nominal"
    return train, val, test, rule


def stratified_split(
    records: Sequence[Mapping[str, Any]], *, seed: int = SEED
) -> Tuple[Dict[str, List[Mapping[str, Any]]], Dict[str, Any]]:
    unknown = sorted({r["verdict"] for r in records} - set(VERDICTS))
    if unknown:
        raise DataError(f"Unknown verdict labels in the ground truth: {unknown}")
    by_class: Dict[str, List[Mapping[str, Any]]] = {label: [] for label in VERDICTS}
    for record in sorted(records, key=lambda r: r["package_name"]):
        by_class[record["verdict"]].append(record)

    rng = random.Random(seed)
    splits: Dict[str, List[Mapping[str, Any]]] = {"train": [], "val": [], "test": []}
    per_class: Dict[str, Any] = {}
    for label in VERDICTS:
        members = list(by_class[label])
        rng.shuffle(members)
        train, val, test, rule = class_allocation(len(members))
        splits["test"].extend(members[:test])
        splits["val"].extend(members[test:test + val])
        splits["train"].extend(members[test + val:])
        per_class[label] = {"total": len(members), "train": train, "val": val, "test": test, "rule": rule}
    for key in splits:
        splits[key].sort(key=lambda r: r["package_name"])

    total = len(records)
    warnings: List[str] = []
    for label, info in per_class.items():
        if info["total"] and info["test"] == 0:
            warnings.append(f"class {label!r} has {info['total']} record(s) and none in the test set: it cannot be evaluated")
        elif info["total"] and info["val"] == 0:
            warnings.append(f"class {label!r} has no validation record")
        elif 0 < info["test"] < 5:
            warnings.append(f"class {label!r} has only {info['test']} test record(s): its F1 is unstable")
        if info["total"] == 0:
            warnings.append(f"class {label!r} has no record in the ground truth")

    report: Dict[str, Any] = {
        "seed": seed,
        "target_ratios": dict(SPLIT_RATIOS),
        "total_records": total,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "realised_ratios": {k: (len(v) / total if total else None) for k, v in splits.items()},
        "per_class": per_class,
        "class_counts": {
            split: {label: sum(1 for r in splits[split] if r["verdict"] == label) for label in VERDICTS}
            for split in splits
        },
        "warnings": warnings,
        "small_class_rules": "see the module docstring of src/data_pipeline/splits.py",
    }
    return splits, report


def add_input_provenance(report: Dict[str, Any], ground_truth_path) -> Dict[str, Any]:
    report = dict(report)
    report["input_file"] = str(ground_truth_path)
    report["input_sha256"] = sha256_file(ground_truth_path)
    report["created_at"] = utc_now_iso()
    return report
