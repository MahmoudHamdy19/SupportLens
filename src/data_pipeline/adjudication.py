"""Supervisor adjudication and construction of the ground truth (thesis Section 3.3.1).

The ground truth is produced only from the two annotators' labels and the supervisor's
decisions. The builder refuses to produce anything while a record is missing from either
annotator or a disagreement has no decision for every differing field.

Adjudication file (JSON)::

    {"adjudicator": "supervisor",
     "decisions": [
       {"package_name": "urllib3",
        "fields": {"verdict": "annotator2",
                   "version_expression": {"value": ["2.x"]}},
        "rationale": "free text"}]}

A field decision is ``"annotator1"``, ``"annotator2"`` or ``{"value": ...}``. Decisions may
only concern fields in which the annotators actually differ. Where the annotators agree
(after whitespace normalisation) the first annotator's surface form is kept.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .agreement import COMPARED_FIELDS, comparable, field_value, find_disagreements
from .annotation import Annotation, validate_annotation
from .constants import COMMITMENT_KEYS, VERDICTS
from .io_utils import DataError, PathLike, read_json

PASS_THROUGH_KEYS = (
    "pushed_at",
    "repo_archived",
    "repo_fork",
    "mapping_uncertain",
    "source_diagnostics",
    "collected_at",
    "policy_path",
    "policy_url",
    "sampling_stratum",
)

GROUND_TRUTH_KEYS = (
    "package_name",
    "github_repo",
    "policy_text",
    "commitment",
    "verdict",
    "releases",
    "vulnerabilities",
    "stars",
    "repo_created_at",
    "policy_word_count",
    "annotation",
)


class AdjudicationError(DataError):
    """The adjudication is incomplete or inconsistent with the annotators' disagreements."""


def load_adjudication(path: PathLike) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise AdjudicationError(f"{path}: expected a JSON object")
    adjudicator = data.get("adjudicator")
    if not isinstance(adjudicator, str) or not adjudicator.strip():
        raise AdjudicationError(f"{path}: 'adjudicator' must name the person who decided (the supervisor)")
    decisions: Dict[str, Dict[str, Any]] = {}
    for entry in data.get("decisions") or []:
        name = entry.get("package_name") if isinstance(entry, dict) else None
        if not name:
            raise AdjudicationError(f"{path}: every decision needs a package_name")
        if name in decisions:
            raise AdjudicationError(f"{path}: duplicate decision for {name}")
        fields = entry.get("fields")
        if not isinstance(fields, dict):
            raise AdjudicationError(f"{path}: {name}: 'fields' must be an object")
        decisions[name] = {"fields": fields, "rationale": str(entry.get("rationale") or "")}
    return adjudicator.strip(), decisions


def adjudication_template(
    first: Mapping[str, Annotation], second: Mapping[str, Annotation]
) -> Dict[str, Any]:
    """Blank decision sheet for the supervisor: it lists the disagreements and decides nothing."""
    decisions = []
    for name, fields in find_disagreements(first, second).items():
        decisions.append(
            {
                "package_name": name,
                "differing_fields": fields,
                "annotator1": {f: field_value(first[name], f) for f in fields},
                "annotator2": {f: field_value(second[name], f) for f in fields},
                "fields": {},
                "rationale": "",
            }
        )
    return {"adjudicator": "", "decisions": decisions}


def _decide(field: str, spec: Any, first: Annotation, second: Annotation) -> Any:
    if spec == "annotator1":
        return field_value(first, field)
    if spec == "annotator2":
        return field_value(second, field)
    if isinstance(spec, dict) and set(spec) == {"value"}:
        return spec["value"]
    raise AdjudicationError(
        f"{first.package_name}: decision for {field!r} must be 'annotator1', 'annotator2' or {{\"value\": ...}}"
    )


def build_ground_truth(
    sample: Sequence[Mapping[str, Any]],
    first: Mapping[str, Annotation],
    second: Mapping[str, Annotation],
    adjudicator: str,
    decisions: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return ground-truth records (INTERFACES.md, section 5) or raise; never a partial result."""
    names = [r["package_name"] for r in sample]
    problems: List[str] = []
    for label, annotations in (("annotator1", first), ("annotator2", second)):
        missing = sorted(set(names) - set(annotations))
        if missing:
            problems.append(f"{label} has no annotation for {len(missing)} record(s), e.g. {missing[:5]}")
    if problems:
        raise AdjudicationError("; ".join(problems))

    disagreements = find_disagreements(first, second)
    unknown = sorted(set(decisions) - set(disagreements))
    if unknown:
        problems.append(
            f"decisions exist for {len(unknown)} record(s) without a disagreement or outside the sample, e.g. {unknown[:5]}"
        )
    for name, differing in disagreements.items():
        decided = set(decisions.get(name, {}).get("fields", {}))
        unresolved = [f for f in differing if f not in decided]
        extra = sorted(decided - set(differing))
        if unresolved:
            problems.append(f"{name}: unresolved fields {unresolved}")
        if extra:
            problems.append(f"{name}: decisions for fields without a disagreement {extra}")
    if problems:
        shown = problems[:10]
        raise AdjudicationError(
            f"Adjudication refused: {len(problems)} problem(s). " + " | ".join(shown) + (" | ..." if len(problems) > 10 else "")
        )

    records: List[Dict[str, Any]] = []
    for source in sorted(sample, key=lambda r: r["package_name"]):
        name = source["package_name"]
        a1, a2 = first[name], second[name]
        chosen = decisions.get(name, {}).get("fields", {})
        values = {
            field: _decide(field, chosen[field], a1, a2) if field in chosen else field_value(a1, field)
            for field in COMPARED_FIELDS
        }
        commitment = validate_annotation(
            name,
            {key: values[key] for key in (*COMMITMENT_KEYS, "temporal_expression")},
            values["verdict"],
            source["policy_text"],
            who="adjudicated",
        )
        record: Dict[str, Any] = {
            "package_name": name,
            "github_repo": source.get("github_repo"),
            "policy_text": source["policy_text"],
            "commitment": commitment,
            "verdict": values["verdict"],
            "releases": source.get("releases") or [],
            "vulnerabilities": source.get("vulnerabilities") or [],
            "stars": source.get("stars"),
            "repo_created_at": source.get("repo_created_at"),
            "policy_word_count": source.get("policy_word_count"),
            "annotation": {
                "annotator1": a1.to_json(),
                "annotator2": a2.to_json(),
                "adjudicated_by": adjudicator if name in disagreements else None,
                "adjudicated_fields": sorted(chosen),
                "adjudication_rationale": decisions[name]["rationale"] if name in disagreements else None,
            },
        }
        for key in PASS_THROUGH_KEYS:
            if key in source:
                record[key] = source[key]
        records.append(record)
    return records


def validate_ground_truth_records(records: Sequence[Mapping[str, Any]]) -> None:
    """Structural and provenance checks shared by the split script and the result verifier."""
    if not records:
        raise DataError("The ground truth is empty")
    seen = set()
    for record in records:
        name = record.get("package_name")
        missing = [k for k in GROUND_TRUTH_KEYS if k not in record]
        if missing:
            raise DataError(f"{name}: ground-truth record lacks {missing}")
        if name in seen:
            raise DataError(f"Duplicate ground-truth record for {name}")
        seen.add(name)
        if record["verdict"] not in VERDICTS:
            raise DataError(f"{name}: verdict {record['verdict']!r} is not one of the five categories")
        annotation = record["annotation"]
        if not all(k in annotation for k in ("annotator1", "annotator2", "adjudicated_by")):
            raise DataError(f"{name}: annotation provenance is incomplete")
        a1, a2 = annotation["annotator1"], annotation["annotator2"]
        differ = a1["verdict"] != a2["verdict"] or any(
            comparable(k, a1["commitment"][k]) != comparable(k, a2["commitment"][k]) for k in COMMITMENT_KEYS
        )
        if differ and not annotation["adjudicated_by"]:
            raise DataError(f"{name}: the annotators disagree but no adjudicator is recorded")
