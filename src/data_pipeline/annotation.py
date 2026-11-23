"""Annotation workflow: Label Studio tasks and annotator exports (thesis Section 3.3.1).

Two independent annotators label, for every sampled policy, the commitment tuple
c = <t, tau, v, T> and the expected verdict y (Table 3.1). This module

* builds blind Label Studio tasks (no model output, no pre-labels, shuffled per annotator;
  Label Studio assigns its own task ids on import, so the file order is the task order),
* parses an annotator's export into validated :class:`Annotation` objects.

Accepted export formats
-----------------------
1. A Label Studio JSON export (list of tasks with ``annotations``). The labelling config is
   ``data/annotations/label_studio_config.xml``: the controls are ``spans`` (labels
   ``CommitmentText`` and ``VersionExpression`` on the policy text), ``support_status``,
   ``temporal_reference``, ``expected_verdict`` and the optional ``notes``.
2. The simple format (also used by the tests)::

       {"annotator": "annotator1",
        "annotations": [
          {"package_name": "urllib3",
           "commitment": {"commitment_text": "...", "version_expression": ["2.x"],
                          "support_status": "explicit_range",
                          "temporal_reference": "vulnerability_publication_time"},
           "verdict": "Consistent", "notes": ""}]}

The optional ``temporal_expression`` (span of the phrase that states T, or null) is a
verbatim substring too and may be given in the commitment of the simple format.

Validation: t and every element of v must be verbatim substrings of the policy text; tau, T
and the verdict must belong to the closed vocabularies; t may be empty only when tau is
``ambiguous``.
"""

from __future__ import annotations

import random
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .constants import COMMITMENT_KEYS, SEED, TAU_VALUES, TEMPORAL_VALUES, VERDICTS
from .io_utils import DataError, PathLike, read_json

ALLOWED_TASK_KEYS = (
    "task_key",
    "package_name",
    "github_repo",
    "policy_text",
    "evidence_summary",
    "stars",
    "release_count",
    "vulnerability_count",
)

SPAN_FROM_NAME = "spans"
LABEL_T = "CommitmentText"
LABEL_V = "VersionExpression"
LABEL_T_REF = "TemporalExpression"


class AnnotationError(DataError):
    """An annotator export violates the codebook or the file format."""


@dataclass(frozen=True)
class Annotation:
    package_name: str
    commitment: Dict[str, Any]
    verdict: str
    notes: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"commitment": self.commitment, "verdict": self.verdict}


# --------------------------------------------------------------------------- tasks


def _date(value: Any) -> str:
    return str(value)[:10] if value else "unknown"


def format_evidence_summary(record: Mapping[str, Any]) -> str:
    """Plain-text evidence panel shown next to the policy (releases, vulnerabilities, sources)."""
    releases = list(record.get("releases") or [])
    dated = [r for r in releases if r.get("published_at")]
    undated = [r for r in releases if not r.get("published_at")]
    dated.sort(key=lambda r: str(r["published_at"]), reverse=True)
    ordered = dated + undated

    lines: List[str] = []
    lines.append(f"Package: {record.get('package_name')}   Repository: {record.get('github_repo')}")
    lines.append(f"Stars: {record.get('stars')}   Evidence collected: {_date(record.get('collected_at'))}")
    lines.append("")
    lines.append(f"RELEASES ({len(ordered)} in total, newest first, version@date)")
    entries = [f"{r.get('version')}@{_date(r.get('published_at'))}" for r in ordered]
    for start in range(0, len(entries), 6):
        lines.append("  " + ", ".join(entries[start:start + 6]))
    if not entries:
        lines.append("  (none)")
    lines.append("")

    vulnerabilities = list(record.get("vulnerabilities") or [])
    unpatched = sum(1 for v in vulnerabilities if not v.get("fixed_versions"))
    lines.append(f"VULNERABILITIES ({len(vulnerabilities)} in total, {unpatched} without a fixed version)")
    if not vulnerabilities:
        lines.append("  (no vulnerability record found in the evidence sources)")
    for number, vuln in enumerate(vulnerabilities, start=1):
        ids = ", ".join(dict.fromkeys([str(vuln.get("id"))] + list(vuln.get("cve_ids") or [])))
        affected = list(vuln.get("affected_versions") or [])
        shown = ", ".join(map(str, affected[:12])) + (f" (+{len(affected) - 12} more)" if len(affected) > 12 else "")
        fixed = ", ".join(map(str, vuln.get("fixed_versions") or [])) or "none recorded"
        summary = str(vuln.get("summary") or "").strip().replace("\n", " ")
        lines.append(f"  [{number}] {ids}  published {_date(vuln.get('published_at'))}")
        lines.append(f"      affected: {shown or 'not recorded'}")
        lines.append(f"      fixed in: {fixed}")
        if summary:
            lines.append(f"      summary: {summary[:160]}")
        if vuln.get("advisory_url"):
            lines.append(f"      advisory: {vuln['advisory_url']}")
    lines.append("")
    lines.append("EVIDENCE SOURCES")
    for source, info in sorted((record.get("source_diagnostics") or {}).items()):
        if isinstance(info, Mapping):
            lines.append(f"  {source}: {info.get('status')} (count {info.get('count')})")
    return "\n".join(lines)


def build_task(record: Mapping[str, Any]) -> Dict[str, Any]:
    data = {
        "task_key": record["package_name"],
        "package_name": record["package_name"],
        "github_repo": record.get("github_repo"),
        "policy_text": record["policy_text"],
        "evidence_summary": format_evidence_summary(record),
        "stars": record.get("stars"),
        "release_count": len(record.get("releases") or []),
        "vulnerability_count": len(record.get("vulnerabilities") or []),
    }
    return {"data": data}


def build_tasks(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [build_task(record) for record in records]


def assert_blind(tasks: Sequence[Mapping[str, Any]]) -> None:
    """Fail if a task carries anything beyond the allow-listed, label-free fields."""
    for task in tasks:
        extra = set(task.get("data", {})) - set(ALLOWED_TASK_KEYS)
        if extra or "annotations" in task or "predictions" in task:
            raise AnnotationError(
                f"Task {task.get('data', {}).get('package_name')} is not blind: "
                f"unexpected content {sorted(extra) or 'annotations/predictions'}"
            )


def shuffled_for_annotator(tasks: Sequence[Dict[str, Any]], annotator_index: int, seed: int = SEED) -> List[Dict[str, Any]]:
    """Independent, reproducible task order for one annotator."""
    order = list(tasks)
    random.Random(seed + annotator_index).shuffle(order)
    return order


# --------------------------------------------------------------------------- parsing


def _clean(text: str) -> str:
    return text.strip()


def validate_annotation(package: str, commitment: Mapping[str, Any], verdict: str, policy: str, *, who: str) -> Dict[str, Any]:
    """Check one annotation against the codebook and return the normalised commitment."""
    prefix = f"{who}: {package}"
    missing = [k for k in COMMITMENT_KEYS if k not in commitment]
    if missing:
        raise AnnotationError(f"{prefix}: commitment lacks {missing}")
    t = commitment["commitment_text"]
    v = commitment["version_expression"]
    tau = commitment["support_status"]
    temporal = commitment["temporal_reference"]
    if not isinstance(t, str):
        raise AnnotationError(f"{prefix}: commitment_text must be a string")
    if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
        raise AnnotationError(f"{prefix}: version_expression must be a list of strings")
    if tau not in TAU_VALUES:
        raise AnnotationError(f"{prefix}: support_status {tau!r} not in {TAU_VALUES}")
    if temporal not in TEMPORAL_VALUES:
        raise AnnotationError(f"{prefix}: temporal_reference {temporal!r} not in {TEMPORAL_VALUES}")
    if verdict not in VERDICTS:
        raise AnnotationError(f"{prefix}: verdict {verdict!r} not in {VERDICTS}")

    t = _clean(t)
    if t:
        if t not in policy:
            raise AnnotationError(f"{prefix}: commitment_text is not a verbatim substring of the policy")
    elif tau != "ambiguous":
        raise AnnotationError(f"{prefix}: commitment_text may be empty only when support_status is 'ambiguous'")

    temporal_text = commitment.get("temporal_expression")
    if temporal_text is not None and not isinstance(temporal_text, str):
        raise AnnotationError(f"{prefix}: temporal_expression must be a string or null")
    temporal_text = _clean(temporal_text) if temporal_text else None
    if temporal_text and temporal_text not in policy:
        raise AnnotationError(f"{prefix}: temporal_expression is not a verbatim substring of the policy")

    tokens: List[str] = []
    for item in v:
        item = _clean(item)
        if not item:
            continue
        if item not in policy:
            raise AnnotationError(f"{prefix}: version token {item!r} is not a verbatim substring of the policy")
        if item not in tokens:
            tokens.append(item)
    return {
        "commitment_text": t,
        "version_expression": tokens,
        "support_status": tau,
        "temporal_reference": temporal,
        "temporal_expression": temporal_text,
    }


def consistency_warnings(annotation: Annotation) -> List[str]:
    """Codebook consistency hints (not errors): tau 'ambiguous' implies the verdict Unmeasurable (rule R5)."""
    warnings = []
    if annotation.commitment["support_status"] == "ambiguous" and annotation.verdict != "Unmeasurable Commitment":
        warnings.append(f"{annotation.package_name}: tau is 'ambiguous' but the verdict is {annotation.verdict!r}")
    return warnings


def _parse_simple(data: Any, policies: Mapping[str, str], who: str) -> Dict[str, Annotation]:
    entries = data.get("annotations") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise AnnotationError(f"{who}: expected a list of annotations")
    result: Dict[str, Annotation] = {}
    for entry in entries:
        name = entry.get("package_name") if isinstance(entry, dict) else None
        if not name:
            raise AnnotationError(f"{who}: every annotation needs a package_name")
        if name in result:
            raise AnnotationError(f"{who}: duplicate annotation for {name}")
        if name not in policies:
            raise AnnotationError(f"{who}: {name} is not part of the sample")
        commitment = validate_annotation(
            name, entry.get("commitment") or {}, entry.get("verdict"), policies[name], who=who
        )
        result[name] = Annotation(name, commitment, entry["verdict"], str(entry.get("notes") or ""))
    return result


def _parse_label_studio(data: List[dict], policies: Mapping[str, str], who: str) -> Dict[str, Annotation]:
    result: Dict[str, Annotation] = {}
    for item in data:
        payload = item.get("data") or {}
        name = payload.get("package_name") or payload.get("task_key")
        if not name:
            raise AnnotationError(f"{who}: a Label Studio task has no package_name")
        if name not in policies:
            raise AnnotationError(f"{who}: {name} is not part of the sample")
        if name in result:
            raise AnnotationError(f"{who}: duplicate task for {name}")
        live = [a for a in item.get("annotations") or [] if not a.get("was_cancelled")]
        if not live:
            continue  # skipped or unannotated: reported later as a missing record
        if len(live) > 1:
            raise AnnotationError(f"{who}: {name} has {len(live)} annotations; an export must hold one per task")

        t_regions: List[str] = []
        v_regions: List[Tuple[int, str]] = []
        temporal_regions: List[str] = []
        choices: Dict[str, List[str]] = {}
        notes = ""
        for number, res in enumerate(live[0].get("result") or []):
            source = res.get("from_name")
            value = res.get("value") or {}
            if source == SPAN_FROM_NAME:
                text = value.get("text")
                if not isinstance(text, str):
                    raise AnnotationError(f"{who}: {name}: a labelled span carries no text")
                labels = value.get("labels") or []
                if LABEL_T in labels:
                    t_regions.append(text)
                if LABEL_V in labels:
                    v_regions.append((value.get("start", number), text))
                if LABEL_T_REF in labels:
                    temporal_regions.append(text)
            elif source in ("support_status", "temporal_reference", "expected_verdict"):
                choices.setdefault(source, []).extend(value.get("choices") or [])
            elif source == "notes":
                notes = " ".join(value.get("text") or [])
        for key in ("support_status", "temporal_reference", "expected_verdict"):
            if len(choices.get(key, [])) != 1:
                raise AnnotationError(f"{who}: {name}: exactly one choice is required for {key}")
        if len(t_regions) > 1:
            raise AnnotationError(f"{who}: {name}: mark one contiguous CommitmentText span, found {len(t_regions)}")
        if len(temporal_regions) > 1:
            raise AnnotationError(f"{who}: {name}: mark at most one TemporalExpression span, found {len(temporal_regions)}")
        v_regions.sort(key=lambda pair: pair[0])
        commitment = validate_annotation(
            name,
            {
                "commitment_text": t_regions[0] if t_regions else "",
                "version_expression": [text for _, text in v_regions],
                "support_status": choices["support_status"][0],
                "temporal_reference": choices["temporal_reference"][0],
                "temporal_expression": temporal_regions[0] if temporal_regions else None,
            },
            choices["expected_verdict"][0],
            policies[name],
            who=who,
        )
        result[name] = Annotation(name, commitment, choices["expected_verdict"][0], notes)
    return result


def export_authenticity_warnings(path: PathLike, min_records: int = 10) -> List[str]:
    """Signs that a Label Studio export was produced by a script rather than by a person.

    Checked over the live annotations: every lead_time identical, all annotations created within
    60 seconds of each other, or a median lead_time under 20 seconds. Exports in the simple
    format carry no timing and give no warning.
    """
    data = read_json(Path(path))
    if not (isinstance(data, list) and all(isinstance(x, dict) and "data" in x for x in data)):
        return []
    live = [a for item in data for a in item.get("annotations") or [] if not a.get("was_cancelled")]
    if len(live) < min_records:
        return []
    who = Path(path).name
    warnings: List[str] = []
    lead_times = sorted(float(a["lead_time"]) for a in live if isinstance(a.get("lead_time"), (int, float)))
    if len(lead_times) == len(live) and len(set(lead_times)) == 1:
        warnings.append(f"{who}: every annotation has the same lead_time ({lead_times[0]} s)")
    if lead_times and lead_times[len(lead_times) // 2] < 20:
        warnings.append(f"{who}: median lead_time is {lead_times[len(lead_times) // 2]:.1f} s per policy")
    stamps = []
    for a in live:
        try:
            stamps.append(datetime.fromisoformat(str(a.get("created_at")).replace("Z", "+00:00")))
        except ValueError:
            pass
    if len(stamps) == len(live) and (max(stamps) - min(stamps)).total_seconds() < 60:
        warnings.append(f"{who}: all {len(live)} annotations were created within one minute")
    return warnings


def parse_annotator_file(path: PathLike, policies: Mapping[str, str], *, who: Optional[str] = None) -> Dict[str, Annotation]:
    """Parse and validate one annotator's export (Label Studio or simple format)."""
    path = Path(path)
    who = who or path.name
    data = read_json(path)
    is_label_studio = isinstance(data, list) and data and all(isinstance(x, dict) and "data" in x for x in data)
    if is_label_studio:
        return _parse_label_studio(data, policies, who)
    is_simple = (isinstance(data, dict) and "annotations" in data) or (
        isinstance(data, list) and all(isinstance(x, dict) and "package_name" in x for x in data)
    )
    if is_simple:
        return _parse_simple(data, policies, who)
    raise AnnotationError(f"{who}: unrecognised annotation file format")


def require_complete(annotations: Mapping[str, Annotation], policies: Mapping[str, str], who: str) -> None:
    missing = sorted(set(policies) - set(annotations))
    if missing:
        raise AnnotationError(
            f"{who}: {len(missing)} sampled record(s) have no annotation, for example {missing[:5]}. "
            "Every sampled policy must be annotated by both annotators."
        )
