"""Inter-annotator agreement before adjudication (thesis Section 3.3.1, Table 3.2).

Cohen's kappa for the verdict (target >= 0.80), percent agreement, per-field agreement for
tau, T, v and t, the verdict confusion matrix between the annotators and the list of
disagreements that the supervisor has to resolve.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence

from .annotation import Annotation, consistency_warnings
from .constants import KAPPA_TARGET, VERDICTS
from .io_utils import DataError

COMPARED_FIELDS = ("verdict", "support_status", "temporal_reference", "version_expression", "commitment_text", "temporal_expression")


def confusion_matrix(first: Sequence[str], second: Sequence[str], labels: Sequence[str]) -> List[List[int]]:
    """Rows are the first rater's labels, columns the second rater's."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for a, b in zip(first, second):
        if a not in index or b not in index:
            raise ValueError(f"Label outside the vocabulary: {a!r} / {b!r}")
        matrix[index[a]][index[b]] += 1
    return matrix


def cohens_kappa(first: Sequence[str], second: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
    """Cohen's kappa = (p_o - p_e) / (1 - p_e) over the fixed label set.

    When both raters always use the same single label, p_e = 1 and kappa is undefined;
    it is reported as 1.0 with ``degenerate`` set so that the situation is visible.
    """
    if len(first) != len(second):
        raise ValueError("The two raters must label the same number of items")
    n = len(first)
    if n == 0:
        raise ValueError("Cannot compute kappa without items")
    matrix = confusion_matrix(first, second, labels)
    observed = sum(matrix[i][i] for i in range(len(labels))) / n
    row = [sum(r) for r in matrix]
    col = [sum(matrix[i][j] for i in range(len(labels))) for j in range(len(labels))]
    expected = sum(row[i] * col[i] for i in range(len(labels))) / (n * n)
    degenerate = expected >= 1.0
    kappa = 1.0 if degenerate else (observed - expected) / (1.0 - expected)
    return {
        "cohens_kappa": kappa,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "degenerate": degenerate,
        "n": n,
        "confusion_matrix": matrix,
    }


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def token_f1(first: str, second: str) -> float:
    a, b = _tokens(first), _tokens(second)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = sum((Counter(a) & Counter(b)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(b)
    return 2 * precision * recall / (precision + recall)


def set_f1(first: Sequence[str], second: Sequence[str]) -> float:
    a, b = set(first), set(second)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    hit = len(a & b)
    if hit == 0:
        return 0.0
    precision, recall = hit / len(a), hit / len(b)
    return 2 * precision * recall / (precision + recall)


def field_value(annotation: Annotation, field: str) -> Any:
    return annotation.verdict if field == "verdict" else annotation.commitment.get(field)


def comparable(field: str, value: Any) -> Any:
    """Normalised form used to decide whether two annotators agree on a field."""
    if field == "version_expression":
        return tuple(sorted({" ".join(item.split()).lower() for item in value}))
    if field in ("commitment_text", "temporal_expression"):
        return " ".join((value or "").split())
    return value


def find_disagreements(first: Mapping[str, Annotation], second: Mapping[str, Annotation]) -> Dict[str, List[str]]:
    """Map each package with at least one differing field to the list of differing fields."""
    result: Dict[str, List[str]] = {}
    for name in sorted(set(first) & set(second)):
        differing = [
            field
            for field in COMPARED_FIELDS
            if comparable(field, field_value(first[name], field)) != comparable(field, field_value(second[name], field))
        ]
        if differing:
            result[name] = differing
    return result


def _categorical_block(first: Sequence[str], second: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
    stats = cohens_kappa(first, second, labels)
    return {
        "percent_agreement": 100.0 * stats["observed_agreement"],
        "observed_agreement": stats["observed_agreement"],
        "expected_agreement": stats["expected_agreement"],
        "cohens_kappa": stats["cohens_kappa"],
        "degenerate": stats["degenerate"],
        "labels": list(labels),
        "confusion_matrix": stats["confusion_matrix"],
    }


def build_agreement_report(
    first: Mapping[str, Annotation],
    second: Mapping[str, Annotation],
    *,
    tau_values: Sequence[str],
    temporal_values: Sequence[str],
) -> Dict[str, Any]:
    """Agreement statistics for two complete annotation sets over the same packages."""
    if set(first) != set(second):
        raise DataError("The two annotators must cover exactly the same packages")
    names = sorted(first)
    if not names:
        raise DataError("No annotated records to compare")

    verdicts = _categorical_block([first[n].verdict for n in names], [second[n].verdict for n in names], VERDICTS)
    tau = _categorical_block(
        [first[n].commitment["support_status"] for n in names],
        [second[n].commitment["support_status"] for n in names],
        tau_values,
    )
    temporal = _categorical_block(
        [first[n].commitment["temporal_reference"] for n in names],
        [second[n].commitment["temporal_reference"] for n in names],
        temporal_values,
    )
    v_exact = [
        comparable("version_expression", first[n].commitment["version_expression"])
        == comparable("version_expression", second[n].commitment["version_expression"])
        for n in names
    ]
    t_exact = [
        comparable("commitment_text", first[n].commitment["commitment_text"])
        == comparable("commitment_text", second[n].commitment["commitment_text"])
        for n in names
    ]
    v_f1 = [
        set_f1(
            comparable("version_expression", first[n].commitment["version_expression"]),
            comparable("version_expression", second[n].commitment["version_expression"]),
        )
        for n in names
    ]
    s_exact = [
        comparable("temporal_expression", first[n].commitment.get("temporal_expression"))
        == comparable("temporal_expression", second[n].commitment.get("temporal_expression"))
        for n in names
    ]
    s_f1 = [
        token_f1(first[n].commitment.get("temporal_expression") or "", second[n].commitment.get("temporal_expression") or "")
        for n in names
    ]
    t_f1 = [token_f1(first[n].commitment["commitment_text"], second[n].commitment["commitment_text"]) for n in names]

    disagreements = []
    for name, fields in find_disagreements(first, second).items():
        disagreements.append(
            {
                "package_name": name,
                "fields": {
                    field: {"annotator1": field_value(first[name], field), "annotator2": field_value(second[name], field)}
                    for field in fields
                },
            }
        )

    warnings: List[str] = []
    for label, annotations in (("annotator1", first), ("annotator2", second)):
        for name in names:
            warnings.extend(f"{label}: {w}" for w in consistency_warnings(annotations[name]))
    if verdicts["cohens_kappa"] < KAPPA_TARGET:
        warnings.append(
            f"verdict kappa {verdicts['cohens_kappa']:.4f} is below the thesis target {KAPPA_TARGET}; "
            "review the codebook with the annotators before adjudication"
        )
    if verdicts["degenerate"]:
        warnings.append("kappa is degenerate: both annotators used a single verdict category throughout")
    if not disagreements:
        warnings.append("the annotators agree on every field of every record; verify that they worked independently")

    return {
        "n_records": len(names),
        "cohens_kappa": verdicts["cohens_kappa"],
        "kappa_target": KAPPA_TARGET,
        "meets_target": verdicts["cohens_kappa"] >= KAPPA_TARGET,
        "observed_agreement": verdicts["observed_agreement"],
        "expected_agreement": verdicts["expected_agreement"],
        "percent_agreement": verdicts["percent_agreement"],
        "degenerate": verdicts["degenerate"],
        "confusion_matrix": {
            "rows": "annotator1",
            "columns": "annotator2",
            "labels": list(VERDICTS),
            "matrix": verdicts["confusion_matrix"],
        },
        "fields": {
            "support_status": tau,
            "temporal_reference": temporal,
            "version_expression": {
                "percent_agreement": 100.0 * sum(v_exact) / len(names),
                "mean_set_f1": sum(v_f1) / len(names),
            },
            "temporal_expression": {
                "percent_agreement": 100.0 * sum(s_exact) / len(names),
                "mean_token_f1": sum(s_f1) / len(names),
            },
            "commitment_text": {
                "percent_agreement": 100.0 * sum(t_exact) / len(names),
                "mean_token_f1": sum(t_f1) / len(names),
            },
        },
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "warnings": warnings,
    }
