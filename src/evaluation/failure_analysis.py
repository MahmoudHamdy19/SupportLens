"""Failure analysis for research question RQ3 (thesis Chapter 1 and Table 3.2).

RQ3 asks which classes of cases predominantly produce failure or insufficient-evidence
outcomes. For every system the analysis reports, from the predictions and the test ground truth:

* ``verdict_error_pairs``: counts of (gold verdict, predicted verdict) for the wrong verdicts,
  as ``{gold: {predicted: count}}``;
* ``wrong_predictions`` and ``insufficient_evidence_predictions``: how many records fall in each
  group and, per Layer 3 rule (R1 to R5) and per reason code of the evidence log, the number of
  those records in which the rule fired or the reason code occurred (each counted once per record);
* ``extraction_errors_by_gold_tau``: per annotated commitment type tau, the number of records,
  tuple exact-match errors, tau errors, verdict errors and the mean t and v F1;
* ``verdict_errors_by_activity``: verdict errors among active and abandoned projects.

Active versus abandoned (Table 3.2 example)
    A project is **active** when its repository is not archived and its last push (``pushed_at``)
    lies at most 365 days before the evidence collection time (``collected_at``). It is
    **abandoned** when it is archived, or when its last push is more than 365 days before the
    collection time. A record without ``collected_at``, or without ``pushed_at`` while not
    archived, is **unclassified** and left out of the comparison. A missing ``archived`` field
    counts as not archived.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.data_pipeline.constants import TAU_VALUES, VERDICTS

ACTIVE = "active"
ABANDONED = "abandoned"
ACTIVITY_WINDOW_DAYS = 365
INSUFFICIENT_EVIDENCE = "Insufficient Evidence"

ACTIVITY_DEFINITION = (
    f"active: not archived and pushed_at at most {ACTIVITY_WINDOW_DAYS} days before collected_at; "
    f"abandoned: archived, or pushed_at more than {ACTIVITY_WINDOW_DAYS} days before collected_at; "
    "unclassified: collected_at missing, or pushed_at missing for a non-archived repository; "
    "a missing archived field counts as not archived"
)


def _time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def project_activity(record: Mapping[str, Any]) -> Optional[str]:
    """``"active"``, ``"abandoned"`` or None (unclassified) under the definition above."""
    collected = _time(record.get("collected_at"))
    if collected is None:
        return None
    if record.get("repo_archived", record.get("archived")) is True:
        return ABANDONED
    pushed = _time(record.get("pushed_at"))
    if pushed is None:
        return None
    return ACTIVE if collected - pushed <= timedelta(days=ACTIVITY_WINDOW_DAYS) else ABANDONED


def _sorted_counts(counter: Counter) -> Dict[str, int]:
    return {key: counter[key] for key in sorted(counter, key=str)}


def _rule_and_reason_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rules: Counter = Counter()
    reasons: Counter = Counter()
    for row in rows:
        rules.update(set(row.get("rules_triggered") or []))
        reasons.update(set(row.get("reason_codes") or []))
    return {"n": len(rows), "rules_triggered": _sorted_counts(rules), "reason_codes": _sorted_counts(reasons)}


def _error_pairs(gold: Sequence[str], predicted: Sequence[str]) -> Dict[str, Dict[str, int]]:
    order = {label: i for i, label in enumerate(VERDICTS)}
    pairs: Dict[str, Counter] = {}
    for g, p in zip(gold, predicted):
        if g != p:
            pairs.setdefault(g, Counter())[p] += 1
    rank = lambda label: (order.get(label, len(order)), str(label))  # noqa: E731
    return {g: {p: pairs[g][p] for p in sorted(pairs[g], key=rank)} for g in sorted(pairs, key=rank)}


def _by_gold_tau(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    columns: Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    values = [r["commitment"]["support_status"] for r in records]
    labels = list(TAU_VALUES) + sorted(set(values) - set(TAU_VALUES), key=str)
    out: Dict[str, Any] = {}
    for tau in labels:
        idx = [i for i, value in enumerate(values) if value == tau]
        if not idx:
            out[tau] = {"n": 0}
            continue
        out[tau] = {
            "n": len(idx),
            "tuple_exact_match_errors": sum(1 for i in idx if columns["tuple_exact_match"][i] < 1.0),
            "tau_errors": sum(1 for i in idx if rows[i]["commitment"].get("support_status") != tau),
            "verdict_errors": sum(1 for i in idx if rows[i]["verdict"] != records[i]["verdict"]),
            "mean_t_f1": sum(columns["t_f1"][i] for i in idx) / len(idx),
            "mean_v_f1": sum(columns["v_f1"][i] for i in idx) / len(idx),
        }
    return out


def failure_analysis(
    records: Sequence[Mapping[str, Any]],
    by_system: Mapping[str, Sequence[Mapping[str, Any]]],
    columns_by_system: Mapping[str, Mapping[str, Sequence[float]]],
) -> Dict[str, Any]:
    """The failure-analysis document for all systems (rows aligned with ``records``)."""
    gold = [r["verdict"] for r in records]
    activity = [project_activity(r) for r in records]
    activity_counts = Counter(a if a else "unclassified" for a in activity)
    systems: Dict[str, Any] = {}
    for system, rows in by_system.items():
        predicted = [row["verdict"] for row in rows]
        wrong = [row for row, g in zip(rows, gold) if row["verdict"] != g]
        insufficient = [row for row in rows if row["verdict"] == INSUFFICIENT_EVIDENCE]
        by_activity: Dict[str, Any] = {}
        for group in (ACTIVE, ABANDONED, "unclassified"):
            idx = [i for i, a in enumerate(activity) if (a or "unclassified") == group]
            by_activity[group] = {
                "n": len(idx),
                "verdict_errors": sum(1 for i in idx if predicted[i] != gold[i]),
                "insufficient_evidence_predicted": sum(1 for i in idx if predicted[i] == INSUFFICIENT_EVIDENCE),
            }
        systems[system] = {
            "n_records": len(rows),
            "verdict_error_pairs": _error_pairs(gold, predicted),
            "wrong_predictions": _rule_and_reason_counts(wrong),
            "insufficient_evidence_predictions": _rule_and_reason_counts(insufficient),
            "extraction_errors_by_gold_tau": _by_gold_tau(records, rows, columns_by_system[system]),
            "verdict_errors_by_activity": by_activity,
        }
    return {
        "n_records": len(records),
        "research_question": "RQ3: which classes of cases predominantly produce failure or insufficient-evidence outcomes",
        "activity_definition": ACTIVITY_DEFINITION,
        "activity_counts": _sorted_counts(activity_counts),
        "gold_verdict_counts": _sorted_counts(Counter(gold)),
        "systems": systems,
    }


__all__: List[str] = [
    "ACTIVE",
    "ABANDONED",
    "ACTIVITY_DEFINITION",
    "ACTIVITY_WINDOW_DAYS",
    "failure_analysis",
    "project_activity",
]
