"""Extraction metrics for the commitment tuple (thesis 3.3.3).

Pure functions, standard library only. Per element:

    t    token-level precision, recall and F1 over the multiset of lower-cased word tokens
    v    set precision, recall and F1 over normalised version tokens
    tau  exact match of the label
    T    exact match of the label

Tuple level:

    Exact Match    1.0 only if all four elements match exactly (t: identical token sequence,
                   v: identical set, tau and T: identical labels), else 0.0
    Partial Match  mean of the four element scores (t: token F1, v: set F1, tau and T: 0/1)

Two empty values count as a perfect match. If exactly one side is empty the score is 0.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

TUPLE_KEYS = ("commitment_text", "version_expression", "support_status", "temporal_reference")
_WORD_RE = re.compile(r"\w+")
_LEADING_V_RE = re.compile(r"^v(?=\d)")

PRF = Tuple[float, float, float]


def tokenize(text: str) -> List[str]:
    """Lower-cased word tokens; punctuation and markdown symbols are ignored."""
    return _WORD_RE.findall((text or "").lower())


def prf_from_counts(n_common: int, n_pred: int, n_gold: int) -> PRF:
    """Precision, recall and F1 from overlap counts."""
    if n_pred == 0 and n_gold == 0:
        return 1.0, 1.0, 1.0
    if n_common == 0:
        return 0.0, 0.0, 0.0
    precision = n_common / n_pred
    recall = n_common / n_gold
    return precision, recall, 2 * precision * recall / (precision + recall)


def token_overlap_counts(pred_text: str, gold_text: str) -> Tuple[int, int, int]:
    """(common tokens as multiset intersection, predicted tokens, gold tokens)."""
    pred, gold = Counter(tokenize(pred_text)), Counter(tokenize(gold_text))
    return sum((pred & gold).values()), sum(pred.values()), sum(gold.values())


def token_prf(pred_text: str, gold_text: str) -> PRF:
    """Token-level precision, recall and F1 of the commitment text t."""
    return prf_from_counts(*token_overlap_counts(pred_text, gold_text))


def normalise_version(token: str) -> str:
    """Case, whitespace and a leading 'v' are ignored: 'V3.1' and '3.1' are the same token."""
    return _LEADING_V_RE.sub("", re.sub(r"\s+", "", str(token)).lower())


def version_set(tokens: Iterable[str]) -> frozenset:
    return frozenset(normalised for normalised in (normalise_version(t) for t in tokens) if normalised)


def set_prf(pred_versions: Iterable[str], gold_versions: Iterable[str]) -> PRF:
    """Set precision, recall and F1 of the version expression v."""
    pred, gold = version_set(pred_versions), version_set(gold_versions)
    return prf_from_counts(len(pred & gold), len(pred), len(gold))


def label_match(pred_label: str, gold_label: str) -> float:
    """Exact match of tau or T."""
    return 1.0 if str(pred_label).strip().lower() == str(gold_label).strip().lower() else 0.0


def as_tuple_dict(commitment: Any) -> Dict[str, Any]:
    """Accept a plain dict or an object with `to_dict()` and keep the four thesis keys."""
    data = commitment.to_dict() if hasattr(commitment, "to_dict") else commitment
    return {key: data[key] for key in TUPLE_KEYS}


def score_tuple(pred: Any, gold: Any) -> Dict[str, float]:
    """All per-example scores of one prediction against its gold tuple."""
    p, g = as_tuple_dict(pred), as_tuple_dict(gold)
    t_precision, t_recall, t_f1 = token_prf(p["commitment_text"], g["commitment_text"])
    v_precision, v_recall, v_f1 = set_prf(p["version_expression"], g["version_expression"])
    tau = label_match(p["support_status"], g["support_status"])
    time = label_match(p["temporal_reference"], g["temporal_reference"])
    t_exact = 1.0 if tokenize(p["commitment_text"]) == tokenize(g["commitment_text"]) else 0.0
    v_exact = 1.0 if version_set(p["version_expression"]) == version_set(g["version_expression"]) else 0.0
    return {
        "t_precision": t_precision,
        "t_recall": t_recall,
        "t_f1": t_f1,
        "v_precision": v_precision,
        "v_recall": v_recall,
        "v_f1": v_f1,
        "tau_match": tau,
        "T_match": time,
        "tuple_exact_match": 1.0 if (t_exact and v_exact and tau and time) else 0.0,
        "tuple_partial_match": (t_f1 + v_f1 + tau + time) / 4.0,
    }


def tuple_exact_match(pred: Any, gold: Any) -> float:
    return score_tuple(pred, gold)["tuple_exact_match"]


def tuple_partial_match(pred: Any, gold: Any) -> float:
    return score_tuple(pred, gold)["tuple_partial_match"]


def mean_scores(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Macro average of per-example score dictionaries (keys of the first row)."""
    if not rows:
        raise ValueError("cannot average an empty list of scores")
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
