"""Extraction metrics (thesis Section 3.3.3, research question Q1).

The per-record scoring is delegated to the Layer 1 helper module
``src/layer1_ai_extraction/metrics.py`` so that training-time validation and the final
evaluation use one definition. This module aggregates the per-record scores.

Contract used from ``metrics.score_tuple(pred, gold) -> dict``; both arguments are commitment
dicts with the four thesis keys, the result holds floats in [0, 1] under these keys:

    t_precision, t_recall, t_f1          token-level scores of the commitment text t
    v_precision, v_recall, v_f1          token-level scores of the version expression v
    tau_match, T_match                   1.0 when tau (respectively T) equals the gold value
    tuple_exact_match                    1.0 when the whole tuple c matches exactly (EM)
    tuple_partial_match                  partial credit for the tuple c (PM)

For the single-label fields tau and T the token-level precision, recall and F1 reduce to
the match indicator; their accuracy and the macro-F1 over the categories are reported.
Scores are averaged over the test records (per-record averaging).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.data_pipeline.constants import TAU_VALUES, TEMPORAL_VALUES

from .verdict_metrics import macro_f1

REQUIRED_SCORE_KEYS = (
    "t_precision",
    "t_recall",
    "t_f1",
    "v_precision",
    "v_recall",
    "v_f1",
    "tau_match",
    "T_match",
    "tuple_exact_match",
    "tuple_partial_match",
)

METRICS_MODULE = "src.layer1_ai_extraction.metrics"
SCORER_NAME = "score_tuple"

Scorer = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, float]]


class MetricsUnavailableError(RuntimeError):
    """The Layer 1 metrics helpers cannot be used; the evaluation must not substitute its own."""


def resolve_scorer() -> Scorer:
    try:
        module = importlib.import_module(METRICS_MODULE)
    except ImportError as exc:
        raise MetricsUnavailableError(
            f"{METRICS_MODULE} cannot be imported ({exc}). The evaluation uses the Layer 1 metric helpers."
        ) from exc
    scorer = getattr(module, SCORER_NAME, None)
    if not callable(scorer):
        raise MetricsUnavailableError(
            f"{METRICS_MODULE} has no callable {SCORER_NAME}(pred, gold). See the contract in "
            "src/evaluation/extraction_eval.py."
        )
    return scorer


def _label_block(gold: Sequence[str], pred: Sequence[str], vocabulary: Sequence[str]) -> Dict[str, float]:
    extras = sorted(set(pred) - set(vocabulary), key=str)
    labels = list(vocabulary) + extras
    return {
        "accuracy": float(np.mean([g == p for g, p in zip(gold, pred)])),
        "macro_f1": macro_f1(gold, pred, labels),
    }


def score_columns(
    predicted: Sequence[Mapping[str, Any]], gold: Sequence[Mapping[str, Any]], scorer: Scorer
) -> Dict[str, List[float]]:
    """Per-record scores of one system, one list per score key, in test-record order."""
    if len(predicted) != len(gold) or not gold:
        raise ValueError("Predicted and gold commitments must be non-empty and of equal length")
    columns: Dict[str, List[float]] = {key: [] for key in REQUIRED_SCORE_KEYS}
    for pred, ref in zip(predicted, gold):
        scores = scorer(pred, ref)
        missing = [k for k in REQUIRED_SCORE_KEYS if k not in scores]
        if missing:
            raise MetricsUnavailableError(f"{SCORER_NAME} returned no value for {missing}")
        for key in REQUIRED_SCORE_KEYS:
            columns[key].append(float(scores[key]))
    return columns


def aggregate_extraction(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    scorer: Scorer,
    columns: Optional[Mapping[str, Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Average the per-record scores of one system over the test records."""
    if len(predicted) != len(gold) or not gold:
        raise ValueError("Predicted and gold commitments must be non-empty and of equal length")
    if columns is None:
        columns = score_columns(predicted, gold, scorer)
    mean = {key: float(np.mean(values)) for key, values in columns.items()}
    return {
        "n_records": len(gold),
        "t": {"precision": mean["t_precision"], "recall": mean["t_recall"], "f1": mean["t_f1"]},
        "v": {"precision": mean["v_precision"], "recall": mean["v_recall"], "f1": mean["v_f1"]},
        "tau": _label_block(
            [g["support_status"] for g in gold], [p.get("support_status") for p in predicted], TAU_VALUES
        ),
        "T": _label_block(
            [g["temporal_reference"] for g in gold], [p.get("temporal_reference") for p in predicted], TEMPORAL_VALUES
        ),
        "tuple": {"exact_match": mean["tuple_exact_match"], "partial_match": mean["tuple_partial_match"]},
    }
