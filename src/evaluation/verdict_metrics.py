"""Verdict metrics (thesis Section 3.3.3, research question Q2): macro-F1 and the 5x5 confusion matrix.

Macro-F1 averages the F1 of the classes that occur in the gold labels or in the predictions
of the evaluated sample. A class that is neither present nor predicted does not contribute,
a class that is predicted but absent from the gold labels contributes an F1 of zero.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from src.data_pipeline.constants import VERDICTS


def label_indices(labels: Sequence[str], values: Sequence[str]) -> np.ndarray:
    lookup = {label: i for i, label in enumerate(labels)}
    try:
        return np.fromiter((lookup[v] for v in values), dtype=np.int64, count=len(values))
    except KeyError as exc:
        raise ValueError(f"Label {exc.args[0]!r} is outside the vocabulary {list(labels)}") from None


def confusion_from_indices(gold: np.ndarray, pred: np.ndarray, n_labels: int) -> np.ndarray:
    """Rows: gold class, columns: predicted class."""
    flat = np.bincount(gold * n_labels + pred, minlength=n_labels * n_labels)
    return flat.reshape(n_labels, n_labels)


def prf_from_confusion(conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class precision, recall and F1; works on one matrix or a stack of matrices (..., L, L)."""
    conf = np.asarray(conf)
    tp = np.diagonal(conf, axis1=-2, axis2=-1).astype(float)
    predicted = conf.sum(axis=-2).astype(float)
    support = conf.sum(axis=-1).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def macro_f1_from_confusion(conf: np.ndarray):
    """Macro-F1 over the classes present in gold or prediction; a float for one matrix, an array for a stack."""
    conf = np.asarray(conf)
    _, _, f1 = prf_from_confusion(conf)
    present = (conf.sum(axis=-1) > 0) | (conf.sum(axis=-2) > 0)
    count = present.sum(axis=-1)
    total = np.where(present, f1, 0.0).sum(axis=-1)
    result = np.divide(total, count, out=np.zeros_like(total, dtype=float), where=count > 0)
    return float(result) if result.ndim == 0 else result


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] = VERDICTS) -> float:
    conf = confusion_from_indices(label_indices(labels, gold), label_indices(labels, pred), len(labels))
    return macro_f1_from_confusion(conf)


def classification_report(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] = VERDICTS) -> Dict[str, Any]:
    """Accuracy, macro-F1, per-class precision/recall/F1 and the confusion matrix."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    if not gold:
        raise ValueError("Cannot score an empty sample")
    conf = confusion_from_indices(label_indices(labels, gold), label_indices(labels, pred), len(labels))
    precision, recall, f1 = prf_from_confusion(conf)
    support = conf.sum(axis=1)
    predicted = conf.sum(axis=0)
    present = (support > 0) | (predicted > 0)
    report: Dict[str, Any] = {
        "n": len(gold),
        "accuracy": float(np.trace(conf) / len(gold)),
        "macro_f1": macro_f1_from_confusion(conf),
        "labels_in_macro": [label for label, keep in zip(labels, present) if keep],
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "predicted": int(predicted[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {
            "rows": "gold",
            "columns": "predicted",
            "labels": list(labels),
            "matrix": conf.tolist(),
        },
    }
    if "Insufficient Evidence" in labels and "Potential Inconsistency" in labels:
        ie, pi = labels.index("Insufficient Evidence"), labels.index("Potential Inconsistency")
        report["insufficient_vs_potential"] = {
            "gold_insufficient_predicted_potential": int(conf[ie, pi]),
            "gold_potential_predicted_insufficient": int(conf[pi, ie]),
        }
    return report
