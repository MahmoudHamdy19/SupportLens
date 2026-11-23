"""Statistical analysis (thesis Section 3.3.4).

* Bootstrap 95% confidence interval of the macro-F1 (1,000 resamples of the test set with
  replacement, seed 42, 2.5th and 97.5th percentiles). The same resample indices are used
  for every system, so the intervals are paired. ``bootstrap_mean_ci`` applies the same
  percentile bootstrap to per-record extraction scores (t F1, v F1, tuple EM and PM).
* McNemar's test with continuity correction on paired per-record correctness (verdict
  correctness, and tuple exact match for the extraction):
  chi2 = (|b - c| - 1)^2 / (b + c), where b counts records the baseline gets right and the
  fine-tuned model wrong, and c the reverse. A difference of zero gives chi2 = 0.
* Mann-Whitney U test for robustness (Q3) between the two halves of a median split on one
  project characteristic. The "low" group holds values <= median, the "high" group values
  > median. ``u_low`` is the usual U statistic of the low group (rank sum minus
  n1 (n1 + 1) / 2); the thesis formula U1 = n1 n2 + n1 (n1 + 1) / 2 - R1 equals ``u_high``.
  ``mann_whitney_groups`` runs the same test between two named groups (active versus
  abandoned projects, research question RQ3).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy import stats as scipy_stats

from src.data_pipeline.constants import SEED, VERDICTS

from .verdict_metrics import confusion_from_indices, label_indices, macro_f1_from_confusion


# --------------------------------------------------------------------------- bootstrap


def bootstrap_indices(n: int, n_resamples: int = 1000, seed: int = SEED) -> np.ndarray:
    """Resample indices, shape (n_resamples, n); generated once and shared by all systems."""
    if n <= 0:
        raise ValueError("Cannot bootstrap an empty sample")
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def bootstrap_macro_f1(
    gold: Sequence[str],
    pred: Sequence[str],
    indices: np.ndarray,
    labels: Sequence[str] = VERDICTS,
    *,
    level: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    gold_idx = label_indices(labels, gold)
    pred_idx = label_indices(labels, pred)
    n_labels = len(labels)
    resamples = len(indices)
    offsets = np.arange(resamples)[:, None] * n_labels * n_labels
    flat = (offsets + gold_idx[indices] * n_labels + pred_idx[indices]).ravel()
    stack = np.bincount(flat, minlength=resamples * n_labels * n_labels).reshape(resamples, n_labels, n_labels)
    scores = macro_f1_from_confusion(stack)
    alpha = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(scores, [alpha, 100.0 - alpha])
    point = macro_f1_from_confusion(confusion_from_indices(gold_idx, pred_idx, len(labels)))
    return {
        "macro_f1": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "level": level,
        "n_resamples": int(len(indices)),
        "seed": seed,
        "bootstrap_mean": float(scores.mean()),
    }


def bootstrap_mean_ci(
    values: Sequence[float], indices: np.ndarray, *, level: float = 0.95, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Percentile bootstrap CI of a per-record mean (extraction F1, tuple EM/PM) on shared indices."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if indices.shape[1] != len(data):
        raise ValueError("the resample indices do not match the number of records")
    scores = data[indices].mean(axis=1)
    alpha = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(scores, [alpha, 100.0 - alpha])
    return {
        "value": float(data.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "level": level,
        "n_resamples": int(len(indices)),
        "seed": seed,
        "bootstrap_mean": float(scores.mean()),
    }


# --------------------------------------------------------------------------- McNemar


def mcnemar_test(baseline_correct: Sequence[bool], proposed_correct: Sequence[bool]) -> Dict[str, Any]:
    """McNemar's test with continuity correction on paired per-record correctness."""
    if len(baseline_correct) != len(proposed_correct):
        raise ValueError("The two correctness vectors must have the same length")
    base = np.asarray(baseline_correct, dtype=bool)
    prop = np.asarray(proposed_correct, dtype=bool)
    b = int(np.sum(base & ~prop))
    c = int(np.sum(~base & prop))
    discordant = b + c
    if discordant == 0:
        chi2, p_value, p_exact = 0.0, 1.0, 1.0
    else:
        chi2 = float(max(abs(b - c) - 1, 0) ** 2 / discordant)
        p_value = float(scipy_stats.chi2.sf(chi2, df=1))
        p_exact = float(scipy_stats.binomtest(min(b, c), discordant, 0.5).pvalue)
    return {
        "b_baseline_correct_proposed_wrong": b,
        "c_baseline_wrong_proposed_correct": c,
        "both_correct": int(np.sum(base & prop)),
        "both_wrong": int(np.sum(~base & ~prop)),
        "n": int(len(base)),
        "chi2": chi2,
        "p_value": p_value,
        "p_value_exact_binomial": p_exact,
        "continuity_correction": True,
        "significant_at_0.05": bool(p_value < 0.05),
        "proposed_better": bool(c > b),
    }


# --------------------------------------------------------------------------- Mann-Whitney


def mann_whitney_median_split(
    correct: Sequence[float], characteristic: Sequence[Optional[float]], name: str
) -> Dict[str, Any]:
    """Compare per-record correctness between the halves of a median split on ``characteristic``."""
    if len(correct) != len(characteristic):
        raise ValueError("correct and characteristic must have the same length")
    pairs = [(float(x), float(y)) for x, y in zip(correct, characteristic) if y is not None and not math.isnan(y)]
    result: Dict[str, Any] = {
        "characteristic": name,
        "n_used": len(pairs),
        "n_missing": len(correct) - len(pairs),
    }
    if len(pairs) < 2:
        return {**result, "status": "skipped", "reason": "fewer than two records with a value"}
    values = np.array([p[1] for p in pairs])
    outcomes = np.array([p[0] for p in pairs])
    median = float(np.median(values))
    low = outcomes[values <= median]
    high = outcomes[values > median]
    result.update({"median": median, "n_low": int(len(low)), "n_high": int(len(high))})
    if len(low) == 0 or len(high) == 0:
        return {**result, "status": "skipped", "reason": "the median split leaves one group empty"}
    result.update({"accuracy_low": float(low.mean()), "accuracy_high": float(high.mean())})
    u_low, u_high, p_value = _mann_whitney_u(low, high)
    result.update(
        {
            "status": "ok",
            "u_low": u_low,
            "u_high": u_high,
            "p_value": p_value,
            "significant_at_0.05": bool(p_value < 0.05),
        }
    )
    return result


def _mann_whitney_u(first: np.ndarray, second: np.ndarray) -> "tuple[float, float, float]":
    """(U of the first group, U of the second group, two-sided asymptotic p with continuity correction)."""
    ranks = scipy_stats.rankdata(np.concatenate([first, second]))
    u_first = float(ranks[: len(first)].sum() - len(first) * (len(first) + 1) / 2)
    u_second = float(len(first) * len(second) - u_first)
    combined = np.concatenate([first, second])
    if np.all(combined == combined[0]):
        return u_first, u_second, 1.0
    p_value = float(
        scipy_stats.mannwhitneyu(first, second, alternative="two-sided", method="asymptotic", use_continuity=True).pvalue
    )
    return u_first, u_second, p_value


def mann_whitney_groups(
    correct: Sequence[float], groups: Sequence[Optional[str]], first: str, second: str
) -> Dict[str, Any]:
    """Compare per-record correctness between two named groups; records with another or no group are left out."""
    if len(correct) != len(groups):
        raise ValueError("correct and groups must have the same length")
    a = np.array([float(x) for x, g in zip(correct, groups) if g == first])
    b = np.array([float(x) for x, g in zip(correct, groups) if g == second])
    result: Dict[str, Any] = {
        "groups": [first, second],
        f"n_{first}": int(len(a)),
        f"n_{second}": int(len(b)),
        "n_unclassified": int(len(correct) - len(a) - len(b)),
    }
    if len(a) == 0 or len(b) == 0:
        return {**result, "status": "skipped", "reason": "one of the two groups is empty"}
    u_first, u_second, p_value = _mann_whitney_u(a, b)
    result.update(
        {
            "status": "ok",
            f"accuracy_{first}": float(a.mean()),
            f"accuracy_{second}": float(b.mean()),
            f"u_{first}": u_first,
            f"u_{second}": u_second,
            "p_value": p_value,
            "significant_at_0.05": bool(p_value < 0.05),
        }
    )
    return result
