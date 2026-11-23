"""Verdict metrics, bootstrap CI, McNemar and Mann-Whitney against hand-computed values (synthetic labels)."""

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from src.data_pipeline.constants import VERDICTS
from src.evaluation.stats import bootstrap_indices, bootstrap_macro_f1, mann_whitney_median_split, mcnemar_test
from src.evaluation.verdict_metrics import classification_report, macro_f1

C, P, PI, IE, U = VERDICTS


# ----------------------------------------------------------------------------- verdict metrics


def test_macro_f1_and_confusion_matrix_by_hand():
    gold = [C, C, IE, IE, IE, PI]
    pred = [C, IE, IE, IE, PI, PI]
    report = classification_report(gold, pred)
    # Consistent: tp1 fp0 fn1 -> P1 R.5 F.6667; Insufficient: tp2 fp1 fn1 -> P.6667 R.6667 F.6667;
    # Potential: tp1 fp1 fn0 -> P.5 R1 F.6667. Partially/Unmeasurable are neither present nor predicted.
    assert report["per_class"][C]["f1"] == pytest.approx(2 / 3)
    assert report["per_class"][IE]["precision"] == pytest.approx(2 / 3)
    assert report["per_class"][PI]["precision"] == pytest.approx(0.5)
    assert report["macro_f1"] == pytest.approx(2 / 3)
    assert report["labels_in_macro"] == [C, PI, IE]
    assert report["accuracy"] == pytest.approx(4 / 6)
    matrix = report["confusion_matrix"]["matrix"]
    assert matrix[VERDICTS.index(IE)][VERDICTS.index(PI)] == 1
    assert report["insufficient_vs_potential"] == {
        "gold_insufficient_predicted_potential": 1,
        "gold_potential_predicted_insufficient": 0,
    }
    assert sum(sum(row) for row in matrix) == 6


def test_predicting_an_absent_class_costs_macro_f1():
    gold = [C, C, C, C]
    assert macro_f1(gold, [C, C, C, C]) == pytest.approx(1.0)
    assert macro_f1(gold, [C, C, C, P]) == pytest.approx((6 / 7 + 0.0) / 2)  # P predicted but absent: F1 = 0


def test_unknown_label_is_an_error():
    with pytest.raises(ValueError):
        classification_report([C], ["Something else"])


# ----------------------------------------------------------------------------- bootstrap


def test_bootstrap_is_deterministic_and_paired():
    gold = [C, C, IE, IE, IE, PI, P, U] * 5
    pred = [C, IE, IE, IE, PI, PI, P, U] * 5
    indices = bootstrap_indices(len(gold), 1000, 42)
    first = bootstrap_macro_f1(gold, pred, indices, seed=42)
    second = bootstrap_macro_f1(gold, pred, bootstrap_indices(len(gold), 1000, 42), seed=42)
    assert first == second
    assert first["n_resamples"] == 1000 and first["level"] == 0.95
    assert first["ci_low"] <= first["macro_f1"] <= first["ci_high"] or first["ci_low"] <= first["bootstrap_mean"] <= first["ci_high"]
    assert 0.0 <= first["ci_low"] <= first["ci_high"] <= 1.0
    other = bootstrap_macro_f1(gold, pred, bootstrap_indices(len(gold), 1000, 7), seed=7)
    assert (other["ci_low"], other["ci_high"]) != (first["ci_low"], first["ci_high"])


def test_bootstrap_of_a_perfect_system_is_degenerate_at_one():
    gold = [C, IE, PI, C, IE]
    result = bootstrap_macro_f1(gold, list(gold), bootstrap_indices(5, 200, 42))
    assert result["ci_low"] == result["ci_high"] == 1.0


def test_bootstrap_percentiles_match_the_definition():
    gold = [C, C, IE, IE, IE, PI]
    pred = [C, IE, IE, IE, PI, PI]
    indices = bootstrap_indices(6, 1000, 42)
    scores = []
    for sample in indices:
        scores.append(macro_f1([gold[i] for i in sample], [pred[i] for i in sample]))
    low, high = np.percentile(scores, [2.5, 97.5])
    result = bootstrap_macro_f1(gold, pred, indices)
    assert result["ci_low"] == pytest.approx(low) and result["ci_high"] == pytest.approx(high)


# ----------------------------------------------------------------------------- McNemar


def test_mcnemar_against_hand_computed_table():
    # Build 40 paired outcomes: both correct 20, both wrong 0, b = 5, c = 15.
    baseline = [True] * 20 + [True] * 5 + [False] * 15
    proposed = [True] * 20 + [False] * 5 + [True] * 15
    result = mcnemar_test(baseline, proposed)
    assert result["b_baseline_correct_proposed_wrong"] == 5
    assert result["c_baseline_wrong_proposed_correct"] == 15
    assert result["both_correct"] == 20 and result["both_wrong"] == 0 and result["n"] == 40
    # chi2 = (|5 - 15| - 1)^2 / 20 = 81 / 20 = 4.05 ; p = erfc(sqrt(chi2 / 2)) for one degree of freedom
    assert result["chi2"] == pytest.approx(4.05)
    assert result["p_value"] == pytest.approx(math.erfc(math.sqrt(4.05 / 2)))
    assert result["p_value"] == pytest.approx(0.04417, abs=1e-4)
    assert result["significant_at_0.05"] is True and result["proposed_better"] is True
    assert result["p_value_exact_binomial"] == pytest.approx(scipy_stats.binomtest(5, 20, 0.5).pvalue)


def test_mcnemar_equal_discordance_and_no_discordance():
    tie = mcnemar_test([True, False, True, False], [False, True, True, False])
    assert (tie["b_baseline_correct_proposed_wrong"], tie["c_baseline_wrong_proposed_correct"]) == (1, 1)
    assert tie["chi2"] == 0.0 and tie["p_value"] == 1.0
    none = mcnemar_test([True, False], [True, False])
    assert none["chi2"] == 0.0 and none["p_value"] == 1.0 and none["proposed_better"] is False


def test_mcnemar_length_mismatch():
    with pytest.raises(ValueError):
        mcnemar_test([True], [True, False])


# ----------------------------------------------------------------------------- Mann-Whitney


def test_mann_whitney_median_split_wiring():
    stars = list(range(1, 11))
    correct = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    result = mann_whitney_median_split(correct, stars, "stars")
    low, high = np.array(correct[:5]), np.array(correct[5:])
    expected = scipy_stats.mannwhitneyu(low, high, alternative="two-sided", method="asymptotic", use_continuity=True)
    assert result["median"] == 5.5
    assert (result["n_low"], result["n_high"]) == (5, 5)
    assert (result["accuracy_low"], result["accuracy_high"]) == (0.0, 1.0)
    assert result["u_low"] == 0.0 and result["u_high"] == 25.0  # U_low + U_high = n1 n2
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["status"] == "ok" and result["significant_at_0.05"] is True


def test_mann_whitney_thesis_formula_equals_u_high_for_the_low_group():
    outcomes = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    values = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    result = mann_whitney_median_split(outcomes, values, "x")
    low = [o for o, v in zip(outcomes, values) if v <= result["median"]]
    high = [o for o, v in zip(outcomes, values) if v > result["median"]]
    ranks = scipy_stats.rankdata(low + high)
    r1 = ranks[: len(low)].sum()
    thesis_u1 = len(low) * len(high) + len(low) * (len(low) + 1) / 2 - r1
    assert result["u_high"] == pytest.approx(thesis_u1)


def test_mann_whitney_skips_degenerate_inputs_and_missing_values():
    equal = mann_whitney_median_split([1, 0, 1, 0], [5, 5, 5, 5], "stars")
    assert equal["status"] == "skipped" and "empty" in equal["reason"]
    with_missing = mann_whitney_median_split([1, 0, 1, 0, 1, 1], [1, None, 3, 4, None, 6], "age")
    assert with_missing["n_used"] == 4 and with_missing["n_missing"] == 2
    all_correct = mann_whitney_median_split([1, 1, 1, 1], [1, 2, 3, 4], "x")
    assert all_correct["p_value"] == 1.0
