"""Extraction metrics (thesis 3.3.3) against hand-computed values."""

import pytest

from src.layer1_ai_extraction import metrics
from src.layer1_ai_extraction.schema import CommitmentExtractionOutput


def gold(text="We support 2.x and 3.x.", versions=("2.x", "3.x"), tau="explicit_range", time="release_time"):
    return {
        "commitment_text": text,
        "version_expression": list(versions),
        "support_status": tau,
        "temporal_reference": time,
    }


def test_tokenize_ignores_case_and_punctuation():
    assert metrics.tokenize("We support 2.x, and **3.x**!") == ["we", "support", "2", "x", "and", "3", "x"]


def test_token_prf_hand_computed():
    # gold: the cat sat (3 tokens); prediction: the cat ran away (4 tokens); overlap: the, cat (2)
    precision, recall, f1 = metrics.token_prf("The cat ran away", "the cat sat")
    assert precision == pytest.approx(2 / 4)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 * (2 / 4) * (2 / 3) / ((2 / 4) + (2 / 3)))


def test_token_prf_uses_multiset_overlap():
    # prediction repeats "a" three times, gold has it once: only one repetition counts
    precision, recall, _ = metrics.token_prf("a a a", "a b")
    assert precision == pytest.approx(1 / 3)
    assert recall == pytest.approx(1 / 2)


def test_token_prf_edge_cases():
    assert metrics.token_prf("", "") == (1.0, 1.0, 1.0)
    assert metrics.token_prf("abc", "") == (0.0, 0.0, 0.0)
    assert metrics.token_prf("", "abc") == (0.0, 0.0, 0.0)
    assert metrics.token_prf("abc", "xyz") == (0.0, 0.0, 0.0)


def test_set_prf_hand_computed_and_normalisation():
    # predicted {2.x, 4.x}, gold {2.x, 3.x}: one common
    precision, recall, f1 = metrics.set_prf(["2.x", "4.x"], ["2.x", "3.x"])
    assert (precision, recall, f1) == (0.5, 0.5, 0.5)
    assert metrics.set_prf(["V2.X", ">= 3.0"], ["2.x", ">=3.0"]) == (1.0, 1.0, 1.0)
    assert metrics.set_prf([], []) == (1.0, 1.0, 1.0)
    assert metrics.set_prf(["2.x"], []) == (0.0, 0.0, 0.0)


def test_label_match():
    assert metrics.label_match("release_time", "release_time") == 1.0
    assert metrics.label_match("release_time", "latest_only") == 0.0


def test_perfect_prediction_scores_one():
    scores = metrics.score_tuple(gold(), gold())
    assert scores["tuple_exact_match"] == 1.0
    assert scores["tuple_partial_match"] == 1.0
    assert all(value == 1.0 for value in scores.values())


def test_exact_match_requires_all_four_elements():
    for change in (
        dict(text="We support 2.x only."),
        dict(versions=("2.x",)),
        dict(tau="latest_only"),
        dict(time="vulnerability_publication_time"),
    ):
        assert metrics.tuple_exact_match(gold(**change), gold()) == 0.0


def test_exact_match_ignores_case_and_punctuation_of_t():
    assert metrics.tuple_exact_match(gold(text="we SUPPORT 2.x  and 3.x"), gold()) == 1.0


def test_partial_match_is_the_mean_of_the_four_element_scores():
    pred = gold(text="We support 2.x", versions=("2.x", "4.x"), tau="latest_only", time="release_time")
    scores = metrics.score_tuple(pred, gold())
    # t: pred tokens we support 2 x (4), gold tokens we support 2 x and 3 x (7) -> P=1, R=4/7, F1=8/11
    t_f1 = 8 / 11
    # v: {2.x, 4.x} vs {2.x, 3.x} -> F1 = 0.5; tau wrong -> 0; T right -> 1
    assert scores["t_f1"] == pytest.approx(t_f1)
    assert scores["v_f1"] == pytest.approx(0.5)
    assert scores["tuple_partial_match"] == pytest.approx((t_f1 + 0.5 + 0.0 + 1.0) / 4)
    assert scores["tuple_exact_match"] == 0.0


def test_score_tuple_accepts_extraction_output_objects():
    output = CommitmentExtractionOutput(**gold())
    assert metrics.score_tuple(output, gold())["tuple_exact_match"] == 1.0


def test_mean_scores_is_the_macro_average():
    rows = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.5}]
    assert metrics.mean_scores(rows) == {"a": 0.5, "b": 0.25}
    with pytest.raises(ValueError):
        metrics.mean_scores([])


def test_overlap_counts_support_micro_averaging():
    assert metrics.token_overlap_counts("a b c", "a b d e") == (2, 3, 4)
    assert metrics.prf_from_counts(2, 3, 4) == pytest.approx((2 / 3, 1 / 2, 2 * (2 / 3) * (1 / 2) / ((2 / 3) + (1 / 2))))
