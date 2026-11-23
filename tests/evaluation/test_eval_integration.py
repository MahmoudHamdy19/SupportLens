"""Consistency of this package with the Layer 1 metric helpers and the Layer 1/3 vocabularies (synthetic inputs)."""

import pytest

from src.data_pipeline.constants import TAU_VALUES, TEMPORAL_VALUES, VERDICTS
from src.evaluation.extraction_eval import REQUIRED_SCORE_KEYS, MetricsUnavailableError, aggregate_extraction, resolve_scorer

GOLD = {
    "commitment_text": "Only version 2.x receives security fixes.",
    "version_expression": ["2.x"],
    "support_status": "explicit_range",
    "temporal_reference": "vulnerability_publication_time",
}


@pytest.fixture(scope="module")
def layer1_scorer():
    try:
        return resolve_scorer()
    except MetricsUnavailableError as exc:  # Layer 1 not available in this checkout
        pytest.skip(str(exc))


def test_layer1_scorer_returns_every_required_key(layer1_scorer):
    scores = layer1_scorer(dict(GOLD), dict(GOLD))
    assert set(REQUIRED_SCORE_KEYS) <= set(scores)
    assert all(scores[k] == 1.0 for k in REQUIRED_SCORE_KEYS)
    wrong = dict(GOLD, support_status="latest_only", version_expression=["3.x"], commitment_text="Nothing is supported.")
    scores = layer1_scorer(wrong, dict(GOLD))
    assert scores["tau_match"] == 0.0 and scores["v_f1"] == 0.0 and scores["tuple_exact_match"] == 0.0
    assert scores["T_match"] == 1.0 and 0.0 < scores["tuple_partial_match"] < 1.0


def test_aggregation_averages_the_layer1_scores(layer1_scorer):
    half_wrong = dict(GOLD, support_status="latest_only")
    result = aggregate_extraction([dict(GOLD), half_wrong], [dict(GOLD), dict(GOLD)], layer1_scorer)
    assert result["n_records"] == 2
    assert result["tuple"]["exact_match"] == pytest.approx(0.5)
    assert result["tau"]["accuracy"] == pytest.approx(0.5)
    assert result["t"]["f1"] == 1.0 and result["v"]["f1"] == 1.0 and result["T"]["accuracy"] == 1.0
    assert result["tuple"]["partial_match"] == pytest.approx((1.0 + 0.75) / 2)


def test_aggregation_rejects_a_scorer_with_missing_keys():
    with pytest.raises(MetricsUnavailableError, match="returned no value"):
        aggregate_extraction([dict(GOLD)], [dict(GOLD)], lambda pred, gold: {"t_f1": 1.0})


def test_vocabularies_agree_with_layers_1_and_3():
    layer3 = pytest.importorskip("src.layer3_rule_assessment")
    assert tuple(layer3.VERDICT_CATEGORIES) == VERDICTS
    layer1 = pytest.importorskip("src.layer1_ai_extraction")
    assert tuple(layer1.SUPPORT_STATUSES) == TAU_VALUES
    assert tuple(layer1.TEMPORAL_REFERENCES) == TEMPORAL_VALUES
