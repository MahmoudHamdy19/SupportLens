"""Schema of the commitment tuple (INTERFACES.md section 1)."""

import pytest
from pydantic import ValidationError

from src.layer1_ai_extraction.schema import (
    COMMITMENT_KEYS,
    OUTPUT_KEYS,
    SUPPORT_STATUSES,
    TEMPORAL_REFERENCES,
    CommitmentExtractionOutput,
    compute_is_measurable,
)


def make(**overrides):
    values = dict(
        commitment_text="Only 2.x receives security fixes.",
        version_expression=["2.x"],
        support_status="explicit_range",
        temporal_reference="vulnerability_publication_time",
    )
    values.update(overrides)
    return CommitmentExtractionOutput(**values)


def test_enumerations_are_exactly_the_thesis_taxonomy():
    assert set(SUPPORT_STATUSES) == {"latest_only", "explicit_range", "all_versions", "ambiguous"}
    assert set(TEMPORAL_REFERENCES) == {"release_time", "vulnerability_publication_time"}


def test_to_dict_has_exactly_the_contract_keys_in_order():
    data = make().to_dict()
    assert tuple(data) == OUTPUT_KEYS
    assert set(COMMITMENT_KEYS) <= set(data)
    for forbidden in ("t", "tau", "v", "T", "SLA", "EOL", "end_of_life", "remediation_sla", "response_sla", "eol_policy"):
        assert forbidden not in data


@pytest.mark.parametrize("field,value", [("support_status", "supported"), ("temporal_reference", "end_of_life")])
def test_enums_are_strict(field, value):
    with pytest.raises(ValidationError):
        make(**{field: value})


@pytest.mark.parametrize("extra", ["t", "tau", "SLA", "eol_policy", "remediation_sla_days"])
def test_unknown_keys_are_rejected(extra):
    with pytest.raises(ValidationError):
        make(**{extra: "x"})


def test_version_expression_must_be_a_list_of_clean_tokens():
    with pytest.raises(ValidationError):
        make(version_expression="2.x")
    with pytest.raises(ValidationError):
        make(version_expression=["2.x", "  "])
    assert make(version_expression=[" 2.x "]).version_expression == ["2.x"]


@pytest.mark.parametrize(
    "status,versions,expected",
    [
        ("ambiguous", [], False),
        ("ambiguous", ["2.x"], False),
        ("explicit_range", [], False),
        ("explicit_range", ["2.x"], True),
        ("latest_only", ["latest"], True),
        ("latest_only", [], True),
        ("all_versions", [], True),
    ],
)
def test_is_measurable_is_derived_from_the_tuple(status, versions, expected):
    assert compute_is_measurable(status, versions) is expected
    assert make(support_status=status, version_expression=versions).to_dict()["is_measurable"] is expected


def test_is_measurable_cannot_be_set_by_the_caller():
    with pytest.raises(ValidationError):
        make(is_measurable=True)


def test_is_measurable_follows_assignment():
    output = make()
    assert output.is_measurable
    output.support_status = "ambiguous"
    assert not output.to_dict()["is_measurable"]


def test_provenance_defaults_do_not_claim_a_model():
    data = make().to_dict()
    assert data["lora_applied"] is False
    assert data["degraded"] is False
    assert data["actual_model_used"] == "none"
    assert "Qwen" not in data["extractor_engine"]


def test_round_trip_through_dict():
    original = make(extractor_engine="x", degraded=True)
    assert CommitmentExtractionOutput.from_dict(original.to_dict()) == original


def test_from_dict_rejects_inconsistent_derived_value():
    data = make().to_dict()
    data["is_measurable"] = not data["is_measurable"]
    with pytest.raises(ValueError):
        CommitmentExtractionOutput.from_dict(data)


def test_empty_tuple_is_ambiguous_and_unmeasurable():
    empty = CommitmentExtractionOutput.empty()
    assert empty.commitment_text == ""
    assert empty.support_status == "ambiguous"
    assert empty.version_expression == []
    assert empty.is_measurable is False


def test_dict_style_access():
    output = make()
    assert output["support_status"] == "explicit_range"
    assert output.get("missing", 7) == 7
