"""Adjudication refuses to produce output while anything is unresolved (synthetic data only)."""

import json

import pytest

from src.data_pipeline.adjudication import (
    AdjudicationError,
    adjudication_template,
    build_ground_truth,
    load_adjudication,
    validate_ground_truth_records,
)
from src.data_pipeline.annotation import Annotation
from src.data_pipeline.cli import adjudicate_main
from src.data_pipeline.io_utils import DataError, read_jsonl, write_jsonl
from supportlens_synthetic import commitment_sentence, dataset


def _annotation(record, verdict="Consistent", **commitment_overrides):
    n = int(record["package_name"].split("-")[1])
    commitment = {
        "commitment_text": commitment_sentence(n),
        "version_expression": ["2.x"],
        "support_status": "explicit_range",
        "temporal_reference": "vulnerability_publication_time",
    }
    commitment.update(commitment_overrides)
    return Annotation(record["package_name"], commitment, verdict)


@pytest.fixture
def setup():
    sample = dataset(4)
    first = {r["package_name"]: _annotation(r) for r in sample}
    second = {r["package_name"]: _annotation(r) for r in sample}
    # two disagreements: verdict of synthpkg-02, tau and verdict of synthpkg-04
    second["synthpkg-02"] = _annotation(sample[1], "Insufficient Evidence")
    second["synthpkg-04"] = _annotation(sample[3], "Unmeasurable Commitment", support_status="ambiguous")
    return sample, first, second


def test_no_disagreement_needs_no_adjudication(setup):
    sample, first, _ = setup
    records = build_ground_truth(sample, first, dict(first), "", {})
    assert len(records) == 4
    assert all(r["annotation"]["adjudicated_by"] is None for r in records)
    validate_ground_truth_records(records)


def test_unresolved_disagreement_is_refused(setup):
    sample, first, second = setup
    with pytest.raises(AdjudicationError, match="unresolved"):
        build_ground_truth(sample, first, second, "supervisor", {})
    # one record decided, the other not
    decisions = {"synthpkg-02": {"fields": {"verdict": "annotator1"}, "rationale": ""}}
    with pytest.raises(AdjudicationError, match="synthpkg-04"):
        build_ground_truth(sample, first, second, "supervisor", decisions)


def test_partially_resolved_record_is_refused(setup):
    sample, first, second = setup
    decisions = {
        "synthpkg-02": {"fields": {"verdict": "annotator1"}, "rationale": ""},
        "synthpkg-04": {"fields": {"verdict": "annotator2"}, "rationale": ""},  # tau still undecided
    }
    with pytest.raises(AdjudicationError, match="unresolved fields"):
        build_ground_truth(sample, first, second, "supervisor", decisions)


def test_missing_record_is_refused(setup):
    sample, first, second = setup
    del second["synthpkg-01"]
    with pytest.raises(AdjudicationError, match="no annotation"):
        build_ground_truth(sample, first, second, "supervisor", {})


def test_decisions_outside_the_disagreements_are_refused(setup):
    sample, first, second = setup
    full = {
        "synthpkg-02": {"fields": {"verdict": "annotator1"}, "rationale": ""},
        "synthpkg-04": {"fields": {"verdict": "annotator1", "support_status": "annotator1"}, "rationale": ""},
    }
    with pytest.raises(AdjudicationError, match="without a disagreement"):
        build_ground_truth(sample, first, second, "supervisor", {**full, "synthpkg-01": {"fields": {"verdict": "annotator1"}, "rationale": ""}})
    extra = {**full, "synthpkg-02": {"fields": {"verdict": "annotator1", "support_status": "annotator1"}, "rationale": ""}}
    with pytest.raises(AdjudicationError, match="without a disagreement"):
        build_ground_truth(sample, first, second, "supervisor", extra)


def test_resolved_disagreements_produce_ground_truth_with_provenance(setup):
    sample, first, second = setup
    decisions = {
        "synthpkg-02": {"fields": {"verdict": "annotator2"}, "rationale": "no vulnerability affects 2.x"},
        "synthpkg-04": {"fields": {"verdict": {"value": "Consistent"}, "support_status": "annotator1"}, "rationale": ""},
    }
    records = {r["package_name"]: r for r in build_ground_truth(sample, first, second, "supervisor", decisions)}
    assert records["synthpkg-02"]["verdict"] == "Insufficient Evidence"
    assert records["synthpkg-04"]["verdict"] == "Consistent"
    assert records["synthpkg-04"]["commitment"]["support_status"] == "explicit_range"
    assert records["synthpkg-02"]["annotation"]["adjudicated_by"] == "supervisor"
    assert records["synthpkg-02"]["annotation"]["adjudicated_fields"] == ["verdict"]
    assert records["synthpkg-01"]["annotation"]["adjudicated_by"] is None
    # the annotators' original labels stay in the record
    assert records["synthpkg-02"]["annotation"]["annotator2"]["verdict"] == "Insufficient Evidence"
    assert records["synthpkg-02"]["annotation"]["annotator1"]["verdict"] == "Consistent"
    # INTERFACES.md section 5 keys, plus pass-through provenance
    for key in ("package_name", "github_repo", "policy_text", "commitment", "verdict", "releases", "vulnerabilities",
                "stars", "repo_created_at", "policy_word_count", "annotation"):
        assert key in records["synthpkg-01"]
    validate_ground_truth_records(list(records.values()))


def test_adjudicated_value_must_satisfy_the_codebook(setup):
    sample, first, second = setup
    decisions = {
        "synthpkg-02": {"fields": {"verdict": {"value": "Maybe"}}, "rationale": ""},
        "synthpkg-04": {"fields": {"verdict": "annotator1", "support_status": "annotator1"}, "rationale": ""},
    }
    with pytest.raises(DataError):
        build_ground_truth(sample, first, second, "supervisor", decisions)


def test_provenance_check_rejects_undecided_disagreement(setup):
    sample, first, second = setup
    records = build_ground_truth(sample, first, dict(first), "", {})
    records[0]["annotation"]["annotator2"] = {"commitment": records[0]["commitment"], "verdict": "Insufficient Evidence"}
    with pytest.raises(DataError, match="no adjudicator"):
        validate_ground_truth_records(records)


def test_template_lists_disagreements_without_deciding(setup):
    _, first, second = setup
    template = adjudication_template(first, second)
    assert {d["package_name"] for d in template["decisions"]} == {"synthpkg-02", "synthpkg-04"}
    assert all(d["fields"] == {} for d in template["decisions"])
    assert template["adjudicator"] == ""


def test_load_adjudication_requires_a_named_adjudicator(tmp_path):
    path = tmp_path / "adj.json"
    path.write_text(json.dumps({"adjudicator": "", "decisions": []}), encoding="utf-8")
    with pytest.raises(AdjudicationError):
        load_adjudication(path)


def _write_export(path, annotations):
    path.write_text(
        json.dumps(
            {
                "annotations": [
                    {"package_name": a.package_name, "commitment": a.commitment, "verdict": a.verdict} for a in annotations.values()
                ]
            }
        ),
        encoding="utf-8",
    )


def test_adjudicate_cli_refuses_then_succeeds(tmp_path, setup):
    sample, first, second = setup
    sample_path = tmp_path / "sample.jsonl"
    write_jsonl(sample_path, sample)
    a1, a2 = tmp_path / "a1.json", tmp_path / "a2.json"
    _write_export(a1, first)
    _write_export(a2, second)
    output = tmp_path / "gt" / "ground_truth.jsonl"
    base = ["--sample", str(sample_path), "--annotator1", str(a1), "--annotator2", str(a2), "--output", str(output)]

    assert adjudicate_main(base) == 3  # disagreements and no adjudication file
    assert not output.exists()

    template = tmp_path / "template.json"
    assert adjudicate_main(base[:6] + ["--template", str(template)]) == 0
    assert not output.exists()

    adjudication = tmp_path / "adj.json"
    adjudication.write_text(
        json.dumps(
            {
                "adjudicator": "supervisor",
                "decisions": [
                    {"package_name": "synthpkg-02", "fields": {"verdict": "annotator1"}, "rationale": "r"},
                    {"package_name": "synthpkg-04", "fields": {"verdict": "annotator1", "support_status": "annotator1"}, "rationale": "r"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert adjudicate_main(base + ["--adjudication", str(adjudication)]) == 0
    records = read_jsonl(output)
    assert len(records) == 4 and {r["verdict"] for r in records} == {"Consistent"}
    assert (tmp_path / "gt" / "ground_truth_meta.json").is_file()


def test_ground_truth_carries_temporal_expression(setup):
    sample, first, second = setup
    phrase = "Older releases"
    assert phrase in sample[0]["policy_text"]
    first["synthpkg-01"] = _annotation(sample[0], temporal_expression=phrase)
    second["synthpkg-01"] = _annotation(sample[0], temporal_expression=phrase)
    second["synthpkg-03"] = _annotation(sample[2], temporal_expression="Only version")
    decisions = {
        "synthpkg-02": {"fields": {"verdict": "annotator1"}, "rationale": ""},
        "synthpkg-04": {"fields": {"verdict": "annotator1", "support_status": "annotator1"}, "rationale": ""},
        "synthpkg-03": {"fields": {"temporal_expression": {"value": None}}, "rationale": "no temporal phrase"},
    }
    records = {r["package_name"]: r for r in build_ground_truth(sample, first, second, "supervisor", decisions)}
    assert records["synthpkg-01"]["commitment"]["temporal_expression"] == phrase
    assert records["synthpkg-02"]["commitment"]["temporal_expression"] is None
    assert records["synthpkg-03"]["commitment"]["temporal_expression"] is None
    with pytest.raises(DataError, match="verbatim"):
        build_ground_truth(sample, first, second, "supervisor", {**decisions, "synthpkg-03": {"fields": {"temporal_expression": {"value": "not in policy"}}, "rationale": ""}})
