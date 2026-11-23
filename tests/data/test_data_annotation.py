"""Blind task files, annotator export parsing and validation (synthetic data only)."""

import json

import pytest

from src.data_pipeline import annotation as ann
from src.data_pipeline.cli import check_agreement_main, prepare_annotation_main
from src.data_pipeline.io_utils import write_jsonl
from supportlens_synthetic import commitment_sentence, dataset


def _sample(count=6):
    return dataset(count)


def _policies(records):
    return {r["package_name"]: r["policy_text"] for r in records}


def _simple_entry(record, verdict="Consistent", **overrides):
    n = int(record["package_name"].split("-")[1])
    commitment = {
        "commitment_text": commitment_sentence(n),
        "version_expression": ["2.x"],
        "support_status": "explicit_range",
        "temporal_reference": "vulnerability_publication_time",
    }
    commitment.update(overrides)
    return {"package_name": record["package_name"], "commitment": commitment, "verdict": verdict}


def _write(path, entries):
    path.write_text(json.dumps({"annotator": "x", "annotations": entries}), encoding="utf-8")
    return path


def test_tasks_are_blind_and_complete():
    records = _sample()
    tasks = ann.build_tasks(records)
    ann.assert_blind(tasks)
    assert {t["data"]["package_name"] for t in tasks} == {r["package_name"] for r in records}
    assert set(tasks[0]["data"]) == set(ann.ALLOWED_TASK_KEYS)
    for task in tasks:
        assert "annotations" not in task and "predictions" not in task


def test_assert_blind_rejects_extra_content():
    tasks = ann.build_tasks(_sample(2))
    tasks[0]["data"]["model_prediction"] = "Consistent"
    with pytest.raises(ann.AnnotationError):
        ann.assert_blind(tasks)
    tasks = ann.build_tasks(_sample(2))
    tasks[1]["predictions"] = []
    with pytest.raises(ann.AnnotationError):
        ann.assert_blind(tasks)


def test_shuffle_is_reproducible_independent_and_complete():
    tasks = ann.build_tasks(_sample(12))
    first = ann.shuffled_for_annotator(tasks, 1)
    again = ann.shuffled_for_annotator(tasks, 1)
    second = ann.shuffled_for_annotator(tasks, 2)

    def ids(items):
        return [t["data"]["package_name"] for t in items]

    assert ids(first) == ids(again)
    assert ids(first) != ids(second)
    assert sorted(ids(first)) == sorted(ids(second)) == sorted(ids(tasks))


def test_evidence_summary_lists_every_vulnerability_and_release():
    record = dataset(3)[2]  # synthpkg-03 has a vulnerability
    text = ann.format_evidence_summary(record)
    assert "SYN-003" in text and "2.0.1" in text and "RELEASES (2 in total" in text


def test_prepare_annotation_writes_three_files(tmp_path):
    sample = tmp_path / "sample.jsonl"
    write_jsonl(sample, _sample(5))
    assert prepare_annotation_main(["--sample", str(sample), "--out-dir", str(tmp_path / "ls")]) == 0
    names = sorted(p.name for p in (tmp_path / "ls").iterdir())
    assert names == ["tasks_all.json", "tasks_annotator1.json", "tasks_annotator2.json"]
    all_tasks = json.loads((tmp_path / "ls" / "tasks_all.json").read_text(encoding="utf-8"))
    first = json.loads((tmp_path / "ls" / "tasks_annotator1.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "ls" / "tasks_annotator2.json").read_text(encoding="utf-8"))
    assert len(all_tasks) == len(first) == len(second) == 5
    assert [t["data"]["package_name"] for t in first] != [t["data"]["package_name"] for t in second]
    assert all("id" not in t for t in first)


def test_simple_format_roundtrip(tmp_path):
    records = _sample(3)
    entries = [_simple_entry(r) for r in records]
    parsed = ann.parse_annotator_file(_write(tmp_path / "a.json", entries), _policies(records))
    assert set(parsed) == {r["package_name"] for r in records}
    assert parsed["synthpkg-01"].commitment["version_expression"] == ["2.x"]


@pytest.mark.parametrize(
    "override",
    [
        {"commitment_text": "A sentence that is not in the policy."},
        {"version_expression": ["9.9"]},
        {"support_status": "supported"},
        {"temporal_reference": "end_of_life"},
        {"commitment_text": ""},  # empty t only allowed for 'ambiguous'
    ],
)
def test_validation_rejects_codebook_violations(tmp_path, override):
    records = _sample(1)
    path = _write(tmp_path / "a.json", [_simple_entry(records[0], **override)])
    with pytest.raises(ann.AnnotationError):
        ann.parse_annotator_file(path, _policies(records))


def test_invalid_verdict_and_unknown_package_rejected(tmp_path):
    records = _sample(1)
    with pytest.raises(ann.AnnotationError):
        ann.parse_annotator_file(_write(tmp_path / "a.json", [_simple_entry(records[0], verdict="Maybe")]), _policies(records))
    entry = _simple_entry(records[0])
    entry["package_name"] = "not-in-sample"
    with pytest.raises(ann.AnnotationError):
        ann.parse_annotator_file(_write(tmp_path / "b.json", [entry]), _policies(records))


def test_empty_commitment_text_allowed_for_ambiguous(tmp_path):
    records = _sample(1)
    entry = _simple_entry(
        records[0], verdict="Unmeasurable Commitment", commitment_text="", version_expression=[], support_status="ambiguous"
    )
    parsed = ann.parse_annotator_file(_write(tmp_path / "a.json", [entry]), _policies(records))
    assert parsed[records[0]["package_name"]].commitment["commitment_text"] == ""


def test_require_complete_reports_missing_records(tmp_path):
    records = _sample(3)
    parsed = ann.parse_annotator_file(_write(tmp_path / "a.json", [_simple_entry(records[0])]), _policies(records))
    with pytest.raises(ann.AnnotationError, match="no annotation"):
        ann.require_complete(parsed, _policies(records), "annotator1")


def _label_studio_item(record, *, t=None, versions=(), tau="explicit_range", temporal="vulnerability_publication_time", verdict="Consistent"):
    result = []
    if t is not None:
        result.append({"from_name": "spans", "to_name": "policy", "type": "labels",
                       "value": {"start": 0, "end": 1, "text": t, "labels": ["CommitmentText"]}})
    for position, token in enumerate(versions):
        result.append({"from_name": "spans", "to_name": "policy", "type": "labels",
                       "value": {"start": 100 + position, "end": 101 + position, "text": token, "labels": ["VersionExpression"]}})
    result.append({"from_name": "support_status", "to_name": "policy", "type": "choices", "value": {"choices": [tau]}})
    result.append({"from_name": "temporal_reference", "to_name": "policy", "type": "choices", "value": {"choices": [temporal]}})
    result.append({"from_name": "expected_verdict", "to_name": "policy", "type": "choices", "value": {"choices": [verdict]}})
    result.append({"from_name": "notes", "to_name": "policy", "type": "textarea", "value": {"text": ["synthetic note"]}})
    return {"data": {"package_name": record["package_name"]}, "annotations": [{"was_cancelled": False, "result": result}]}


def test_label_studio_export_is_parsed(tmp_path):
    records = _sample(2)
    export = [
        _label_studio_item(records[0], t=commitment_sentence(1), versions=["2.x"]),
        {"data": {"package_name": records[1]["package_name"]}, "annotations": []},  # skipped task
    ]
    path = tmp_path / "ls.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    parsed = ann.parse_annotator_file(path, _policies(records))
    assert list(parsed) == [records[0]["package_name"]]
    assert parsed[records[0]["package_name"]].commitment["commitment_text"] == commitment_sentence(1)
    assert parsed[records[0]["package_name"]].notes == "synthetic note"


def test_label_studio_rejects_two_commitment_spans_and_missing_choice(tmp_path):
    records = _sample(1)
    item = _label_studio_item(records[0], t=commitment_sentence(1))
    item["annotations"][0]["result"].append(
        {"from_name": "spans", "type": "labels", "value": {"start": 5, "end": 9, "text": "Only", "labels": ["CommitmentText"]}}
    )
    path = tmp_path / "ls.json"
    path.write_text(json.dumps([item]), encoding="utf-8")
    with pytest.raises(ann.AnnotationError, match="contiguous"):
        ann.parse_annotator_file(path, _policies(records))

    item = _label_studio_item(records[0], t=commitment_sentence(1))
    item["annotations"][0]["result"] = [r for r in item["annotations"][0]["result"] if r["from_name"] != "expected_verdict"]
    path.write_text(json.dumps([item]), encoding="utf-8")
    with pytest.raises(ann.AnnotationError, match="expected_verdict"):
        ann.parse_annotator_file(path, _policies(records))


def test_check_agreement_cli_end_to_end_and_identical_files_refused(tmp_path):
    records = _sample(6)
    sample = tmp_path / "sample.jsonl"
    write_jsonl(sample, records)
    verdicts1 = ["Consistent", "Consistent", "Insufficient Evidence", "Insufficient Evidence", "Consistent", "Insufficient Evidence"]
    verdicts2 = ["Consistent", "Consistent", "Insufficient Evidence", "Consistent", "Consistent", "Insufficient Evidence"]
    a1 = _write(tmp_path / "a1.json", [_simple_entry(r, v) for r, v in zip(records, verdicts1)])
    a2 = _write(tmp_path / "a2.json", [_simple_entry(r, v) for r, v in zip(records, verdicts2)])
    out = tmp_path / "report.json"
    assert check_agreement_main(["--sample", str(sample), "--annotator1", str(a1), "--annotator2", str(a2), "--output", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n_records"] == 6 and report["n_disagreements"] == 1
    # po = 5/6; marginals: first C3/I3, second C4/I2 -> pe = 0.5*(4/6) + 0.5*(2/6) = 0.5
    assert report["cohens_kappa"] == pytest.approx((5 / 6 - 0.5) / 0.5)
    assert report["inputs"]["annotator1_sha256"] != report["inputs"]["annotator2_sha256"]

    clone = tmp_path / "clone.json"
    clone.write_text(a1.read_text(encoding="utf-8"), encoding="utf-8")
    code = check_agreement_main(["--sample", str(sample), "--annotator1", str(a1), "--annotator2", str(clone), "--output", str(tmp_path / "x.json")])
    assert code == 2
    assert not (tmp_path / "x.json").exists()


def _timed_export(path, lead_times, created):
    items = [
        {"data": {"package_name": f"pkg{i}"}, "annotations": [{"lead_time": lt, "created_at": ts, "result": []}]}
        for i, (lt, ts) in enumerate(zip(lead_times, created))
    ]
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def test_script_generated_label_studio_exports_are_flagged(tmp_path):
    burst = [f"2026-11-19T15:42:09.{i:06d}+00:00" for i in range(12)]
    scripted = _timed_export(tmp_path / "scripted.json", [180.0] * 12, burst)
    warnings = ann.export_authenticity_warnings(scripted)
    assert any("same lead_time" in w for w in warnings)
    assert any("within one minute" in w for w in warnings)

    hours = [f"2026-11-19T{10 + i // 4:02d}:{(i % 4) * 15:02d}:00+00:00" for i in range(12)]
    human = _timed_export(tmp_path / "human.json", [150.0 + 37 * i for i in range(12)], hours)
    assert ann.export_authenticity_warnings(human) == []
    assert ann.export_authenticity_warnings(tmp_path / "human.json", min_records=100) == []
