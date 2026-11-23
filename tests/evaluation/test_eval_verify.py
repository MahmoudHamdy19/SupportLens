"""verify_results catches tampered numbers; render fills README markers (synthetic data only)."""

import json
from pathlib import Path

import pytest

from src.data_pipeline.agreement import build_agreement_report
from src.data_pipeline.annotation import Annotation
from src.data_pipeline.constants import TAU_VALUES, TEMPORAL_VALUES, VERDICTS
from src.data_pipeline.io_utils import read_jsonl, write_jsonl
from src.data_pipeline.splits import stratified_split
from src.evaluation.cli import render_main, verify_main
from src.evaluation.repro import build_manifest
from src.evaluation.runner import run_evaluation
from src.evaluation.verify import MARKER, compare, format_value, lookup, render_markers, verify
from supportlens_synthetic import StubEngine, StubExtractor, gold_record, stub_scorer

C, P, PI, IE, U = VERDICTS
GOLD = [C, C, IE, IE, IE, PI, P, U, C, IE]


def _engine():
    gold = {f"synthpkg-{n:02d}": v for n, v in enumerate(GOLD, start=1)}
    rules = {"weak text": IE, "tuned text": "oracle"}
    return StubEngine(lambda c, p: gold[p] if rules[c["commitment_text"]] == "oracle" else rules[c["commitment_text"]])


def _commitment(text):
    return {"commitment_text": text, "version_expression": ["2.x"], "support_status": "explicit_range", "temporal_reference": "release_time"}


@pytest.fixture
def run(tmp_path):
    """A complete synthetic evaluation: test file, results and manifest in a temporary folder."""
    gt = tmp_path / "ground_truth"
    out = tmp_path / "evaluation"
    records = [gold_record(n, v, stars=n * 7) for n, v in enumerate(GOLD, start=1)]
    write_jsonl(gt / "test.jsonl", records)
    factories = {
        "regex_baseline": lambda: StubExtractor(commitment=_commitment("weak text")),
        "qwen_finetuned_lora": lambda: StubExtractor(lora=True, commitment=_commitment("tuned text")),
    }
    run_evaluation(records, factories, _engine(), stub_scorer, out, probe_gpu=False, n_resamples=1000,
                   data_files={"test": gt / "test.jsonl"})
    return {"gt": gt, "out": out, "tmp": tmp_path, "records": records}


def _verify(run, markdown=()):
    return verify(eval_dir=run["out"], ground_truth_dir=run["gt"], annotations_dir=run["tmp"] / "annotations",
                  scorer=stub_scorer, markdown_files=markdown)


def _edit_json(path, edit):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    edit(data)
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def test_untouched_results_verify(run):
    report = _verify(run)
    assert report.ok, report.problems
    assert any("verdict_metrics.json recomputed" in c for c in report.checked)
    assert any("hash of predictions" in c for c in report.checked)


def test_tampered_headline_number_is_caught(run):
    _edit_json(run["out"] / "verdict_metrics.json", lambda d: d["systems"]["regex_baseline"].update(macro_f1=0.99))
    report = _verify(run)
    assert not report.ok
    assert any("verdict_metrics.systems.regex_baseline.macro_f1" in p for p in report.problems)


@pytest.mark.parametrize(
    "document, edit",
    [
        ("extraction_metrics.json", lambda d: d["systems"]["qwen_finetuned_lora"]["tuple"].update(exact_match=0.5)),
        ("statistical_tests.json", lambda d: d["mcnemar"]["baselines"]["regex_baseline"].update(p_value=0.001)),
        ("statistical_tests.json", lambda d: d["bootstrap_macro_f1"]["systems"]["regex_baseline"].update(ci_low=0.9)),
        ("statistical_tests.json", lambda d: d["bootstrap_extraction"]["systems"]["regex_baseline"]["t_f1"].update(ci_high=0.5)),
        ("statistical_tests.json", lambda d: d["mcnemar_tuple_exact_match"]["baselines"]["regex_baseline"].update(chi2=9.0)),
        ("statistical_tests.json", lambda d: d["mann_whitney_activity"]["systems"]["regex_baseline"].update(p_value=0.01)),
        ("failure_analysis.json", lambda d: d["systems"]["regex_baseline"]["wrong_predictions"].update(n=0)),
        ("failure_analysis.json", lambda d: d.update(activity_definition="anything")),
    ],
)
def test_tampering_with_other_documents_is_caught(run, document, edit):
    _edit_json(run["out"] / document, edit)
    assert not _verify(run).ok


def test_missing_failure_analysis_is_reported(run):
    (run["out"] / "failure_analysis.json").unlink()
    assert "failure_analysis.json is missing" in _verify(run).problems


def test_tampered_predictions_are_caught_by_hash_and_recomputation(run):
    rows = read_jsonl(run["out"] / "predictions.jsonl")
    rows[0]["verdict"] = U if rows[0]["verdict"] != U else C
    write_jsonl(run["out"] / "predictions.jsonl", rows)
    problems = " ".join(_verify(run).problems)
    assert "hash mismatch" in problems and "verdict_metrics" in problems


def test_changed_ground_truth_is_caught(run):
    records = read_jsonl(run["gt"] / "test.jsonl")
    records[0]["verdict"] = U
    write_jsonl(run["gt"] / "test.jsonl", records)
    problems = " ".join(_verify(run).problems)
    assert "test was changed" in problems


def test_smoke_or_non_strict_runs_are_not_verifiable(run):
    _edit_json(run["out"] / "run_manifest.json", lambda d: d["run"].update(limit=5))
    assert any("smoke test" in p for p in _verify(run).problems)
    _edit_json(run["out"] / "run_manifest.json", lambda d: d["run"].update(limit=None, strict=False))
    assert any("not strict" in p for p in _verify(run).problems)


def test_missing_predictions_fail(run):
    (run["out"] / "predictions.jsonl").unlink()
    assert not _verify(run).ok


# ----------------------------------------------------------------------------- markdown markers


def test_markers_are_checked_against_the_recomputed_value(run, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "| System | Macro-F1 |\n|---|---|\n"
        "| Tuned | <!-- RESULT:verdict_metrics.systems.qwen_finetuned_lora.macro_f1 -->1.0000<!-- /RESULT --> |\n"
        "| Regex | <!-- RESULT:verdict_metrics.systems.regex_baseline.accuracy|.2f -->0.40<!-- /RESULT --> |\n",
        encoding="utf-8",
    )
    assert _verify(run, [readme]).ok
    readme.write_text(readme.read_text(encoding="utf-8").replace("1.0000", "0.9000"), encoding="utf-8")
    report = _verify(run, [readme])
    assert not report.ok and any("shows '0.9000'" in p for p in report.problems)


def test_marker_with_unknown_key_fails(run, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("<!-- RESULT:verdict_metrics.systems.nope.macro_f1 -->0.5<!-- /RESULT -->", encoding="utf-8")
    assert any("does not exist" in p for p in _verify(run, [readme]).problems)


def test_render_fills_markers_and_verify_then_passes(run, tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluation.cli.resolve_scorer", lambda: stub_scorer)
    readme = tmp_path / "README.md"
    readme.write_text(
        "Macro-F1: <!-- RESULT:verdict_metrics.systems.qwen_finetuned_lora.macro_f1 -->TBD<!-- /RESULT -->; "
        "b: <!-- RESULT:statistical_tests.mcnemar.baselines.regex_baseline.b_baseline_correct_proposed_wrong -->?<!-- /RESULT -->; "
        "n: <!-- RESULT:verdict_metrics.n_records|d -->?<!-- /RESULT -->\n",
        encoding="utf-8",
    )
    args = ["--eval-dir", str(run["out"]), "--ground-truth-dir", str(run["gt"]), "--annotations-dir", str(run["tmp"] / "annotations"),
            "--markdown", str(readme)]
    assert render_main(args) == 0
    text = readme.read_text(encoding="utf-8")
    assert "Macro-F1: <!-- RESULT:verdict_metrics.systems.qwen_finetuned_lora.macro_f1 -->1.0000<!-- /RESULT -->" in text
    assert ">10<!--" in text
    assert _verify(run, [readme]).ok
    assert verify_main(args) == 0
    readme.write_text(text.replace("1.0000", "0.5000"), encoding="utf-8")
    assert verify_main(args) == 1


def test_render_refuses_unknown_keys_and_writes_nothing(run, tmp_path):
    readme = tmp_path / "README.md"
    original = "<!-- RESULT:verdict_metrics.systems.qwen_finetuned_lora.macro_f1 -->TBD<!-- /RESULT --> <!-- RESULT:nope.key -->x<!-- /RESULT -->"
    readme.write_text(original, encoding="utf-8")
    args = ["--eval-dir", str(run["out"]), "--ground-truth-dir", str(run["gt"]), "--annotations-dir", str(run["tmp"] / "annotations"),
            "--markdown", str(readme)]
    assert render_main(args) == 2
    assert readme.read_text(encoding="utf-8") == original


def test_marker_helpers():
    assert lookup({"a": {"b": [1, {"c": 2}]}}, "a.b.1.c") == 2
    with pytest.raises(KeyError):
        lookup({"a": 1}, "a.b")
    assert format_value(0.123456) == "0.1235" and format_value(3) == "3" and format_value(None) == "n/a"
    assert format_value(0.5, ".1%") == "50.0%" and format_value(True) == "yes"
    text, problems = render_markers("<!--RESULT:x.y|.2f-->old<!--/RESULT-->", {"x": {"y": 1.23456}})
    assert text == "<!-- RESULT:x.y|.2f -->1.23<!-- /RESULT -->" and not problems
    assert len(MARKER.findall(text)) == 1


def test_compare_flags_every_kind_of_difference():
    assert compare({"a": 1.0, "b": [1, 2]}, {"a": 1.0 + 1e-12, "b": [1, 2]}) == []
    assert compare({"a": 1.0}, {"a": 1.1}) and compare({"a": 1}, {}) and compare({}, {"a": 1})
    assert compare({"a": [1, 2]}, {"a": [1]}) and compare({"a": True}, {"a": 1})


# ----------------------------------------------------------------------------- agreement and splits


def _ground_truth_with_two_annotators():
    records = []
    for n, verdict in enumerate([C, C, IE, IE, IE, PI, P, U, C, IE, IE, C], start=1):
        record = gold_record(n, verdict)
        if n % 4 == 0:  # the second annotator differs; the supervisor sided with the first
            other = dict(record["annotation"]["annotator1"], verdict=C if verdict != C else IE)
            record["annotation"]["annotator2"] = other
            record["annotation"]["adjudicated_by"] = "supervisor"
        records.append(record)
    return records


def test_agreement_and_split_reports_are_recomputed_from_the_data(run):
    records = _ground_truth_with_two_annotators()
    gt = run["gt"]
    write_jsonl(gt / "ground_truth.jsonl", records)
    splits, split_report = stratified_split(records)
    for name, items in splits.items():
        write_jsonl(gt / f"{name}.jsonl", items)
    # the evaluation fixture used a different test.jsonl; restore consistent evaluation inputs
    first = {r["package_name"]: Annotation(r["package_name"], r["annotation"]["annotator1"]["commitment"], r["annotation"]["annotator1"]["verdict"]) for r in records}
    second = {r["package_name"]: Annotation(r["package_name"], r["annotation"]["annotator2"]["commitment"], r["annotation"]["annotator2"]["verdict"]) for r in records}
    agreement = build_agreement_report(first, second, tau_values=TAU_VALUES, temporal_values=TEMPORAL_VALUES)
    annotations = run["tmp"] / "annotations"
    annotations.mkdir()
    (annotations / "agreement_report.json").write_text(json.dumps(agreement), encoding="utf-8")
    (gt / "split_report.json").write_text(json.dumps(split_report), encoding="utf-8")

    report = _verify_without_evaluation(run)
    assert not any("agreement_report" in p or "split_report" in p for p in report.problems), report.problems
    assert any("agreement_report.json recomputed" in c for c in report.checked)
    assert any("split_report.json recomputed" in c for c in report.checked)

    _edit_json(annotations / "agreement_report.json", lambda d: d.update(cohens_kappa=0.95))
    assert any("agreement_report.cohens_kappa" in p for p in _verify_without_evaluation(run).problems)
    _edit_json(gt / "split_report.json", lambda d: d["split_sizes"].update(test=99))
    assert any("split_report.split_sizes.test" in p for p in _verify_without_evaluation(run).problems)


def _verify_without_evaluation(run):
    return verify(eval_dir=run["tmp"] / "no_evaluation", ground_truth_dir=run["gt"], annotations_dir=run["tmp"] / "annotations",
                  scorer=stub_scorer, test_file=run["gt"] / "test.jsonl")


# ----------------------------------------------------------------------------- manifest


def test_manifest_records_environment_hashes_and_hyperparameters(tmp_path):
    models = tmp_path / "models" / "qwen_lora"
    models.mkdir(parents=True)
    (models / "run_metadata.json").write_text(json.dumps({"learning_rate": 2e-4, "lora_rank": 16}), encoding="utf-8")
    data = tmp_path / "test.jsonl"
    data.write_text('{"a": 1}\n', encoding="utf-8")
    manifest = build_manifest(data_files={"test": data, "absent": tmp_path / "nope.jsonl"}, models_dir=tmp_path / "models",
                              probe_gpu=False, run={"strict": True})
    assert manifest["seed"] == 42 and manifest["python"]["version"] and manifest["platform"]["system"]
    assert set(manifest["libraries"]) >= {"numpy", "scipy", "torch", "transformers", "peft", "pydantic"}
    assert manifest["libraries"]["numpy"]
    assert manifest["data_files"]["test"]["sha256"] and "absent" not in manifest["data_files"]
    assert manifest["hyperparameters"]["qwen_lora"] == {"learning_rate": 2e-4, "lora_rank": 16}
    assert manifest["created_at_utc"].endswith("+00:00") and manifest["run"] == {"strict": True}
    assert manifest["gpu"]["available"] is None  # not probed in this test
    json.dumps(manifest)


def test_gpu_info_with_a_fake_cuda_torch(monkeypatch):
    import sys
    import types

    from src.evaluation.repro import gpu_info

    fake = types.SimpleNamespace(
        __version__="0.0-fake",
        version=types.SimpleNamespace(cuda="12.x"),
        cuda=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: "Synthetic GPU", device_count=lambda: 1),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert gpu_info() == {"available": True, "torch": "0.0-fake", "cuda_version": "12.x", "device_name": "Synthetic GPU", "device_count": 1}


def test_gpu_info_without_torch_is_an_explicit_negative(monkeypatch):
    import sys

    from src.evaluation.repro import gpu_info

    monkeypatch.setitem(sys.modules, "torch", None)  # makes "import torch" raise ImportError
    assert gpu_info() == {"available": False, "torch": None, "cuda_version": None, "device_name": None}
