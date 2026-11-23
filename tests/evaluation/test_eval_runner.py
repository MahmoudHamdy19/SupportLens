"""Evaluation runner with stub extractors and a stub engine (offline, synthetic data only)."""

import json
import math

import pytest

from src.data_pipeline.constants import VERDICTS
from src.data_pipeline.io_utils import read_jsonl, write_jsonl
from src.evaluation.cli import run_evaluation_main
from src.evaluation.runner import EvaluationAborted, compute_results, run_evaluation, run_systems
from supportlens_synthetic import ModelUnavailableError, StubEngine, StubExplainer, StubExtractor, gold_record, stub_scorer

C, P, PI, IE, U = VERDICTS
GOLD_VERDICTS = [C, C, IE, IE, IE, PI, P, U]
FILES = ("predictions.jsonl", "extraction_metrics.json", "verdict_metrics.json", "statistical_tests.json", "failure_analysis.json",
         "run_manifest.json")


def records():
    return [gold_record(n, verdict, stars=n * 10) for n, verdict in enumerate(GOLD_VERDICTS, start=1)]


def gold_verdicts_by_package():
    return {r["package_name"]: r["verdict"] for r in records()}


def engine_with(texts_to_verdict):
    """Engine whose answer depends on the PREDICTED commitment text; the gold text is never mapped."""
    gold = gold_verdicts_by_package()

    def verdict_for(commitment, package):
        rule = texts_to_verdict[commitment["commitment_text"]]
        return gold[package] if rule == "oracle" else rule

    return StubEngine(verdict_for)


def factories(**overrides):
    base = {
        "regex_baseline": lambda: StubExtractor(commitment=_commitment("regex text"), engine="regex"),
        "roberta_finetuned": lambda: StubExtractor(commitment=_commitment("roberta text"), engine="roberta"),
        "qwen_zero_shot": lambda: StubExtractor(lora=False, commitment=_commitment("zero-shot text"), engine="qwen"),
        "qwen_finetuned_lora": lambda: StubExtractor(lora=True, commitment=_commitment("tuned text"), engine="qwen+lora"),
    }
    base.update(overrides)
    return base


def _commitment(text):
    return {"commitment_text": text, "version_expression": ["3.x"], "support_status": "explicit_range", "temporal_reference": "release_time"}


ENGINE_RULES = {"regex text": IE, "roberta text": IE, "zero-shot text": C, "tuned text": "oracle"}


def test_predicted_commitments_not_gold_reach_the_engine(tmp_path):
    engine = engine_with(ENGINE_RULES)
    data = records()
    run_evaluation(data, factories(), engine, stub_scorer, tmp_path, probe_gpu=False, n_resamples=50)
    gold_texts = {r["commitment"]["commitment_text"] for r in data}
    received = {c["commitment_text"] for c in engine.received}
    assert received == {"regex text", "roberta text", "zero-shot text", "tuned text"}
    assert not received & gold_texts
    assert len(engine.received) == 4 * len(data)
    # the record's own releases and vulnerabilities are handed over, plus a fixed evidence time
    by_package = {r["package_name"]: r for r in data}
    for kwargs in engine.kwargs:
        record = by_package[kwargs["package_name"]]
        assert kwargs["releases"] == record["releases"] and kwargs["vulnerabilities"] == record["vulnerabilities"]
        assert kwargs["reference_time"].isoformat() == "2026-06-01T00:00:00+00:00"


def test_results_match_hand_computation(tmp_path):
    data = records()
    result = run_evaluation(data, factories(), engine_with(ENGINE_RULES), stub_scorer, tmp_path, probe_gpu=False, n_resamples=200)
    docs = result["docs"]
    regex = docs["verdict_metrics"]["systems"]["regex_baseline"]
    # only Insufficient Evidence is predicted: precision 3/8, recall 1, F1 6/11; macro over the five present classes
    assert regex["accuracy"] == pytest.approx(3 / 8)
    assert regex["per_class"][IE]["f1"] == pytest.approx(6 / 11)
    assert regex["macro_f1"] == pytest.approx((6 / 11) / 5)
    tuned = docs["verdict_metrics"]["systems"]["qwen_finetuned_lora"]
    assert tuned["macro_f1"] == 1.0 and tuned["accuracy"] == 1.0
    mc = docs["statistical_tests"]["mcnemar"]["baselines"]["regex_baseline"]
    # regex is right on the 3 Insufficient Evidence records, tuned on all 8: b = 0, c = 5
    assert (mc["b_baseline_correct_proposed_wrong"], mc["c_baseline_wrong_proposed_correct"]) == (0, 5)
    assert mc["chi2"] == pytest.approx(16 / 5)
    assert mc["p_value"] == pytest.approx(math.erfc(math.sqrt(1.6)))
    assert set(docs["statistical_tests"]["mcnemar"]["baselines"]) == {"regex_baseline", "roberta_finetuned", "qwen_zero_shot"}
    assert set(docs["statistical_tests"]["mann_whitney"]["systems"]["qwen_finetuned_lora"]) == {
        "project_age_years", "stars", "policy_length_words"
    }
    ci = docs["statistical_tests"]["bootstrap_macro_f1"]
    assert ci["n_resamples"] == 200 and ci["seed"] == 42
    extraction = docs["extraction_metrics"]["systems"]["regex_baseline"]
    assert extraction["tau"]["accuracy"] == 1.0 and extraction["T"]["accuracy"] == 0.0
    assert extraction["tuple"]["exact_match"] == 0.0


def test_all_files_are_written_with_full_provenance(tmp_path):
    data = records()
    run_evaluation(data, factories(), engine_with(ENGINE_RULES), stub_scorer, tmp_path, probe_gpu=False, n_resamples=20,
                   data_files={"test": tmp_path / "missing.jsonl"})
    assert all((tmp_path / name).is_file() for name in FILES)
    rows = read_jsonl(tmp_path / "predictions.jsonl")
    assert len(rows) == 4 * len(data)
    for row in rows:
        assert set(row["provenance"]) == {"extractor_engine", "actual_model_used", "lora_applied", "execution_device", "degraded"}
        assert row["assessment_engine"] == "StubEngine" and row["verdict"] in VERDICTS
        assert row["explanation"] and row["evidence_trail"]
    tuned = [r for r in rows if r["system"] == "qwen_finetuned_lora"]
    assert all(r["provenance"]["lora_applied"] is True for r in tuned)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 42 and manifest["run"]["strict"] is True and manifest["run"]["degraded_predictions"] == 0
    assert manifest["data_files"]["predictions"]["sha256"]


def test_strict_run_aborts_when_a_model_is_unavailable_and_writes_nothing(tmp_path):
    broken = factories(qwen_finetuned_lora=lambda: StubExtractor(lora=True, fail=ModelUnavailableError("LoRA adapter not found")))
    with pytest.raises(EvaluationAborted, match="never falls back"):
        run_evaluation(records(), broken, engine_with(ENGINE_RULES), stub_scorer, tmp_path / "out", probe_gpu=False)
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_strict_run_aborts_when_the_model_fails_during_construction(tmp_path):
    def build():
        raise ModelUnavailableError("CUDA is required")

    with pytest.raises(EvaluationAborted, match="unavailable"):
        run_systems(records(), factories(qwen_zero_shot=build), engine_with(ENGINE_RULES), strict=True)


def test_non_strict_run_skips_an_unavailable_system_and_says_so(tmp_path):
    broken = factories(qwen_finetuned_lora=lambda: StubExtractor(fail=ModelUnavailableError("adapter missing")))
    result = run_evaluation(records(), broken, engine_with(ENGINE_RULES), stub_scorer, tmp_path, strict=False,
                            probe_gpu=False, n_resamples=20)
    assert list(result["skipped"]) == ["qwen_finetuned_lora"]
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert "qwen_finetuned_lora" in manifest["run"]["skipped_systems"]
    assert result["docs"]["statistical_tests"]["mcnemar"]["status"] == "skipped"
    assert "qwen_finetuned_lora" not in result["docs"]["verdict_metrics"]["systems"]


def test_other_errors_are_never_skipped(tmp_path):
    broken = factories(roberta_finetuned=lambda: StubExtractor(fail=ValueError("bug")))
    with pytest.raises(EvaluationAborted, match="failed"):
        run_systems(records(), broken, engine_with(ENGINE_RULES), strict=False)


def test_strict_run_rejects_degraded_output_and_mislabelled_systems():
    degraded = factories(regex_baseline=lambda: StubExtractor(degraded=True, commitment=_commitment("regex text")))
    with pytest.raises(EvaluationAborted, match="fallback"):
        run_systems(records(), degraded, engine_with(ENGINE_RULES), strict=True)
    mislabelled = factories(qwen_zero_shot=lambda: StubExtractor(lora=True, commitment=_commitment("zero-shot text")))
    with pytest.raises(EvaluationAborted, match="contradicts"):
        run_systems(records(), mislabelled, engine_with(ENGINE_RULES), strict=True)
    with pytest.raises(EvaluationAborted, match="contradicts"):
        run_systems(records(), factories(qwen_finetuned_lora=lambda: StubExtractor(lora=False, commitment=_commitment("tuned text"))),
                    engine_with(ENGINE_RULES), strict=False)


def test_missing_provenance_is_rejected_in_strict_mode():
    class Bare:
        def extract(self, text):
            return _commitment("regex text")

    with pytest.raises(EvaluationAborted, match="provenance"):
        run_systems(records(), factories(regex_baseline=Bare), engine_with(ENGINE_RULES), strict=True, systems=["regex_baseline"])


def test_engine_verdict_outside_the_five_categories_is_rejected():
    engine = StubEngine(lambda c, p: "Probably fine")
    with pytest.raises(EvaluationAborted, match="five categories"):
        run_systems(records(), factories(), engine, systems=["regex_baseline"])


def test_compute_results_rejects_incomplete_or_duplicate_predictions(tmp_path):
    data = records()
    rows, _ = run_systems(data, factories(), engine_with(ENGINE_RULES), systems=["regex_baseline"])
    from src.data_pipeline.io_utils import DataError

    with pytest.raises(DataError, match="do not match"):
        compute_results(data, rows[:-1], stub_scorer, n_resamples=10)
    with pytest.raises(DataError, match="Duplicate"):
        compute_results(data, rows + rows[:1], stub_scorer, n_resamples=10)


def test_cli_refuses_without_ground_truth_or_with_leakage(tmp_path, capsys):
    assert run_evaluation_main(["--ground-truth-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")]) == 2
    err = capsys.readouterr().err
    assert "never creates labels" in err and not (tmp_path / "out").exists()

    data = records()
    write_jsonl(tmp_path / "test.jsonl", data[:4])
    write_jsonl(tmp_path / "train.jsonl", data[3:6])  # one package in both
    assert run_evaluation_main(["--ground-truth-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")]) == 2
    assert "leakage" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_limit_writes_to_a_separate_smoke_folder(tmp_path, monkeypatch):
    import src.evaluation.cli as cli

    data = records()
    write_jsonl(tmp_path / "test.jsonl", data)
    monkeypatch.setattr(cli, "default_factories", lambda: factories())
    monkeypatch.setattr(cli, "default_engine", lambda: engine_with(ENGINE_RULES))
    monkeypatch.setattr(cli, "default_explainer", lambda: None)
    monkeypatch.setattr(cli, "resolve_scorer", lambda: stub_scorer)
    monkeypatch.setattr("src.evaluation.repro.gpu_info", lambda: {"available": False, "torch": None})
    code = run_evaluation_main(["--ground-truth-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"), "--limit", "3",
                                "--bootstrap-resamples", "20", "--systems", "regex_baseline,qwen_finetuned_lora"])
    assert code == 0
    smoke = tmp_path / "out" / "smoke_limit_3"
    assert all((smoke / name).is_file() for name in FILES)
    assert not (tmp_path / "out" / "verdict_metrics.json").exists()
    manifest = json.loads((smoke / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run"]["limit"] == 3 and manifest["run"]["smoke_test"] is True
    assert len(read_jsonl(smoke / "predictions.jsonl")) == 2 * 3


def test_explanations_come_from_layer4_for_the_predicted_commitment():
    data = records()
    explainer = StubExplainer()
    engine = StubEngine(lambda c, p: C, with_explanation=False)
    rows, _ = run_systems(data, factories(), engine, systems=["qwen_finetuned_lora"], explainer=explainer)
    assert len(explainer.calls) == len(data)
    assert all(call["commitment"]["commitment_text"] == "tuned text" for call in explainer.calls)  # predicted, not gold
    assert all("PyPI JSON API" in call["source_references"][0] for call in explainer.calls)
    assert all(row["explanation"] == "Layer 4 stub explanation: Consistent" for row in rows)
    # without an explainer and without an engine explanation the field stays empty and is never invented
    rows, _ = run_systems(data, factories(), engine, systems=["qwen_finetuned_lora"])
    assert all(row["explanation"] is None for row in rows)


class _Unavailable(StubExtractor):
    def __init__(self, problems, **kwargs):
        super().__init__(**kwargs)
        self.problems = problems

    def availability_problems(self):
        return list(self.problems)


def test_preflight_reports_every_missing_model_before_any_record_is_processed():
    extractors = []

    def tracked(problems, **kwargs):
        def build():
            extractor = _Unavailable(problems, **kwargs)
            extractors.append(extractor)
            return extractor

        return build

    setup = factories(
        roberta_finetuned=tracked(["no trained RoBERTa weights in models/roberta"], commitment=_commitment("roberta text")),
        qwen_finetuned_lora=tracked(["LoRA adapter not found in models/qwen", "CUDA is not available"], lora=True),
    )
    with pytest.raises(EvaluationAborted) as raised:
        run_systems(records(), setup, engine_with(ENGINE_RULES), strict=True)
    message = str(raised.value)
    assert "no trained RoBERTa weights" in message and "LoRA adapter not found" in message and "CUDA is not available" in message
    assert all(extractor.calls == 0 for extractor in extractors)  # nothing was extracted

    rows, skipped = run_systems(records(), setup, engine_with(ENGINE_RULES), strict=False)
    assert set(skipped) == {"roberta_finetuned", "qwen_finetuned_lora"}
    assert {row["system"] for row in rows} == {"regex_baseline", "qwen_zero_shot"}
