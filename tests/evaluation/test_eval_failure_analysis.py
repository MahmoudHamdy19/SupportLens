"""Q1 significance and uncertainty, RQ3 failure analysis and mapping_uncertain (synthetic data only)."""

import json

import numpy as np
import pytest
from scipy import stats as scipy_stats

from src.data_pipeline.constants import VERDICTS
from src.evaluation.failure_analysis import ACTIVITY_DEFINITION, project_activity
from src.evaluation.runner import compute_results, run_evaluation, run_systems
from src.evaluation.stats import bootstrap_indices, bootstrap_mean_ci, mann_whitney_groups
from supportlens_synthetic import gold_record, stub_scorer

C, P, PI, IE, U = VERDICTS
COLLECTED = "2026-06-01T00:00:00+00:00"

# (gold verdict, pushed_at, archived, mapping_uncertain, gold tau)
CASES = [
    (C, "2026-05-01T00:00:00Z", False, None, "explicit_range"),   # active
    (C, "2025-07-01T00:00:00Z", None, None, "explicit_range"),    # active (335 days), archived absent
    (IE, "2024-01-01T00:00:00Z", False, True, "latest_only"),     # abandoned (stale), mapping uncertain
    (IE, "2026-05-01T00:00:00Z", True, None, "latest_only"),      # abandoned (archived)
    (PI, None, False, False, "all_versions"),                     # unclassified
    (U, "2025-05-01T00:00:00Z", False, None, "ambiguous"),        # abandoned (396 days)
]


def make_records():
    records = []
    for n, (verdict, pushed, archived, uncertain, tau) in enumerate(CASES, start=1):
        record = gold_record(n, verdict, stars=n, tau=tau)
        record["collected_at"] = COLLECTED
        if pushed is not None:
            record["pushed_at"] = pushed
        if archived is not None:
            record["archived"] = archived
        if uncertain is not None:
            record["mapping_uncertain"] = uncertain
        records.append(record)
    return records


class GoldExtractor:
    """Returns the gold commitment of the record whose policy it receives (a perfect extractor)."""

    def __init__(self, records, lora=True):
        self.by_policy = {r["policy_text"]: r["commitment"] for r in records}
        self.lora = lora

    def extract(self, text):
        return {**self.by_policy[text], "is_measurable": True, "extractor_engine": "gold-stub",
                "actual_model_used": "stub", "lora_applied": self.lora, "execution_device": "cpu", "degraded": False}


class FixedExtractor:
    def extract(self, text):
        return {"commitment_text": "wrong text", "version_expression": ["9.x"], "support_status": "explicit_range",
                "temporal_reference": "release_time", "is_measurable": True, "extractor_engine": "fixed-stub",
                "actual_model_used": "stub", "lora_applied": False, "execution_device": "cpu", "degraded": False}


class Result:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return dict(self.data)


class LoggingEngine:
    """Gold verdict for the perfect extractor, Insufficient Evidence (R4) otherwise; logs mapping_uncertain."""

    def __init__(self, records):
        self.gold = {r["package_name"]: r["verdict"] for r in records}
        self.mapping_flags = {}

    def evaluate(self, commitment, vulnerabilities, releases, reference_time=None, package_name=None,
                 source_diagnostics=None, mapping_uncertain=False):
        self.mapping_flags[package_name] = mapping_uncertain
        if commitment["commitment_text"] == "wrong text":
            verdict, rules = IE, ["R4"]
            log = [{"rule_id": "R4", "reason": "no_vulnerability_in_scope"}, {"rule_id": "R4", "reason": "no_vulnerability_in_scope"}]
        else:
            verdict, rules = self.gold[package_name], ["R1"]
            log = [{"rule_id": "R1", "reason": "fixed_in_time"}]
        if mapping_uncertain:
            log.append({"rule_id": "R4", "reason": "mapping_uncertain"})
        return Result({"verdict": verdict, "rules_triggered": rules, "evidence_log": log,
                       "explanation": "synthetic", "evidence_trail": ["synthetic"]})


@pytest.fixture
def evaluated(tmp_path):
    records = make_records()
    engine = LoggingEngine(records)
    factories = {"regex_baseline": FixedExtractor, "qwen_finetuned_lora": lambda: GoldExtractor(records)}
    result = run_evaluation(records, factories, engine, stub_scorer, tmp_path, probe_gpu=False, n_resamples=1000)
    return {"records": records, "engine": engine, "docs": result["docs"], "out": tmp_path}


# --------------------------------------------------------------------------- mapping_uncertain


def test_mapping_uncertain_from_the_record_reaches_the_engine(evaluated):
    flags = evaluated["engine"].mapping_flags
    assert flags == {"synthpkg-01": False, "synthpkg-02": False, "synthpkg-03": True,
                     "synthpkg-04": False, "synthpkg-05": False, "synthpkg-06": False}
    rows = [json.loads(line) for line in (evaluated["out"] / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["package_name"]: r["mapping_uncertain"] for r in rows if r["system"] == "regex_baseline"}["synthpkg-03"] is True
    tuned = {r["package_name"]: r for r in rows if r["system"] == "qwen_finetuned_lora"}
    assert tuned["synthpkg-03"]["reason_codes"] == ["fixed_in_time", "mapping_uncertain"]
    assert tuned["synthpkg-01"]["reason_codes"] == ["fixed_in_time"]


def test_mapping_uncertain_is_false_unless_the_record_says_true():
    records = make_records()
    records[0]["mapping_uncertain"] = "yes"  # not a boolean True
    engine = LoggingEngine(records)
    run_systems(records, {"regex_baseline": FixedExtractor}, engine, systems=["regex_baseline"])
    assert engine.mapping_flags["synthpkg-01"] is False and engine.mapping_flags["synthpkg-03"] is True


# --------------------------------------------------------------------------- Q1 significance and CIs


def test_mcnemar_on_tuple_exact_match(evaluated):
    tests = evaluated["docs"]["statistical_tests"]
    mc = tests["mcnemar_tuple_exact_match"]["baselines"]["regex_baseline"]
    # the fixed extractor never matches the tuple, the perfect one always does: b = 0, c = 6
    assert (mc["b_baseline_correct_proposed_wrong"], mc["c_baseline_wrong_proposed_correct"]) == (0, 6)
    assert mc["chi2"] == pytest.approx(25 / 6)
    assert mc["p_value"] == pytest.approx(scipy_stats.chi2.sf(25 / 6, df=1))
    assert "tuple exact match" in tests["mcnemar_tuple_exact_match"]["outcome"]
    # the verdict McNemar is still reported: regex is right on the two IE records only
    verdict = tests["mcnemar"]["baselines"]["regex_baseline"]
    assert (verdict["b_baseline_correct_proposed_wrong"], verdict["c_baseline_wrong_proposed_correct"]) == (0, 4)


def test_extraction_bootstrap_cis(evaluated):
    block = evaluated["docs"]["statistical_tests"]["bootstrap_extraction"]
    assert (block["n_resamples"], block["seed"], block["level"]) == (1000, 42, 0.95)
    assert block["metrics"] == ["t_f1", "v_f1", "tuple_exact_match", "tuple_partial_match"]
    tuned, regex = block["systems"]["qwen_finetuned_lora"], block["systems"]["regex_baseline"]
    for metric in block["metrics"]:
        assert (tuned[metric]["value"], tuned[metric]["ci_low"], tuned[metric]["ci_high"]) == (1.0, 1.0, 1.0)
    assert (regex["t_f1"]["value"], regex["tuple_exact_match"]["ci_high"]) == (0.0, 0.0)
    # partial match varies with tau (explicit_range matches on records 1 and 2): 0.25 there, 0 elsewhere
    values = [0.25, 0.25, 0, 0, 0, 0]
    indices = bootstrap_indices(6, 1000, 42)
    expected = np.percentile(np.asarray(values)[indices].mean(axis=1), [2.5, 97.5])
    assert regex["tuple_partial_match"]["value"] == pytest.approx(1 / 12)
    assert [regex["tuple_partial_match"]["ci_low"], regex["tuple_partial_match"]["ci_high"]] == pytest.approx(list(expected))


def test_bootstrap_mean_ci_uses_the_given_indices():
    indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 2]])
    ci = bootstrap_mean_ci([0.0, 0.5, 1.0], indices, seed=7)
    assert ci["value"] == 0.5 and ci["n_resamples"] == 4 and ci["seed"] == 7
    assert ci["bootstrap_mean"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0, 0.0], indices)


# --------------------------------------------------------------------------- RQ3


def test_project_activity_definition():
    base = {"collected_at": COLLECTED}
    assert project_activity({**base, "pushed_at": "2025-06-01T00:00:00+00:00"}) == "active"  # exactly 365 days
    assert project_activity({**base, "pushed_at": "2025-05-31T23:00:00+00:00"}) == "abandoned"
    assert project_activity({**base, "pushed_at": "2026-05-01T00:00:00Z", "archived": True}) == "abandoned"
    assert project_activity({**base, "archived": True}) == "abandoned"
    assert project_activity(base) is None
    assert project_activity({"pushed_at": "2026-05-01T00:00:00Z"}) is None
    assert "365 days" in ACTIVITY_DEFINITION


def test_mann_whitney_active_versus_abandoned(evaluated):
    block = evaluated["docs"]["statistical_tests"]["mann_whitney_activity"]
    assert block["grouping"] == ACTIVITY_DEFINITION
    regex = block["systems"]["regex_baseline"]
    # regex is right exactly on records 3 and 4 (gold IE); active = 1, 2; abandoned = 3, 4, 6; 5 unclassified
    assert (regex["n_active"], regex["n_abandoned"], regex["n_unclassified"]) == (2, 3, 1)
    assert regex["accuracy_active"] == 0.0 and regex["accuracy_abandoned"] == pytest.approx(2 / 3)
    reference = scipy_stats.mannwhitneyu([0, 0], [1, 1, 0], alternative="two-sided", method="asymptotic", use_continuity=True)
    assert regex["p_value"] == pytest.approx(reference.pvalue)
    assert regex["u_active"] + regex["u_abandoned"] == 6
    assert block["systems"]["qwen_finetuned_lora"]["p_value"] == 1.0  # all correct in both groups


def test_mann_whitney_groups_skips_an_empty_group():
    result = mann_whitney_groups([1, 0], ["active", None], "active", "abandoned")
    assert result["status"] == "skipped" and result["n_unclassified"] == 1


def test_failure_analysis_counts(evaluated):
    analysis = evaluated["docs"]["failure_analysis"]
    assert analysis["n_records"] == 6
    assert analysis["activity_counts"] == {"abandoned": 3, "active": 2, "unclassified": 1}
    regex = analysis["systems"]["regex_baseline"]
    assert regex["verdict_error_pairs"] == {C: {IE: 2}, PI: {IE: 1}, U: {IE: 1}}
    assert regex["wrong_predictions"] == {"n": 4, "rules_triggered": {"R4": 4},
                                          "reason_codes": {"no_vulnerability_in_scope": 4}}
    # each reason counted once per record; record 3 also carries mapping_uncertain
    assert regex["insufficient_evidence_predictions"] == {
        "n": 6, "rules_triggered": {"R4": 6}, "reason_codes": {"mapping_uncertain": 1, "no_vulnerability_in_scope": 6}
    }
    by_tau = regex["extraction_errors_by_gold_tau"]
    assert by_tau["explicit_range"] == {"n": 2, "tuple_exact_match_errors": 2, "tau_errors": 0, "verdict_errors": 2,
                                        "mean_t_f1": 0.0, "mean_v_f1": 0.0}
    assert by_tau["latest_only"]["tau_errors"] == 2 and by_tau["latest_only"]["verdict_errors"] == 0
    assert by_tau["ambiguous"]["n"] == 1 and by_tau["all_versions"]["n"] == 1
    assert regex["verdict_errors_by_activity"] == {
        "active": {"n": 2, "verdict_errors": 2, "insufficient_evidence_predicted": 2},
        "abandoned": {"n": 3, "verdict_errors": 1, "insufficient_evidence_predicted": 3},
        "unclassified": {"n": 1, "verdict_errors": 1, "insufficient_evidence_predicted": 1},
    }
    tuned = analysis["systems"]["qwen_finetuned_lora"]
    assert tuned["verdict_error_pairs"] == {} and tuned["wrong_predictions"]["n"] == 0
    assert tuned["insufficient_evidence_predictions"]["n"] == 2


def test_failure_analysis_is_written_and_recomputable(evaluated):
    stored = json.loads((evaluated["out"] / "failure_analysis.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (evaluated["out"] / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    recomputed = compute_results(evaluated["records"], rows, stub_scorer)
    assert recomputed["failure_analysis"] == stored


def test_old_predictions_without_reason_codes_still_compute():
    records = make_records()
    rows, _ = run_systems(records, {"regex_baseline": FixedExtractor}, LoggingEngine(records), systems=["regex_baseline"])
    for row in rows:
        row.pop("reason_codes")
    docs = compute_results(records, rows, stub_scorer, n_resamples=20)
    assert docs["failure_analysis"]["systems"]["regex_baseline"]["wrong_predictions"]["reason_codes"] == {}
    assert docs["statistical_tests"]["mcnemar_tuple_exact_match"]["status"] == "skipped"
