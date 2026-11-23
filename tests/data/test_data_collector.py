"""Collector loop with stubbed Layer 0 and Layer 2 (offline, synthetic data only)."""

import json

import pytest
import requests

from src.data_pipeline.collector import (
    CandidateListError,
    CollectionAborted,
    CollectorConfig,
    DatasetCollector,
    build_candidates,
    classify_exclusion,
    fetch_top_packages,
    load_seed_packages,
)
from src.data_pipeline.io_utils import read_jsonl
from supportlens_synthetic import policy_text, release, vulnerability

TOKEN = "ghp_synthetic_secret_token_for_tests"


class FakePipeline:
    """Replaces DataAcquisitionPipeline: outcomes are scripted per package name."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def process_package(self, name, require_version_support=False):
        assert require_version_support is True
        self.calls.append(name)
        outcome = self.outcomes[name]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def included(name, text=None):
    return {
        "package_name": name,
        "status": "INCLUDED",
        "github_repo": f"synthetic/{name}",
        "raw_policy_text": text or policy_text(1).replace("\n", "\r\n"),
        "security_policy_path": "SECURITY.md",
        "security_policy_url": "https://example.invalid/SECURITY.md",
        "releases": [release("1.0", "2020-01-01T00:00:00Z")],
        "repo_metadata": {"stars": 7, "created_at": "2019-01-01T00:00:00+00:00", "pushed_at": "2026-01-01T00:00:00+00:00"},
    }


def excluded(reason, status=None):
    return {"status": "EXCLUDED", "exclusion_reason": reason, "retrieval_status": status}


class FakeEvidence:
    def __init__(self, osv="SUCCESS", advisory="SUCCESS"):
        self.osv, self.advisory = osv, advisory
        self.tokens = []

    def collect_unified_evidence(self, name, pypi_releases=None, github_token=None):
        self.tokens.append(github_token)
        return {
            "package_name": name,
            "vulnerabilities": [vulnerability("SYN-1", ["1.0"], [])],
            "source_diagnostics": {"osv": {"status": self.osv, "count": 1}, "github_advisory": {"status": self.advisory, "count": 0}},
        }


def make(tmp_path, outcomes, *, evidence=None, **config_overrides):
    config = CollectorConfig(
        seed_file=tmp_path / "seed.txt",
        top_cache=tmp_path / "top.json",
        output=tmp_path / "dataset.jsonl",
        audit_dir=tmp_path / "audit",
        delay=0,
        only_packages=list(outcomes),
        **config_overrides,
    )
    pipeline = FakePipeline(outcomes)
    collector = DatasetCollector(config, pipeline=pipeline, evidence_collector=evidence or FakeEvidence(), sleep=lambda s: None)
    return collector, pipeline


def test_included_record_has_all_required_fields(tmp_path):
    collector, _ = make(tmp_path, {"alpha": included("alpha")})
    summary = collector.run()
    assert (summary.included, summary.dataset_size) == (1, 1)
    record = read_jsonl(tmp_path / "dataset.jsonl")[0]
    for key in ("package_name", "github_repo", "policy_text", "policy_path", "policy_url", "releases", "vulnerabilities",
                "source_diagnostics", "stars", "repo_created_at", "pushed_at", "policy_word_count", "collected_at"):
        assert key in record
    assert record["stars"] == 7 and record["policy_word_count"] == len(record["policy_text"].split())
    assert "\r" not in record["policy_text"]  # line endings normalised to LF


def test_resume_skips_finished_and_definitively_excluded_packages(tmp_path):
    outcomes = {
        "alpha": included("alpha"),
        "beta": excluded("EXCLUDED: No SECURITY.md found in repository 'x/beta' (HTTP 404 across candidate paths)."),
    }
    collector, pipeline = make(tmp_path, outcomes)
    first = collector.run()
    assert (first.included, first.excluded, first.attempted) == (1, 1, 2)

    collector2, pipeline2 = make(tmp_path, outcomes)
    second = collector2.run()
    assert pipeline2.calls == [] and second.skipped_done == 2 and second.attempted == 0
    assert len(read_jsonl(tmp_path / "dataset.jsonl")) == 1


def test_resume_honours_the_layer0_exclusion_log(tmp_path):
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit" / "exclusion_log.jsonl").write_text(
        json.dumps({"package_name": "beta", "exclusion_reason": "EXCLUDED: No verifiable GitHub repository link found in PyPI metadata."}) + "\n"
        + json.dumps({"package_name": "gamma", "exclusion_reason": "EXCLUDED: Connection error to PyPI API (timeout)."}) + "\n",
        encoding="utf-8",
    )
    outcomes = {"beta": excluded("x"), "gamma": included("gamma")}
    collector, pipeline = make(tmp_path, outcomes)
    collector.run()
    assert pipeline.calls == ["gamma"]  # beta is definitive; gamma's failure was transient and is retried


def test_transient_failures_are_retried_not_excluded(tmp_path):
    outcomes = {"alpha": excluded("EXCLUDED: GitHub verification could not be confirmed for 'a/b' (rate limit)."),
                "beta": excluded("EXCLUDED: ACCESS_FAILURE: Failed to reach GitHub", "ACCESS_FAILURE")}
    collector, _ = make(tmp_path, outcomes)
    summary = collector.run()
    assert summary.transient == 2 and summary.excluded == 0
    collector2, pipeline2 = make(tmp_path, outcomes)
    collector2.run()
    assert sorted(pipeline2.calls) == ["alpha", "beta"]


def test_exceptions_are_logged_and_the_run_continues(tmp_path):
    outcomes = {"alpha": requests.ConnectionError("network down"), "beta": included("beta")}
    collector, _ = make(tmp_path, outcomes)
    summary = collector.run()
    assert summary.errors == 1 and summary.included == 1
    log = read_jsonl(tmp_path / "audit" / "collection_log.jsonl")
    assert [e["outcome"] for e in log] == ["ERROR", "INCLUDED"]


def test_target_and_max_candidates_stop_conditions(tmp_path):
    names = {f"p{i}": included(f"p{i}") for i in range(5)}
    collector, pipeline = make(tmp_path, names, target=2)
    summary = collector.run()
    assert summary.dataset_size == 2 and len(pipeline.calls) == 2 and "target" in summary.stopped_reason

    collector2, pipeline2 = make(tmp_path / "other", names, max_candidates=3)
    summary2 = collector2.run()
    assert len(pipeline2.calls) == 3 and "max-candidates" in summary2.stopped_reason

    collector3, pipeline3 = make(tmp_path, names, target=2)  # already at the target: nothing to do
    collector3.run()
    assert pipeline3.calls == []


def test_incomplete_evidence_is_transient_unless_partial_is_allowed(tmp_path):
    collector, _ = make(tmp_path, {"alpha": included("alpha")}, evidence=FakeEvidence(advisory="RATE_LIMITED"))
    summary = collector.run()
    assert summary.transient == 1 and not (tmp_path / "dataset.jsonl").exists()

    collector2, _ = make(tmp_path / "partial", {"alpha": included("alpha")}, evidence=FakeEvidence(osv="FAILURE"), allow_partial_evidence=True)
    collector2.run()
    record = read_jsonl(tmp_path / "partial" / "dataset.jsonl")[0]
    assert record["source_diagnostics"]["osv"]["status"] == "FAILURE"  # the failure stays visible


def test_abort_after_consecutive_transient_failures(tmp_path):
    outcomes = {f"p{i}": excluded("EXCLUDED: Connection error to PyPI API (x).") for i in range(6)}
    collector, _ = make(tmp_path, outcomes, abort_after_transient=3)
    with pytest.raises(CollectionAborted, match="GITHUB_TOKEN"):
        collector.run()


def test_token_is_passed_on_but_never_written(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    evidence = FakeEvidence()
    outcomes = {"alpha": included("alpha"), "beta": RuntimeError(f"failure mentioning {TOKEN}")}
    collector, _ = make(tmp_path, outcomes, evidence=evidence)
    collector.run()
    assert evidence.tokens == [TOKEN]
    for path in list(tmp_path.rglob("*")):
        if path.is_file():
            assert TOKEN not in path.read_text(encoding="utf-8", errors="ignore")


def test_classify_exclusion():
    assert classify_exclusion("EXCLUDED: Package 'x' not found on PyPI.") == "definitive"
    assert classify_exclusion("EXCLUDED: SECURITY.md in repository 'a/b' contains no explicit version-support statement") == "definitive"
    assert classify_exclusion("EXCLUDED: PyPI API returned HTTP 503.") == "transient"
    assert classify_exclusion("EXCLUDED: PyPI API returned HTTP 403.") == "definitive"
    assert classify_exclusion("whatever", "ACCESS_FAILURE") == "transient"


def test_candidates_seed_first_deduplicated_and_normalised(tmp_path):
    seed = tmp_path / "seed.txt"
    seed.write_text("# comment\nAlpha_Pkg\n\nbeta\n", encoding="utf-8")
    names = build_candidates(load_seed_packages(seed), ["alpha-pkg", "gamma", "beta", "delta"], top_limit=3)
    assert names == ["Alpha_Pkg", "beta", "gamma"]
    assert build_candidates([], [], only=["x", "X"]) == ["x"]


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_top_list_is_downloaded_cached_and_reused(tmp_path):
    cache = tmp_path / "cache" / "top.json"
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _Response({"rows": [{"project": "one", "download_count": 3}, {"project": "two", "download_count": 2}]})

    assert fetch_top_packages(cache, http_get=fake_get) == ["one", "two"]
    assert cache.is_file() and len(calls) == 1
    assert fetch_top_packages(cache, http_get=fake_get) == ["one", "two"]
    assert len(calls) == 1  # served from the cache


def test_top_list_failure_uses_cache_or_raises(tmp_path):
    def failing_get(url, **kwargs):
        raise requests.ConnectionError("offline")

    with pytest.raises(CandidateListError, match="--seed-only"):
        fetch_top_packages(tmp_path / "none.json", http_get=failing_get)
    cache = tmp_path / "top.json"
    cache.write_text(json.dumps({"rows": [{"project": "cached"}]}), encoding="utf-8")
    assert fetch_top_packages(cache, refresh=True, http_get=failing_get) == ["cached"]
