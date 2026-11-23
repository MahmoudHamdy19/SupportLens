"""Collector records: preprocessed policy_text, raw_policy_text and mapping-uncertainty fields (offline)."""

import pytest

from src.data_pipeline.collector import CollectorConfig, DatasetCollector, build_record
from src.data_pipeline.io_utils import read_jsonl
from src.layer0_data_acquisition.policy_preprocessing import PolicyPreprocessor
from supportlens_synthetic import release, vulnerability

RAW = (
    "# Security Policy\r\n<!-- maintainers: keep this table current -->\r\n"
    "[![CI](https://img.shields.io/ci.svg)](https://ci.example)\r\n\r\n\r\n\r\n"
    "## Supported Versions\r\n\r\nOnly 2.x receives security fixes.\r\n"
)


def _result(raw=RAW, preprocessed=None, meta=None):
    result = {
        "package_name": "alpha",
        "status": "INCLUDED",
        "github_repo": "synthetic/alpha",
        "raw_policy_text": raw,
        "security_policy_path": "SECURITY.md",
        "security_policy_url": "https://example.invalid/SECURITY.md",
        "releases": [release("2.0", "2020-01-01T00:00:00Z")],
        "repo_metadata": meta or {"stars": 7, "created_at": "2019-01-01T00:00:00+00:00",
                                  "pushed_at": "2026-01-01T00:00:00+00:00", "archived": False, "fork": False},
    }
    if preprocessed is not None:
        result["preprocessed_text"] = preprocessed
    return result


EVIDENCE = {"vulnerabilities": [vulnerability("SYN-1", ["2.0"], [])],
            "source_diagnostics": {"osv": {"status": "SUCCESS"}, "github_advisory": {"status": "SUCCESS"}}}


def _meta(**extra):
    base = {"stars": 7, "created_at": "2019-01-01T00:00:00+00:00", "pushed_at": "2026-01-01T00:00:00+00:00"}
    base.update(extra)
    return base


def test_policy_text_is_the_layer0_preprocessed_text():
    cleaned = PolicyPreprocessor.clean_text(RAW)
    record = build_record(_result(preprocessed=cleaned), EVIDENCE, _meta(fork=False, archived=False), "now")
    assert record["policy_text"] == cleaned
    assert "<!--" not in record["policy_text"] and "shields.io" not in record["policy_text"]
    assert "\n\n\n" not in record["policy_text"] and "\r" not in record["policy_text"]
    assert record["raw_policy_text"] == RAW.replace("\r\n", "\n")
    assert record["policy_word_count"] == len(record["policy_text"].split())


def test_policy_text_falls_back_to_clean_text_when_preprocessed_is_absent():
    record = build_record(_result(), EVIDENCE, _meta(), "now")
    assert record["policy_text"] == PolicyPreprocessor.clean_text(RAW)


@pytest.mark.parametrize("fork, archived, uncertain", [
    (False, False, False), (True, False, True), (False, True, True), (True, True, True), (None, None, False),
])
def test_mapping_uncertainty_fields(fork, archived, uncertain):
    record = build_record(_result(), EVIDENCE, _meta(fork=fork, archived=archived), "now")
    assert record["repo_fork"] is fork and record["repo_archived"] is archived
    assert record["mapping_uncertain"] is uncertain


class _Pipeline:
    def __init__(self, result):
        self.result = result

    def process_package(self, name, require_version_support=False):
        return self.result


class _Evidence:
    def collect_unified_evidence(self, name, pypi_releases=None, github_token=None):
        return EVIDENCE


def test_collector_loop_stores_fork_and_archived_from_layer0(tmp_path):
    meta = {"stars": 7, "created_at": "2019-01-01T00:00:00+00:00", "pushed_at": "2026-01-01T00:00:00+00:00",
            "archived": True, "fork": False}
    result = _result(preprocessed=PolicyPreprocessor.clean_text(RAW), meta=meta)
    config = CollectorConfig(seed_file=tmp_path / "seed.txt", top_cache=tmp_path / "top.json",
                             output=tmp_path / "dataset.jsonl", audit_dir=tmp_path / "audit", delay=0,
                             only_packages=["alpha"])
    DatasetCollector(config, pipeline=_Pipeline(result), evidence_collector=_Evidence(), sleep=lambda s: None).run()
    record = read_jsonl(tmp_path / "dataset.jsonl")[0]
    assert record["repo_archived"] is True and record["repo_fork"] is False
    assert record["mapping_uncertain"] is True
    assert record["policy_text"] == PolicyPreprocessor.clean_text(RAW)
