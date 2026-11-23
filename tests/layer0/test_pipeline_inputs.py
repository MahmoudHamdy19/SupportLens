"""Layer 0 fork metadata, Layer 1 input = preprocessed text, and mapping uncertainty in the pipeline (offline)."""

import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import supportlens_cli
from src import supportlens_pipeline
from src.layer0_data_acquisition.policy_preprocessing import PolicyPreprocessor
from src.layer0_data_acquisition.repository_mapping import RepositoryMapper
from src.layer3_rule_assessment import AssessmentRuleEngine

RAW = "# Security\n<!-- hidden note -->\n![badge](https://img.shields.io/x.svg)\n\n\n\nOnly 2.x receives fixes.\n"
NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Layer 0: fork flag in the repository metadata
# ---------------------------------------------------------------------------

def test_verify_via_github_api_reports_fork(monkeypatch):
    import github

    repo = SimpleNamespace(full_name="owner/repo", default_branch="main", stargazers_count=3, forks_count=1,
                           created_at=None, pushed_at=None, archived=False, fork=True,
                           html_url="https://github.com/owner/repo")

    class FakeGithub:
        def __init__(self, *a, **k):
            pass

        def get_repo(self, name):
            return repo

        def close(self):
            pass

    monkeypatch.setattr(github, "Github", FakeGithub)
    valid, branch, _, meta, status = RepositoryMapper.verify_via_github_api("owner", "repo")
    assert valid and status == "VERIFIED" and meta["fork"] is True and meta["archived"] is False


def test_layer0_carries_fork_into_repo_metadata(monkeypatch, tmp_path):
    from src.layer0_data_acquisition import DataAcquisitionPipeline, security_md_retrieval
    from src.layer0_data_acquisition.pypi_metadata import PyPIMetadataCollector

    pypi = {"info": {"project_urls": {"Source": "https://github.com/owner/repo"}},
            "releases": {"2.0.0": [{"upload_time_iso_8601": "2024-01-01T00:00:00Z"}]}}
    meta = {"full_name": "owner/repo", "default_branch": "main", "stars": 10, "created_at": "2020-01-01T00:00:00+00:00",
            "pushed_at": "2024-01-01T00:00:00+00:00", "archived": False, "fork": True,
            "html_url": "https://github.com/owner/repo"}
    monkeypatch.setattr(PyPIMetadataCollector, "fetch_metadata", lambda self, name: (pypi, None))
    monkeypatch.setattr(RepositoryMapper, "verify_via_github_api",
                        classmethod(lambda cls, o, r, github_token=None, **k: (True, "main", "verified", meta, "VERIFIED")))
    monkeypatch.setattr(security_md_retrieval.SecurityPolicyRetriever, "retrieve_policy_detailed",
                        lambda self, repo_full_name, default_branch: {
                            "retrieval_status": "FOUND", "retrieval_reason": "synthetic", "content": RAW,
                            "path": "SECURITY.md", "url": "https://example.invalid/SECURITY.md"})
    out = DataAcquisitionPipeline(exclusion_log_path=str(tmp_path / "log.jsonl")).process_package("demo")
    assert out["repo_metadata"]["fork"] is True
    assert out["repo_metadata"]["archived"] is False
    assert out["preprocessed_text"] == PolicyPreprocessor.clean_text(RAW)


# ---------------------------------------------------------------------------
# Pipeline: preprocessed Layer 1 input and mapping_uncertain
# ---------------------------------------------------------------------------

class _Layer0:
    def __init__(self, repo_metadata):
        self.repo_metadata = repo_metadata

    def process_package(self, name):
        return {
            "status": "INCLUDED", "package_name": name, "github_repo": "owner/repo",
            "raw_policy_text": RAW, "preprocessed_text": PolicyPreprocessor.clean_text(RAW),
            "releases": [{"version": "2.0.0", "tag": "2.0.0", "published_at": "2023-01-01T00:00:00Z"}],
            "repo_metadata": self.repo_metadata,
        }


class _Extractor:
    def __init__(self):
        self.inputs = []

    def extract(self, text):
        self.inputs.append(text)
        return SimpleNamespace(to_dict=lambda: {
            "commitment_text": "Only 2.x receives fixes.", "support_status": "explicit_range",
            "version_expression": ["2.x"], "temporal_reference": "vulnerability_publication_time", "is_measurable": True,
        })


class _Layer2:
    def collect_unified_evidence(self, name, pypi_releases=None, github_token=None):
        vuln = {"id": "CVE-2024-0001", "cve_ids": ["CVE-2024-0001"], "ghsa_ids": [],
                "published_at": "2023-02-01T00:00:00Z", "affected_versions": ["2.0.0"], "affected_ranges": [],
                "fixed_versions": ["2.0.1"], "advisory_url": None}
        return {"vulnerabilities": [vuln], "source_diagnostics": {"osv": {"status": "SUCCESS"}}}


class _Layer4:
    def build_and_save(self, **kwargs):
        return {"report_id": "synthetic"}


class _RecordingEngine(AssessmentRuleEngine):
    def __init__(self):
        self.kwargs = None

    def evaluate(self, *args, **kwargs):
        self.kwargs = kwargs
        return super().evaluate(*args, **kwargs)


def _pipeline(repo_metadata):
    pipe = supportlens_pipeline.SupportLensPipeline.__new__(supportlens_pipeline.SupportLensPipeline)
    pipe.layer0 = _Layer0(repo_metadata)
    pipe.layer1 = _Extractor()
    pipe.layer2 = _Layer2()
    pipe.layer3 = _RecordingEngine()
    pipe.layer4 = _Layer4()
    pipe.github_token = None
    return pipe


def test_layer1_receives_the_preprocessed_text():
    pipe = _pipeline({"archived": False, "fork": False})
    result = pipe.audit_package("demo", reference_time=NOW)
    assert pipe.layer1.inputs == [PolicyPreprocessor.clean_text(RAW)]
    assert "<!--" not in pipe.layer1.inputs[0]
    assert pipe.layer3.kwargs["mapping_uncertain"] is False
    assert result["layer3"]["verdict"] == "Consistent"


@pytest.mark.parametrize("meta", [{"fork": True, "archived": False}, {"fork": False, "archived": True}])
def test_fork_or_archived_repository_marks_mapping_uncertain(meta):
    pipe = _pipeline(meta)
    result = pipe.audit_package("demo", reference_time=NOW)
    assert pipe.layer3.kwargs["mapping_uncertain"] is True
    assert result["mapping_uncertain"] is True
    assert result["layer3"]["verdict"] == "Insufficient Evidence"
    assert any(e["reason"] == "mapping_uncertain" for e in result["layer3"]["evidence_log"])


# ---------------------------------------------------------------------------
# CLI --file applies the same preprocessing
# ---------------------------------------------------------------------------

def test_cli_file_applies_clean_text(monkeypatch, tmp_path, capsys):
    policy = tmp_path / "SECURITY.md"
    policy.write_text(RAW, encoding="utf-8")
    seen = []

    class FakeExtractor:
        def __init__(self, strict=False):
            pass

        def extract(self, text):
            seen.append(text)
            return SimpleNamespace(to_dict=lambda: {"commitment_text": "x"})

    monkeypatch.setattr(supportlens_cli, "QwenCommitmentExtractor", FakeExtractor)
    monkeypatch.setattr(supportlens_cli, "_load_env_file", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["supportlens_cli.py", "--file", str(policy)])
    assert supportlens_cli.main() == 0
    assert seen == [PolicyPreprocessor.clean_text(RAW)]
    assert json.loads(capsys.readouterr().out) == {"commitment_text": "x"}
