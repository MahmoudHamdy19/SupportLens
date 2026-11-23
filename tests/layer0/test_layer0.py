"""Offline tests for Layer 0 (thesis 3.2.1). All inputs are synthetic and every network call is replaced."""

import json

import pytest

from src.layer0_data_acquisition import DataAcquisitionPipeline, PolicyPreprocessor, PyPIMetadataCollector, RepositoryMapper

SYNTHETIC_POLICY = """# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 2.x     | yes       |
| 1.x     | no        |

## Reporting a Vulnerability

Email security@example.org.
"""


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/owner/repo", ("owner", "repo")),
    ("github.com/owner/repo.git", ("owner", "repo")),
    ("https://github.com.evil.com/owner/repo", None),
    ("https://gitlab.com/owner/repo", None),
    ("https://github.com/features/actions", None),
    ("https://github.com/owner", None),
])
def test_github_url_scope(url, expected):
    assert RepositoryMapper.validate_github_url_scope(url) == expected


def test_release_dates_never_invented():
    data = {"releases": {
        "1.0.0": [{"upload_time_iso_8601": "2024-01-02T00:00:00Z"}],
        "0.9.0": [{"upload_time_iso_8601": "2023-05-01T00:00:00Z"}],
        "0.1.0": [],
    }}
    releases = PyPIMetadataCollector().extract_releases(data)
    assert [r["version"] for r in releases] == ["0.9.0", "1.0.0", "0.1.0"]
    assert releases[-1]["published_at"] is None


def test_version_support_section_and_line_numbers():
    detail = PolicyPreprocessor.extract_version_support_section(SYNTHETIC_POLICY)
    assert detail["found"] is True
    assert detail["heading"].lower().startswith("supported")
    assert detail["line_start"] >= 1 and detail["line_end"] >= detail["line_start"]
    assert ["2.x", "yes"] in detail["tables"]
    sections = PolicyPreprocessor.extract_relevant_sections(SYNTHETIC_POLICY)
    assert "patching_sla" not in sections


def test_prioritised_text_puts_version_support_first():
    text = PolicyPreprocessor.prioritize_version_support(SYNTHETIC_POLICY)
    assert text.index("2.x") < text.index("security@example.org")


def _patch_network(monkeypatch, repo_valid=True, policy=SYNTHETIC_POLICY):
    pypi = {"info": {"project_urls": {"Source": "https://github.com/owner/repo"}},
            "releases": {"2.0.0": [{"upload_time_iso_8601": "2024-01-01T00:00:00Z"}]}}
    monkeypatch.setattr(PyPIMetadataCollector, "fetch_metadata", lambda self, name: (pypi, None))
    meta = {"full_name": "owner/repo", "default_branch": "main", "stars": 10,
            "created_at": "2020-01-01T00:00:00+00:00", "pushed_at": "2024-01-01T00:00:00+00:00",
            "archived": False, "html_url": "https://github.com/owner/repo"}
    if repo_valid:
        monkeypatch.setattr(RepositoryMapper, "verify_via_github_api",
                            classmethod(lambda cls, o, r, github_token=None, **k: (True, "main", "verified", meta, "VERIFIED")))
    else:
        monkeypatch.setattr(RepositoryMapper, "verify_via_github_api",
                            classmethod(lambda cls, o, r, github_token=None, **k: (False, None, "EXCLUDED: 404", None, "NOT_FOUND")))
    from src.layer0_data_acquisition import security_md_retrieval
    monkeypatch.setattr(security_md_retrieval.SecurityPolicyRetriever, "retrieve_policy_detailed",
                        lambda self, repo_full_name, default_branch: {
                            "retrieval_status": "FOUND" if policy else "MISSING_FILE",
                            "retrieval_reason": "synthetic", "content": policy,
                            "path": "SECURITY.md", "url": "https://raw.githubusercontent.com/owner/repo/main/SECURITY.md"})


def test_pipeline_included_with_repo_metadata(monkeypatch, tmp_path):
    _patch_network(monkeypatch)
    out = DataAcquisitionPipeline(exclusion_log_path=str(tmp_path / "log.jsonl")).process_package("demo", require_version_support=True)
    assert out["status"] == "INCLUDED"
    assert out["repo_metadata"]["stars"] == 10
    assert out["has_version_support_statement"] is True
    assert out["policy_word_count"] > 0


def test_pipeline_excludes_and_logs_missing_policy(monkeypatch, tmp_path):
    _patch_network(monkeypatch, policy=None)
    log = tmp_path / "log.jsonl"
    out = DataAcquisitionPipeline(exclusion_log_path=str(log)).process_package("demo")
    assert out["status"] == "EXCLUDED"
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["package_name"] == "demo" and "SECURITY.md" in entry["exclusion_reason"]


def test_pipeline_excludes_policy_without_version_statement(monkeypatch, tmp_path):
    _patch_network(monkeypatch, policy="# Security\n\nPlease email security@example.org to report issues.\n")
    out = DataAcquisitionPipeline(exclusion_log_path=str(tmp_path / "log.jsonl")).process_package("demo", require_version_support=True)
    assert out["status"] == "EXCLUDED"
    assert "version-support" in out["exclusion_reason"]
