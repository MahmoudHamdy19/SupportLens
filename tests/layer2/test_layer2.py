"""Offline tests for Layer 2 (thesis 3.2.3). Synthetic releases and advisories; the network is replaced."""

from datetime import datetime, timezone

import pytest

from src.layer2_evidence_layer import EvidenceCollector, NormalizedVersionScope, convert_wildcard_to_specifier
from src.layer2_evidence_layer import evidence_sources

RELEASES = [
    {"version": "1.0.0", "tag": "1.0.0", "published_at": "2023-01-01T00:00:00Z", "source": "synthetic"},
    {"version": "1.1.0", "tag": "1.1.0", "published_at": "2023-06-01T00:00:00Z", "source": "synthetic"},
    {"version": "2.0.0", "tag": "2.0.0", "published_at": "2024-01-01T00:00:00Z", "source": "synthetic"},
]


@pytest.mark.parametrize("expr,version,covered", [
    ("2.x", "2.0.0", True),
    ("2.x", "1.1.0", False),
    (">=1.1", "1.1.0", True),
    (">=1.1", "1.0.0", False),
])
def test_wildcards_and_ranges(expr, version, covered):
    scope = NormalizedVersionScope([expr], RELEASES)
    assert scope.is_version_covered(version) is covered


def test_latest_resolved_at_reference_time():
    scope = NormalizedVersionScope(["latest"], RELEASES)
    mid_2023 = datetime(2023, 7, 1, tzinfo=timezone.utc)
    assert scope.is_version_covered("1.1.0", at_time=mid_2023) is True
    assert scope.is_version_covered("2.0.0", at_time=mid_2023) is False
    assert scope.is_version_covered("2.0.0", at_time=datetime(2024, 2, 1, tzinfo=timezone.utc)) is True


def test_unparseable_expression_gives_empty_scope():
    assert NormalizedVersionScope(["the stable one"], RELEASES).specifiers == []
    assert convert_wildcard_to_specifier("5.2.x") is not None


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload


def test_dedup_by_cve_across_osv_and_github(monkeypatch, tmp_path):
    osv = {"vulns": [{
        "id": "GHSA-aaaa-bbbb-cccc", "aliases": ["CVE-2024-0001"], "summary": "synthetic",
        "published": "2024-02-01T00:00:00Z", "modified": "2024-02-02T00:00:00Z",
        "affected": [{"package": {"name": "demo", "ecosystem": "PyPI"},
                      "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "2.0.1"}]}],
                      "versions": ["2.0.0"]}],
    }]}
    gh = [{"ghsa_id": "GHSA-aaaa-bbbb-cccc", "cve_id": "CVE-2024-0001", "summary": "synthetic",
           "published_at": "2024-02-01T00:00:00Z", "html_url": "https://github.com/advisories/GHSA-aaaa-bbbb-cccc",
           "vulnerabilities": [{"package": {"ecosystem": "pip", "name": "demo"},
                                "vulnerable_version_range": "< 2.0.1", "first_patched_version": "2.0.1"}]}]
    monkeypatch.setattr(evidence_sources.requests, "post", lambda *a, **k: _Resp(200, osv))
    monkeypatch.setattr(evidence_sources.requests, "get",
                        lambda url, *a, **k: _Resp(200, gh) if "advisories" in url else _Resp(404, {}))
    monkeypatch.setattr(evidence_sources, "PROJECT_ROOT", tmp_path)
    out = EvidenceCollector().collect_unified_evidence("demo", pypi_releases=RELEASES)
    assert out["total_vulnerabilities"] == 1
    rec = out["vulnerabilities"][0]
    assert rec["id"] == "CVE-2024-0001"
    assert "2.0.1" in rec["fixed_versions"]
    assert rec["advisory_url"].startswith("https://osv.dev/vulnerability/")
    assert out["source_diagnostics"]["osv"]["status"] == "SUCCESS"


def test_osv_failure_is_reported(monkeypatch, tmp_path):
    import requests
    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(evidence_sources.requests, "post", boom)
    monkeypatch.setattr(evidence_sources.requests, "get", boom)
    monkeypatch.setattr(evidence_sources, "PROJECT_ROOT", tmp_path)
    out = EvidenceCollector().collect_unified_evidence("demo", pypi_releases=RELEASES)
    assert out["source_diagnostics"]["osv"]["status"] == "FAILURE"
    assert out["vulnerabilities"] == []


def test_multi_interval_and_git_ranges():
    """Synthetic OSV ranges: two ECOSYSTEM intervals and one GIT range that must be ignored."""
    releases = [{"version": v, "tag": v, "published_at": "2023-01-01T00:00:00Z", "source": "synthetic"}
                for v in ["1.0.0", "1.5.0", "2.0.0", "2.5.0", "3.0.0"]]
    scope = NormalizedVersionScope([">=1.0"], releases)
    ranges = [{"type": "ECOSYSTEM", "events": [{"introduced": "1.0.0"}, {"fixed": "1.5.0"},
                                               {"introduced": "2.0.0"}, {"fixed": "2.5.0"}]},
              {"type": "GIT", "events": [{"introduced": "0"}, {"fixed": "abc123"}]}]
    hit, versions, uncertain = scope.is_range_intersecting(ranges)
    assert hit and not uncertain
    assert sorted(versions) == ["1.0.0", "2.0.0"]
