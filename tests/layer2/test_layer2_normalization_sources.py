"""Offline tests for Layer 2 version normalisation (bare tokens, relative terms, pandas timeline),
evidence-source pagination and the NVD lookup. Synthetic data; the network is replaced."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.layer2_evidence_layer import EvidenceCollector, NormalizedVersionScope, convert_wildcard_to_specifier
from src.layer2_evidence_layer import evidence_sources
from src.layer2_evidence_layer.version_normalization import (
    RELATIVE_TERMS,
    VersionNormalizer,
    latest_stable_version,
    release_timeline,
)

RELEASES = [
    {"version": "1.0.0", "tag": "1.0.0", "published_at": "2023-01-01T00:00:00Z", "source": "synthetic"},
    {"version": "1.1.0", "tag": "1.1.0", "published_at": "2023-06-01T00:00:00Z", "source": "synthetic"},
    {"version": "2.0.0", "tag": "2.0.0", "published_at": "2024-01-01T00:00:00Z", "source": "synthetic"},
]


# ---------------------------------------------------------------------------
# Bare release-line tokens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token, expected", [
    ("1.2", ">=1.2.0, <1.3.0"),
    ("v1.2", ">=1.2.0, <1.3.0"),
    ("3", ">=3, <4"),
    ("1.2.3", "==1.2.3"),
    ("5.2.x", ">=5.2.0, <5.3.0"),
])
def test_bare_tokens_cover_the_whole_release_line(token, expected):
    assert convert_wildcard_to_specifier(token) == expected


@pytest.mark.parametrize("token, version, covered", [
    ("1.2", "1.2.0", True), ("1.2", "1.2.9", True), ("1.2", "1.3.0", False), ("1.2", "1.1.9", False),
    ("3", "3.0.0", True), ("3", "3.9.1", True), ("3", "4.0.0", False), ("3", "2.9.9", False),
    ("1.2.3", "1.2.3", True), ("1.2.3", "1.2.4", False),
])
def test_bare_token_coverage(token, version, covered):
    assert NormalizedVersionScope([token], RELEASES).is_version_covered(version) is covered


# ---------------------------------------------------------------------------
# Relative terms and the pandas release timeline
# ---------------------------------------------------------------------------

PRE_RELEASES = RELEASES + [
    {"version": "3.0.0rc1", "tag": "3.0.0rc1", "published_at": "2024-03-01T00:00:00Z"},
    {"version": "3.0.0.dev1", "tag": "3.0.0.dev1", "published_at": "2024-02-01T00:00:00Z"},
    {"version": "2.1.0", "tag": "2.1.0", "published_at": None},
]


@pytest.mark.parametrize("term", sorted(RELATIVE_TERMS))
def test_relative_terms_exclude_pre_and_dev_releases(term):
    at = datetime(2024, 4, 1, tzinfo=timezone.utc)
    specs = VersionNormalizer.resolve_specifiers([term], PRE_RELEASES, at)
    assert [str(s) for s in specs] == ["==2.0.0"]
    assert NormalizedVersionScope([term], PRE_RELEASES).is_version_covered("3.0.0rc1", at_time=at) is False


@pytest.mark.parametrize("term", ["latest", "current", "most recent", "newest", "latest stable"])
def test_required_relative_terms_are_recognised(term):
    assert term in RELATIVE_TERMS


def test_newest_is_resolved_at_t():
    scope = NormalizedVersionScope(["newest"], RELEASES)
    assert scope.is_version_covered("1.1.0", at_time=datetime(2023, 7, 1, tzinfo=timezone.utc)) is True
    assert scope.is_version_covered("2.0.0", at_time=datetime(2023, 7, 1, tzinfo=timezone.utc)) is False


def test_release_timeline_is_a_dataframe_filtered_at_t():
    frame = release_timeline(PRE_RELEASES, datetime(2023, 7, 1, tzinfo=timezone.utc))
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["version", "published_at"]
    assert [str(v) for v in frame["version"]] == ["1.0.0", "1.1.0"]
    full = release_timeline(PRE_RELEASES)
    assert len(full) == len(PRE_RELEASES)
    assert int(full["published_at"].isna().sum()) == 1


def test_latest_stable_version_never_uses_undated_releases_at_t():
    assert str(latest_stable_version(PRE_RELEASES, datetime(2030, 1, 1, tzinfo=timezone.utc))) == "2.0.0"
    assert str(latest_stable_version(PRE_RELEASES)) == "2.1.0"
    assert latest_stable_version(RELEASES, datetime(2020, 1, 1, tzinfo=timezone.utc)) is None
    assert latest_stable_version([], None) is None


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, status, payload, next_url=None):
        self.status_code, self._payload = status, payload
        self.headers = {"Link": f'<{next_url}>; rel="next", <https://x/last>; rel="last"'} if next_url else {}

    def json(self):
        return self._payload


def _osv_vuln(vid, cve=None, published="2024-02-01T00:00:00Z"):
    return {"id": vid, "aliases": [cve] if cve else [], "published": published,
            "affected": [{"package": {"name": "demo", "ecosystem": "PyPI"},
                          "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "2.0.1"}]}]}]}


def _gh_item(ghsa):
    return {"ghsa_id": ghsa, "cve_id": None, "published_at": "2024-02-01T00:00:00Z",
            "vulnerabilities": [{"package": {"ecosystem": "pip", "name": "demo"}, "first_patched_version": "2.0.1"}]}


def test_osv_and_github_pagination_is_followed(monkeypatch, tmp_path):
    osv_pages = [{"vulns": [_osv_vuln("PYSEC-1")], "next_page_token": "tok"}, {"vulns": [_osv_vuln("PYSEC-2")]}]
    seen_tokens = []

    def post(url, json=None, **k):
        seen_tokens.append(json.get("page_token"))
        return _Resp(200, osv_pages[len(seen_tokens) - 1])

    def get(url, *a, **k):
        if url.startswith("https://api.github.com/advisories?page=2"):
            return _Resp(200, [_gh_item("GHSA-2222-2222-2222")])
        if "advisories" in url:
            return _Resp(200, [_gh_item("GHSA-1111-1111-1111")], next_url="https://api.github.com/advisories?page=2")
        raise AssertionError("no NVD call expected without CVEs")

    monkeypatch.setattr(evidence_sources.requests, "post", post)
    monkeypatch.setattr(evidence_sources.requests, "get", get)
    monkeypatch.setattr(evidence_sources, "PROJECT_ROOT", tmp_path)
    out = EvidenceCollector().collect_unified_evidence("demo", pypi_releases=RELEASES)
    diag = out["source_diagnostics"]
    assert seen_tokens == [None, "tok"]
    assert diag["osv"]["status"] == "SUCCESS" and diag["osv"]["count"] == 2 and diag["osv"]["pages"] == 2
    assert diag["github_advisory"]["status"] == "SUCCESS" and diag["github_advisory"]["count"] == 2
    assert diag["nvd"]["status"] == "NO_CVES"
    ids = {r["id"] for r in out["vulnerabilities"]}
    assert ids == {"PYSEC-1", "PYSEC-2", "GHSA-1111-1111-1111", "GHSA-2222-2222-2222"}
    assert all("NVD" not in r["sources"] for r in out["vulnerabilities"])


def test_requests_links_attribute_is_used_for_github_pagination():
    class WithLinks:
        links = {"next": {"url": "https://api.github.com/advisories?page=3"}}

    assert evidence_sources._next_link(WithLinks()) == "https://api.github.com/advisories?page=3"
    assert evidence_sources._next_link(_Resp(200, [])) is None


def test_pagination_cap_is_reported_as_truncated(monkeypatch, tmp_path):
    monkeypatch.setattr(evidence_sources, "MAX_PAGES", 3)
    calls = []

    def post(url, json=None, **k):
        calls.append(1)
        return _Resp(200, {"vulns": [_osv_vuln(f"PYSEC-{len(calls)}")], "next_page_token": "more"})

    monkeypatch.setattr(evidence_sources.requests, "post", post)
    monkeypatch.setattr(evidence_sources.requests, "get", lambda url, *a, **k: _Resp(200, []))
    monkeypatch.setattr(evidence_sources, "PROJECT_ROOT", tmp_path)
    out = EvidenceCollector().collect_unified_evidence("demo", pypi_releases=RELEASES)
    assert len(calls) == 3
    assert out["source_diagnostics"]["osv"]["status"] == "FAILURE"
    assert out["source_diagnostics"]["osv"]["truncated"] is True


# ---------------------------------------------------------------------------
# NVD
# ---------------------------------------------------------------------------

def _nvd_payload(cve, published="2024-01-15T10:00:00.000"):
    return {"totalResults": 1, "vulnerabilities": [{"cve": {"id": cve, "published": published, "lastModified": published}}]}


def _run_with_nvd(monkeypatch, root, osv_vulns, nvd_get, **collector_kwargs):
    monkeypatch.setattr(evidence_sources.requests, "post", lambda *a, **k: _Resp(200, {"vulns": osv_vulns}))

    def get(url, *a, **k):
        if "advisories" in url:
            return _Resp(200, [])
        return nvd_get(url, *a, **k)

    monkeypatch.setattr(evidence_sources.requests, "get", get)
    monkeypatch.setattr(evidence_sources, "PROJECT_ROOT", root)
    monkeypatch.delenv("NVD_API_KEY", raising=False)
    monkeypatch.delenv("SUPPORTLENS_NVD_MAX_LOOKUPS", raising=False)
    sleeps = []
    collector = EvidenceCollector(sleep=sleeps.append, **collector_kwargs)
    return collector.collect_unified_evidence("demo", pypi_releases=RELEASES), sleeps


def test_nvd_fills_missing_publication_date_and_is_cached(monkeypatch, tmp_path):
    vulns = [_osv_vuln("PYSEC-1", "CVE-2024-0001", published=None),
             _osv_vuln("PYSEC-2", "CVE-2024-0002", published="2024-03-01T00:00:00Z")]

    def nvd_get(url, params=None, **k):
        assert url == evidence_sources.NVD_API_URL
        return _Resp(200, _nvd_payload(params["cveId"]))

    out, sleeps = _run_with_nvd(monkeypatch, tmp_path, vulns, nvd_get)
    by_id = {r["id"]: r for r in out["vulnerabilities"]}
    assert by_id["CVE-2024-0001"]["published_at"] == "2024-01-15T10:00:00.000"
    assert by_id["CVE-2024-0001"]["published_at_source"] == "NVD"
    assert by_id["CVE-2024-0002"]["published_at"] == "2024-03-01T00:00:00Z"
    assert all("NVD" in r["sources"] and "NVD" in r["database_source"] for r in by_id.values())
    nvd = out["source_diagnostics"]["nvd"]
    assert nvd["status"] == "SUCCESS" and nvd["retrieved"] == 2
    assert sleeps == [evidence_sources.NVD_DELAY_NO_KEY]
    assert (tmp_path / "data" / "cache" / "nvd" / "CVE-2024-0001.json").exists()

    def offline(*a, **k):
        raise AssertionError("cached CVEs must not be fetched again")

    again, sleeps = _run_with_nvd(monkeypatch, tmp_path, vulns, offline)
    assert again["source_diagnostics"]["nvd"]["status"] == "SUCCESS"
    assert again["source_diagnostics"]["nvd"]["cached"] == 2 and sleeps == []


def test_nvd_api_key_shortens_the_delay(monkeypatch, tmp_path):
    vulns = [_osv_vuln("PYSEC-1", "CVE-2024-0001"), _osv_vuln("PYSEC-2", "CVE-2024-0002")]
    monkeypatch.setattr(evidence_sources.requests, "post", lambda *a, **k: _Resp(200, {"vulns": vulns}))
    headers_seen = []

    def get(url, *a, headers=None, params=None, **k):
        if "advisories" in url:
            return _Resp(200, [])
        headers_seen.append(headers)
        return _Resp(200, _nvd_payload(params["cveId"]))

    monkeypatch.setattr(evidence_sources.requests, "get", get)
    monkeypatch.setattr(evidence_sources, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("NVD_API_KEY", "synthetic-key")
    sleeps = []
    out = EvidenceCollector(sleep=sleeps.append).collect_unified_evidence("demo", pypi_releases=RELEASES)
    assert sleeps == [evidence_sources.NVD_DELAY_WITH_KEY]
    assert all(h.get("apiKey") == "synthetic-key" for h in headers_seen)
    assert "synthetic-key" not in str(out)


def test_nvd_failure_is_reported_and_not_claimed_as_a_source(monkeypatch, tmp_path):
    out, _ = _run_with_nvd(monkeypatch, tmp_path, [_osv_vuln("PYSEC-1", "CVE-2024-0001", published=None)],
                           lambda url, params=None, **k: _Resp(503, {}))
    rec = out["vulnerabilities"][0]
    assert "NVD" not in rec["sources"] and "NVD" not in rec["database_source"]
    assert rec["published_at"] is None
    nvd = out["source_diagnostics"]["nvd"]
    assert nvd["status"] == "FAILURE" and nvd["failed"] == 1 and "HTTP 503" in nvd["error"]
    assert "CROSS_REFERENCED" not in str(out["source_diagnostics"])


def test_nvd_rate_limit_and_lookup_cap(monkeypatch, tmp_path):
    vulns = [_osv_vuln(f"PYSEC-{n}", f"CVE-2024-000{n}") for n in range(1, 4)]
    out, _ = _run_with_nvd(monkeypatch, tmp_path, vulns, lambda url, params=None, **k: _Resp(429, {}))
    assert out["source_diagnostics"]["nvd"]["status"] == "RATE_LIMITED"

    capped, _ = _run_with_nvd(monkeypatch, tmp_path / "cap", vulns,
                              lambda url, params=None, **k: _Resp(200, _nvd_payload(params["cveId"])),
                              max_nvd_lookups=1)
    nvd = capped["source_diagnostics"]["nvd"]
    assert nvd["status"] == "PARTIAL" and nvd["retrieved"] == 1 and nvd["skipped"] == 2
    assert sum("NVD" in r["sources"] for r in capped["vulnerabilities"]) == 1


def test_cve_unknown_to_nvd_is_not_found_not_failure(monkeypatch, tmp_path):
    out, _ = _run_with_nvd(monkeypatch, tmp_path, [_osv_vuln("PYSEC-1", "CVE-2024-0001")],
                           lambda url, params=None, **k: _Resp(200, {"totalResults": 0, "vulnerabilities": []}))
    nvd = out["source_diagnostics"]["nvd"]
    assert nvd["status"] == "SUCCESS" and nvd["not_found"] == 1 and nvd["retrieved"] == 0
    assert "NVD" not in out["vulnerabilities"][0]["sources"]


def test_osv_record_is_not_labelled_nvd_without_a_lookup(monkeypatch, tmp_path):
    out, _ = _run_with_nvd(monkeypatch, tmp_path, [_osv_vuln("PYSEC-1", "CVE-2024-0001")],
                           lambda url, params=None, **k: _Resp(500, {}))
    assert out["vulnerabilities"][0]["sources"] == ["OSV"]
