"""
Layer 2: Evidence Layer
Component: Vulnerability and Release Evidence Sources
=====================================================
Component Specification:
  - PyPI: Release versions, Release dates (Chronological history)
  - OSV (Primary): Vulnerability records, Affected versions, Fixed versions
  - GitHub Advisory: Security advisories, Affected versions, Fixed versions
  - NVD: CVE records with publication dates (thesis 3.1.3.1), used to fill a missing published_at

Pagination: OSV `next_page_token` and GitHub Advisory `Link: rel="next"` are followed up to
MAX_PAGES pages per source. Reaching the cap means V may be incomplete, so the source is
reported as FAILURE with "truncated": True.

NVD rate limits (public API: 5 requests per 30 s without a key, 50 with NVD_API_KEY): live NVD
calls are spaced NVD_DELAY_NO_KEY (6 s) or NVD_DELAY_WITH_KEY (0.6 s) apart, and at most
max_nvd_lookups live calls are made per package (default NVD_MAX_LOOKUPS_NO_KEY or
NVD_MAX_LOOKUPS_WITH_KEY, override with SUPPORTLENS_NVD_MAX_LOOKUPS). CVEs beyond the cap are
not looked up and are counted as "skipped" (status PARTIAL, not a failure). Successful responses
are cached in data/cache/nvd/<CVE>.json and reused without a network call. "NVD" is recorded as a
source of a record only when an NVD record for one of its CVEs was actually retrieved; failed
lookups set diagnostics["nvd"]["status"] to FAILURE (RATE_LIMITED when every failure was HTTP 403/429).
NVD answers an invalid apiKey with HTTP 404, so the .env.example placeholder is ignored, and on the
first 404 with a key the key is dropped (api_key_rejected: True) and the lookup is repeated without it
under the public limits.

Key Capabilities:
1. Multi-source evidence collection in a single unified pipeline with per-source status.
2. Canonical vulnerability de-duplication using CVE IDs and alias resolution with provenance tracking.
3. Full preservation of affected ranges, including unpatched / zero-day vulnerabilities (fixed_versions: []).
4. Strict validation that affected records correspond to the verified PyPI package and ecosystem.
5. Strict temporal integrity: missing dates remain None/Unknown; never invent surrogate dates.
"""

import json
import os
import logging
import re
import time
from pathlib import Path
import requests
from typing import List, Dict, Any, Optional, Tuple, Set

PROJECT_ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger(__name__)

OSV_QUERY_URL = "https://api.osv.dev/v1/query"
GITHUB_ADVISORY_API = "https://api.github.com/advisories"
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

MAX_PAGES = 20
NVD_DELAY_NO_KEY = 6.0
NVD_DELAY_WITH_KEY = 0.6
NVD_MAX_LOOKUPS_NO_KEY = 25
NVD_MAX_LOOKUPS_WITH_KEY = 250
_CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d+$")


def nvd_api_key() -> Optional[str]:
    """NVD_API_KEY from the environment, or None when it is unset, blank or the .env.example placeholder."""
    key = (os.environ.get("NVD_API_KEY") or "").strip()
    if not key or key.lower().startswith("your_"):
        return None
    return key


def _next_link(resp: Any) -> Optional[str]:
    """URL of the `Link: rel="next"` page of a GitHub REST response, if any."""
    links = getattr(resp, "links", None)
    if isinstance(links, dict) and (links.get("next") or {}).get("url"):
        return links["next"]["url"]
    header = (getattr(resp, "headers", None) or {}).get("Link") or ""
    match = re.search(r'<([^>]+)>\s*;\s*rel="next"', header)
    return match.group(1) if match else None


def parse_nvd_record(payload: Dict[str, Any], cve_id: str) -> Optional[Dict[str, Any]]:
    """Extract {cve_id, published, last_modified} for `cve_id` from an NVD 2.0 response, or None."""
    for item in (payload or {}).get("vulnerabilities") or []:
        cve = (item or {}).get("cve") or {}
        if cve.get("id") == cve_id:
            return {"cve_id": cve_id, "published": cve.get("published"), "last_modified": cve.get("lastModified")}
    return None


class EvidenceCollector:
    """Collects, validates, and de-duplicates multi-source empirical evidence."""

    def __init__(self, timeout: int = 15, max_nvd_lookups: Optional[int] = None, sleep=time.sleep):
        self.timeout = timeout
        self.headers = {"User-Agent": "SupportLens-Layer2-EvidenceCollector/2.0"}
        self.max_nvd_lookups = max_nvd_lookups
        self._sleep = sleep

    def collect_unified_evidence(
        self,
        package_name: str,
        pypi_releases: Optional[List[Dict[str, Any]]] = None,
        github_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes unified multi-source empirical collection:
        1. PyPI Release History
        2. OSV Database (Independent Query)
        3. GitHub Advisory Database (Independent API Query)
        4. NVD CVE Enrichment (Independent Enrichment Query)
        Returns unified de-duplicated records with transparent per-source diagnostics.
        """
        diagnostics = {
            "pypi": {"status": "SUCCESS" if pypi_releases else "NOT_PROVIDED", "count": len(pypi_releases or []), "endpoint": "pypi.org/pypi/<pkg>/json"},
            "osv": {"status": "PENDING", "count": 0, "endpoint": OSV_QUERY_URL, "error": None},
            "github_advisory": {"status": "PENDING", "count": 0, "endpoint": GITHUB_ADVISORY_API, "error": None},
            "nvd": {"status": "PENDING", "count": 0, "endpoint": NVD_API_URL, "error": None}
        }

        # 1. Query OSV independently, following next_page_token
        osv_vulns = []
        try:
            payload: Dict[str, Any] = {"package": {"name": package_name, "ecosystem": "PyPI"}}
            pages = 0
            while True:
                resp = requests.post(OSV_QUERY_URL, json=payload, headers=self.headers, timeout=self.timeout)
                pages += 1
                if resp.status_code != 200:
                    diagnostics["osv"]["status"] = "FAILURE"
                    diagnostics["osv"]["error"] = f"HTTP {resp.status_code}"
                    break
                body = resp.json() or {}
                osv_vulns.extend(body.get("vulns", []) or [])
                token = body.get("next_page_token")
                if not token:
                    diagnostics["osv"]["status"] = "SUCCESS"
                    break
                if pages >= MAX_PAGES:
                    diagnostics["osv"]["status"] = "FAILURE"
                    diagnostics["osv"]["error"] = f"page cap of {MAX_PAGES} reached; results truncated"
                    diagnostics["osv"]["truncated"] = True
                    break
                payload = {**payload, "page_token": token}
            diagnostics["osv"]["pages"] = pages
            diagnostics["osv"]["count"] = len(osv_vulns)
        except requests.RequestException as e:
            diagnostics["osv"]["status"] = "FAILURE"
            diagnostics["osv"]["error"] = str(e)

        # 2. Query GitHub Advisory Database independently (affects parameter per GitHub Advisory REST API)
        gh_headers = dict(self.headers)
        token = github_token or os.environ.get("GITHUB_TOKEN")
        if token:
            gh_headers["Authorization"] = f"token {token}"
        gh_headers["Accept"] = "application/vnd.github.v3+json"

        gh_vulns = []
        try:
            gh_params = {"ecosystem": "pip", "affects": package_name, "per_page": 100}
            gh_resp = requests.get(GITHUB_ADVISORY_API, params=gh_params, headers=gh_headers, timeout=self.timeout)
            if gh_resp.status_code not in (200, 403, 429):
                # Fallback parameter if affects returns non-200 on specific GitHub enterprise proxies
                gh_params_alt = {"ecosystem": "pip", "package": package_name, "per_page": 100}
                gh_alt = requests.get(GITHUB_ADVISORY_API, params=gh_params_alt, headers=gh_headers, timeout=self.timeout)
                if gh_alt.status_code == 200:
                    gh_resp = gh_alt
            pages = 0
            while True:
                pages += 1
                if gh_resp.status_code == 200:
                    gh_vulns.extend(gh_resp.json() or [])
                elif gh_resp.status_code in (403, 429):
                    diagnostics["github_advisory"]["status"] = "RATE_LIMITED"
                    diagnostics["github_advisory"]["error"] = f"HTTP {gh_resp.status_code} (Rate limit reached)"
                    break
                else:
                    diagnostics["github_advisory"]["status"] = "FAILURE"
                    diagnostics["github_advisory"]["error"] = f"HTTP {gh_resp.status_code}"
                    break
                next_url = _next_link(gh_resp)
                if not next_url:
                    diagnostics["github_advisory"]["status"] = "SUCCESS"
                    break
                if pages >= MAX_PAGES:
                    diagnostics["github_advisory"]["status"] = "FAILURE"
                    diagnostics["github_advisory"]["error"] = f"page cap of {MAX_PAGES} reached; results truncated"
                    diagnostics["github_advisory"]["truncated"] = True
                    break
                gh_resp = requests.get(next_url, headers=gh_headers, timeout=self.timeout)
            diagnostics["github_advisory"]["pages"] = pages
            diagnostics["github_advisory"]["count"] = len(gh_vulns)
        except requests.RequestException as e:
            diagnostics["github_advisory"]["status"] = "FAILURE"
            diagnostics["github_advisory"]["error"] = str(e)

        # 3. Merge raw multi-source items and de-duplicate
        combined_raw = list(osv_vulns)
        # Adapt GitHub Advisory items if independently retrieved
        for g_item in gh_vulns:
            cve_id = g_item.get("cve_id")
            ghsa_id = g_item.get("ghsa_id")
            aliases = []
            if cve_id:
                aliases.append(cve_id)
            if ghsa_id:
                aliases.append(ghsa_id)
            for ident in g_item.get("identifiers", []):
                val = ident.get("value")
                if val and val not in aliases:
                    aliases.append(val)

            affected_entries = []
            for v_entry in g_item.get("vulnerabilities", []):
                v_pkg = v_entry.get("package", {})
                v_name = (v_pkg.get("name") or "").lower().replace("_", "-")
                v_eco = (v_pkg.get("ecosystem") or "").lower()
                if v_name and v_name != package_name.lower().replace("_", "-"):
                    continue
                if v_eco and v_eco not in ["pip", "pypi"]:
                    continue

                first_patched = v_entry.get("first_patched_version")
                range_str = v_entry.get("vulnerable_version_range")
                events = [{"introduced": "0"}]
                if first_patched:
                    clean_fp = str(first_patched).strip().lstrip("vV")
                    events.append({"fixed": clean_fp})

                affected_entries.append({
                    "package": {"name": package_name, "ecosystem": "PyPI"},
                    "ranges": [{
                        "type": "ECOSYSTEM",
                        "events": events,
                        "vulnerable_version_range": range_str
                    }]
                })

            if not affected_entries:
                # Fallback check for patched_versions if vulnerabilities array is omitted
                pv = g_item.get("patched_versions")
                if pv:
                    clean_pv = str(pv).replace("<", "").replace("=", "").strip().lstrip("vV")
                    events = [{"introduced": "0"}, {"fixed": clean_pv}] if clean_pv else [{"introduced": "0"}]
                    affected_entries.append({
                        "package": {"name": package_name, "ecosystem": "PyPI"},
                        "ranges": [{"type": "ECOSYSTEM", "events": events}]
                    })
                else:
                    affected_entries.append({
                        "package": {"name": package_name, "ecosystem": "PyPI"},
                        "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}]}]
                    })

            combined_raw.append({
                "id": ghsa_id or cve_id,
                "aliases": aliases,
                "published": g_item.get("published_at"),
                "modified": g_item.get("updated_at"),
                "summary": g_item.get("summary", ""),
                "database_source": "GitHub Advisory API (Independent)",
                "affected": affected_entries
            })

        # Normalize and de-duplicate
        normalized_records = self._normalize_and_deduplicate(combined_raw, package_name)

        # 4. NVD lookup (thesis 3.1.3.1): publication dates for CVE-identified records
        diagnostics["nvd"].update(self._enrich_with_nvd(normalized_records))

        return {
            "package_name": package_name,
            "vulnerabilities": normalized_records,
            "total_vulnerabilities": len(normalized_records),
            "unpatched_count": sum(1 for v in normalized_records if not v["is_patched"]),
            "source_diagnostics": diagnostics
        }

    def _normalize_and_deduplicate(
        self,
        raw_items: List[Dict[str, Any]],
        target_package: str
    ) -> List[Dict[str, Any]]:
        """
        Normalizes records, verifies package scope, and merges aliases into canonical records.
        """
        canonical_map: Dict[str, Dict[str, Any]] = {}
        pkg_lower = target_package.lower().replace("_", "-")

        alias_to_canonical: Dict[str, str] = {}

        for item in raw_items:
            raw_id = item.get("id", "")
            aliases = list(item.get("aliases", []))
            cve_aliases = [a for a in aliases if a.startswith("CVE-")]
            ghsa_aliases = [a for a in aliases if a.startswith("GHSA-")]

            # Canonical primary key: prefer CVE ID if present, else primary ID
            canonical_id = cve_aliases[0] if cve_aliases else raw_id

            # Validate affected package: must match target package in PyPI ecosystem
            affected_versions: List[str] = []
            fixed_versions: List[str] = []
            affected_ranges: List[Dict[str, Any]] = []

            for aff in item.get("affected", []):
                aff_pkg = aff.get("package", {})
                aff_name = (aff_pkg.get("name") or "").lower().replace("_", "-")
                aff_eco = (aff_pkg.get("ecosystem") or "").lower()

                # Scope check: package name and ecosystem must match
                if aff_name and aff_name != pkg_lower and aff_name != target_package.lower():
                    continue
                if aff_eco and aff_eco != "pypi":
                    continue

                for v in aff.get("versions", []):
                    if v not in affected_versions:
                        affected_versions.append(v)

                for r in aff.get("ranges", []):
                    range_type = r.get("type", "ECOSYSTEM")
                    events = r.get("events", [])
                    range_entry = {
                        "type": range_type,
                        "events": events,
                        "vulnerable_version_range": r.get("vulnerable_version_range")
                    }
                    if range_entry not in affected_ranges:
                        affected_ranges.append(range_entry)

                    for event in events:
                        if "fixed" in event:
                            fix_v = event["fixed"].strip().lstrip("vV")
                            if fix_v and fix_v not in fixed_versions:
                                fixed_versions.append(fix_v)

            # Rejection Gate (F12): If the record contains no matching affected package data, reject it
            if not affected_versions and not affected_ranges and not fixed_versions:
                continue

            # Determine sources
            # Sources: where this record was actually retrieved from. NVD is added only after a
            # successful NVD lookup (see _enrich_with_nvd).
            from_github_api = str(item.get("database_source") or "").startswith("GitHub Advisory")
            sources = ["GitHub Advisory"] if from_github_api else ["OSV"]

            # Strict date handling: never invent timestamps
            pub_date = item.get("published")
            mod_date = item.get("modified")
            is_patched = len(fixed_versions) > 0

            # Transitive alias resolution: find if canonical_id, raw_id, or any alias already known
            candidate_keys = [canonical_id, raw_id] + aliases
            merge_key = None
            for ck in candidate_keys:
                if ck in alias_to_canonical:
                    merge_key = alias_to_canonical[ck]
                    break
                elif ck in canonical_map:
                    merge_key = ck
                    break

            if merge_key and merge_key in canonical_map:
                existing = canonical_map[merge_key]
                # Merge affected versions
                for v in affected_versions:
                    if v not in existing["affected_versions"]:
                        existing["affected_versions"].append(v)
                # Merge fixed versions
                for fv in fixed_versions:
                    if fv not in existing["fixed_versions"]:
                        existing["fixed_versions"].append(fv)
                # Merge affected ranges (F12)
                for ar in affected_ranges:
                    if ar not in existing["affected_ranges"]:
                        existing["affected_ranges"].append(ar)
                # Merge sources & provenance
                for s in sources:
                    if s not in existing["sources"]:
                        existing["sources"].append(s)
                # Merge aliases, CVEs, GHSAs
                for a in aliases:
                    if a not in existing["aliases"]:
                        existing["aliases"].append(a)
                for cid in cve_aliases:
                    if cid not in existing["cve_ids"]:
                        existing["cve_ids"].append(cid)
                for gid in ghsa_aliases:
                    if gid not in existing["ghsa_ids"]:
                        existing["ghsa_ids"].append(gid)

                # Date preservation: adopt pub_date if existing was missing
                if not existing.get("published_at") and pub_date:
                    existing["published_at"] = pub_date

                existing["is_patched"] = len(existing["fixed_versions"]) > 0
                existing["database_source"] = " / ".join(existing["sources"])

                # Update transitive mapping
                for ck in candidate_keys:
                    alias_to_canonical[ck] = merge_key
            else:
                initial_cves = list(dict.fromkeys(cve_aliases + ([raw_id] if raw_id.startswith("CVE-") else [])))
                initial_ghsas = list(dict.fromkeys(ghsa_aliases + ([raw_id] if raw_id.startswith("GHSA-") else [])))
                canonical_map[canonical_id] = {
                    "id": canonical_id,
                    "osv_id": raw_id,
                    "cve_ids": initial_cves,
                    "ghsa_ids": initial_ghsas,
                    "aliases": aliases,
                    "summary": item.get("summary", "No summary provided"),
                    "details": item.get("details", ""),
                    "published_at": pub_date,  # Strict: None if missing
                    "modified_at": mod_date,
                    "affected_versions": affected_versions,
                    "affected_ranges": affected_ranges,
                    "fixed_versions": fixed_versions,
                    "is_patched": is_patched,
                    "sources": sources,
                    "database_source": " / ".join(sources),
                    "advisory_url": f"https://osv.dev/vulnerability/{raw_id}"
                }
                for ck in candidate_keys:
                    alias_to_canonical[ck] = canonical_id

        return list(canonical_map.values())

    def _nvd_lookup_cap(self, has_key: bool) -> int:
        if self.max_nvd_lookups is not None:
            return max(int(self.max_nvd_lookups), 0)
        env = (os.environ.get("SUPPORTLENS_NVD_MAX_LOOKUPS") or "").strip()
        if env.isdigit():
            return int(env)
        return NVD_MAX_LOOKUPS_WITH_KEY if has_key else NVD_MAX_LOOKUPS_NO_KEY

    def _enrich_with_nvd(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Look up each CVE in NVD (cache first) and fill a missing published_at from NVD `published`.

        Returns the NVD diagnostics: status NO_CVES, SUCCESS, PARTIAL (lookup cap reached, no
        failures), FAILURE or RATE_LIMITED, with counts of retrieved, not_found, failed, skipped
        and cached CVEs.
        """
        cves = list(dict.fromkeys(
            c for r in records for c in r.get("cve_ids", []) if _CVE_PATTERN.match(str(c))
        ))
        result: Dict[str, Any] = {"count": 0, "retrieved": 0, "not_found": 0, "failed": 0,
                                  "skipped": 0, "cached": 0, "error": None}
        if not cves:
            result["status"] = "NO_CVES"
            return result

        cache_dir = PROJECT_ROOT / "data" / "cache" / "nvd"
        api_key = nvd_api_key()
        nvd_headers = dict(self.headers)
        if api_key:
            nvd_headers["apiKey"] = api_key
        delay = NVD_DELAY_WITH_KEY if api_key else NVD_DELAY_NO_KEY
        cap = self._nvd_lookup_cap(bool(api_key))
        result["api_key_used"] = bool(api_key)
        result["api_key_rejected"] = False

        found: Dict[str, Dict[str, Any]] = {}
        live_calls = 0
        errors: List[str] = []
        rate_limited = 0
        for cve in cves:
            cached_file = cache_dir / f"{cve}.json"
            payload = None
            if cached_file.exists():
                try:
                    payload = json.loads(cached_file.read_text(encoding="utf-8"))
                    result["cached"] += 1
                except (OSError, ValueError):
                    payload = None
            if payload is None:
                if live_calls >= cap:
                    result["skipped"] += 1
                    continue
                if live_calls:
                    self._sleep(delay)
                live_calls += 1
                try:
                    resp = requests.get(NVD_API_URL, params={"cveId": cve}, headers=nvd_headers, timeout=self.timeout)
                    if resp.status_code == 404 and "apiKey" in nvd_headers:
                        # NVD answers an invalid apiKey with HTTP 404 for every CVE. Drop the key,
                        # fall back to the public limits and repeat this lookup once without it.
                        log.warning("NVD rejected NVD_API_KEY (HTTP 404); continuing without the key.")
                        nvd_headers.pop("apiKey")
                        result["api_key_rejected"] = True
                        delay = NVD_DELAY_NO_KEY
                        cap = min(cap, self._nvd_lookup_cap(False))
                        self._sleep(delay)
                        resp = requests.get(NVD_API_URL, params={"cveId": cve}, headers=nvd_headers, timeout=self.timeout)
                except requests.RequestException as e:
                    errors.append(f"{cve}: {e}")
                    continue
                if resp.status_code in (403, 429):
                    rate_limited += 1
                    errors.append(f"{cve}: HTTP {resp.status_code} (rate limit)")
                    continue
                if resp.status_code != 200:
                    errors.append(f"{cve}: HTTP {resp.status_code}")
                    continue
                try:
                    payload = resp.json()
                except ValueError:
                    errors.append(f"{cve}: invalid JSON")
                    continue
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cached_file.write_text(json.dumps(payload), encoding="utf-8")
                except OSError:
                    pass
            record = parse_nvd_record(payload, cve)
            if record is None:
                result["not_found"] += 1
            else:
                found[cve] = record

        for rec in records:
            hits = [found[c] for c in rec.get("cve_ids", []) if c in found]
            if not hits:
                continue
            rec["nvd_published_at"] = hits[0].get("published")
            if not rec.get("published_at") and hits[0].get("published"):
                rec["published_at"] = hits[0]["published"]
                rec["published_at_source"] = "NVD"
            if "NVD" not in rec["sources"]:
                rec["sources"].append("NVD")
            rec["database_source"] = " / ".join(rec["sources"])

        result["max_lookups"] = cap
        result["delay_seconds"] = delay
        result["retrieved"] = result["count"] = len(found)
        result["failed"] = len(errors)
        if errors:
            result["status"] = "RATE_LIMITED" if rate_limited == len(errors) else "FAILURE"
            extra = f" and {len(errors) - 5} more" if len(errors) > 5 else ""
            result["error"] = "; ".join(errors[:5]) + extra
        elif result["skipped"]:
            result["status"] = "PARTIAL"
            result["error"] = f"{result['skipped']} CVE(s) not looked up: lookup cap of {cap} per package reached"
        else:
            result["status"] = "SUCCESS"
        return result

    def fetch_osv_vulnerabilities(self, package_name: str) -> List[Dict[str, Any]]:
        """
        Backward-compatible interface returning normalized vulnerability list.
        """
        res = self.collect_unified_evidence(package_name)
        return res["vulnerabilities"]
