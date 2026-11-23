"""
Layer 2: Evidence Layer
Component: Normalized Version Scope
====================================
Component Specification:
Produces the structured 'Normalized Version Scope' containing parsed specifiers,
active release tags at reference time T, and version coverage checking functions.
Compares supported scope against vulnerability records and passes structured metrics to Layer 3.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from packaging.version import Version, InvalidVersion
from packaging.specifiers import SpecifierSet

from .version_normalization import RELATIVE_TERMS, VersionNormalizer, clean_version_string, parse_iso_datetime

class NormalizedVersionScope:
    """Encapsulates the normalized support boundaries and coverage checking."""

    def __init__(
        self,
        raw_expressions: List[str],
        releases: List[Dict[str, Any]],
        reference_time: Optional[datetime] = None
    ):
        self.raw_expressions = raw_expressions
        self.releases = releases
        self.reference_time = reference_time
        self.specifiers: List[SpecifierSet] = VersionNormalizer.resolve_specifiers(
            raw_expressions, releases, reference_time
        )

    def is_version_covered(self, version_str: str, at_time: Optional[datetime] = None) -> bool:
        """
        Determines if a version string falls within the normalized scope.
        If at_time is provided, dynamically resolves temporal terms ('latest') at that anchor.
        """
        clean = clean_version_string(version_str)
        try:
            v_obj = Version(clean)
        except InvalidVersion:
            return False

        active_specifiers = self.specifiers
        if at_time is not None and any(str(e).strip().lower() in RELATIVE_TERMS for e in self.raw_expressions):
            active_specifiers = VersionNormalizer.resolve_specifiers(
                self.raw_expressions, self.releases, at_time
            )

        if not active_specifiers:
            return False

        for spec in active_specifiers:
            if v_obj in spec:
                return True
        return False

    def is_range_intersecting(
        self,
        affected_ranges: List[Dict[str, Any]],
        at_time: Optional[datetime] = None
    ) -> Tuple[bool, List[str], bool]:
        """
        Expands vulnerability version ranges against package releases at reference time T
        and computes the mathematical intersection with policy's supported version specifiers.
        Returns: (has_intersection, expanded_in_scope_versions, is_uncertain)
        """
        if not affected_ranges:
            return False, [], False

        expanded_in_scope = []
        parsed_any_range = False

        # Gather relevant releases up to reference time T
        eligible_releases = []
        for r in self.releases:
            p_dt = parse_iso_datetime(r.get("published_at"))
            if at_time is not None and p_dt and p_dt > at_time:
                continue
            v_clean = clean_version_string(r.get("tag", ""))
            try:
                eligible_releases.append(Version(v_clean))
            except InvalidVersion:
                pass

        if not eligible_releases:
            # Cannot expand without known release history
            return False, [], True

        for r_entry in affected_ranges:
            events = r_entry.get("events", [])
            # GIT ranges are expressed in commit hashes and cannot be mapped onto PyPI versions.
            if not events or str(r_entry.get("type", "")).upper() == "GIT":
                continue

            # OSV events may describe several intervals: each "introduced" opens one,
            # the next "fixed"/"last_affected"/"limit" closes it.
            intervals = []
            current = None
            for ev in events:
                if "introduced" in ev:
                    raw = ev["introduced"]
                    try:
                        start = None if raw in ["0", None, ""] else Version(clean_version_string(raw))
                    except InvalidVersion:
                        continue
                    current = {"start": start, "end": None, "inclusive_end": False}
                    intervals.append(current)
                for key, inclusive in (("fixed", False), ("limit", False), ("last_affected", True)):
                    if key in ev and ev[key] and current is not None and current["end"] is None:
                        try:
                            current["end"] = Version(clean_version_string(ev[key]))
                            current["inclusive_end"] = inclusive
                        except InvalidVersion:
                            pass
            if not intervals:
                continue
            parsed_any_range = True

            for cand_v in eligible_releases:
                affected = False
                for iv in intervals:
                    if iv["start"] is not None and cand_v < iv["start"]:
                        continue
                    if iv["end"] is not None and (cand_v > iv["end"] if iv["inclusive_end"] else cand_v >= iv["end"]):
                        continue
                    affected = True
                    break
                if not affected:
                    continue
                v_str = str(cand_v)
                if self.is_version_covered(v_str, at_time=at_time) and v_str not in expanded_in_scope:
                    expanded_in_scope.append(v_str)

        if not parsed_any_range:
            return False, [], True

        return len(expanded_in_scope) > 0, expanded_in_scope, False

    def get_supported_releases_count(self) -> int:
        """Returns total releases falling within the normalized scope."""
        count = 0
        for r in self.releases:
            if self.is_version_covered(r.get("tag", "")):
                count += 1
        return count

    def compare_with_vulnerabilities(
        self,
        vulns: List[Dict[str, Any]],
        releases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Methodological Step: Compare declared supported versions with empirical affected versions.
        Passes grounded, reproducible comparison metrics to Layer 3 Rule Engine.
        """
        rel_list = releases or self.releases
        release_dates = {}
        for r in rel_list:
            v_clean = clean_version_string(r.get("tag", ""))
            p_dt = parse_iso_datetime(r.get("published_at"))
            if v_clean and p_dt:
                release_dates[v_clean] = p_dt

        comparison_results = []
        in_scope_count = 0
        unpatched_in_scope = []
        remediated_in_scope = []

        for v in vulns:
            vuln_id = v.get("id") or v.get("osv_id")
            pub_dt = parse_iso_datetime(v.get("published_at"))
            aff_versions = v.get("affected_versions", [])
            fixed_versions = v.get("fixed_versions", [])
            is_patched = v.get("is_patched", len(fixed_versions) > 0)

            # Check if any affected version was supported at publication time
            affected_in_scope = [
                ver for ver in aff_versions if self.is_version_covered(ver, at_time=pub_dt)
            ]

            is_in_scope = len(affected_in_scope) > 0
            # If affected_versions list is empty, expand ranges and evaluate mathematical intersection
            if not is_in_scope and not aff_versions and v.get("affected_ranges"):
                has_intersect, expanded_in_scope, uncertain = self.is_range_intersecting(
                    v.get("affected_ranges", []), at_time=pub_dt
                )
                if has_intersect:
                    is_in_scope = True
                    affected_in_scope = expanded_in_scope
                else:
                    is_in_scope = False

            # Calculate actual remediation time without inventing dates
            remediation_days = None
            if is_patched and pub_dt and fixed_versions:
                # Find earliest release date among fixed versions
                fixed_dates = [release_dates[clean_version_string(fv)] for fv in fixed_versions if clean_version_string(fv) in release_dates]
                if fixed_dates:
                    earliest_fix_dt = min(fixed_dates)
                    if earliest_fix_dt >= pub_dt:
                        remediation_days = round((earliest_fix_dt - pub_dt).total_seconds() / 86400.0, 1)

            record = {
                "vuln_id": vuln_id,
                "is_in_scope": is_in_scope,
                "affected_in_scope": affected_in_scope,
                "is_patched": is_patched,
                "fixed_versions": fixed_versions,
                "published_at": v.get("published_at"),
                "remediation_days": remediation_days
            }
            comparison_results.append(record)

            if is_in_scope:
                in_scope_count += 1
                if not is_patched:
                    unpatched_in_scope.append(record)
                else:
                    remediated_in_scope.append(record)

        return {
            "total_vulnerabilities_checked": len(vulns),
            "vulnerabilities_in_scope_count": in_scope_count,
            "unpatched_in_scope_count": len(unpatched_in_scope),
            "remediated_in_scope_count": len(remediated_in_scope),
            "unpatched_in_scope": unpatched_in_scope,
            "remediated_in_scope": remediated_in_scope,
            "details": comparison_results
        }
