"""
Layer 3: Rule-Based Assessment Layer
Component: Rule Engine (Algorithm 1, thesis Section 3.2.4)
===========================================================
Deterministic evaluation of a commitment c = <t, tau, v, T> against vulnerability
evidence V and release history R. Rules are applied in strict priority order:

  R5 Unmeasurable Commitment : tau is ambiguous, or v cannot be normalised (halt).
  R4 Insufficient Evidence   : V or R missing, a failed source, or uncertain mapping (halt).
  R1 Consistent              : covered vulnerability, fixed on every covered version.
  R2 Partially Consistent    : covered vulnerability, fixed only on some release lines.
  R3 Potential Inconsistency : covered vulnerability, unfixed beyond the reasonable timeframe.
  R4 (per vulnerability)     : covered but unfixed within the timeframe, or dates/mapping missing.

R1 to R4 are assigned per vulnerability and aggregated by highest severity (Algorithm 1,
Table 3.1), checked in this exact order:

  1. no assessed vulnerability                 -> Insufficient Evidence
  2. any R3 together with any R1 or R2         -> Partially Consistent (supported for some
                                                  vulnerabilities but not others)
  3. any R3 (even when R4 records exist)       -> Potential Inconsistency
  4. any R2                                    -> Partially Consistent
  5. any R4                                    -> Insufficient Evidence
  6. otherwise (all R1)                        -> Consistent

A confirmed R3 is therefore never hidden by R4 outcomes. Vulnerabilities that affect no
covered version are ignored (out of scope).

Interpretations where the thesis is silent (also stated in the evidence trail):
  * An empty vulnerability list is treated as "V is missing": there is nothing to test the
    commitment against, so the commitment is not judged consistent.
  * If no vulnerability affects a covered version, the result is Insufficient Evidence:
    Algorithm 1 leaves this case open and the commitment was never tested.
  * An empty version expression is completed from tau: latest_only -> "latest",
    all_versions -> "all". explicit_range with no expression is unmeasurable (R5).
  * The commitment time T is the vulnerability publication time or, for release_time, the
    release date of each affected version. T only matters when the scope contains a relative
    term such as "latest"; for fixed ranges coverage does not depend on T.
  * reference_time is only the evaluation clock used to measure elapsed time since publication.
  * A release line is the major version. A covered affected version is remediated when a
    fixed release with a higher version exists on the same major version.
  * Elapsed time is counted from vulnerability publication, in whole days rounded up. Unfixed
    for more than REASONABLE_TIMEFRAME_DAYS gives R3; exactly that many days is still R4.
  * Source statuses FAILURE and RATE_LIMITED count as a source failure.

The engine performs no network access and no file I/O.
"""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from packaging.version import InvalidVersion, Version

from src.layer2_evidence_layer.normalized_scope import NormalizedVersionScope
from src.layer2_evidence_layer.version_normalization import (
    RELATIVE_TERMS,
    VersionNormalizer,
    clean_version_string,
    parse_iso_datetime,
)

from .schema import EvidenceLogEntry, RuleAssessmentResult
from .verdict_categories import VerdictCategory

REASONABLE_TIMEFRAME_DAYS = 90

RULE_VERDICTS: Dict[str, VerdictCategory] = {
    "R1": VerdictCategory.CONSISTENT,
    "R2": VerdictCategory.PARTIALLY_CONSISTENT,
    "R3": VerdictCategory.POTENTIAL_INCONSISTENCY,
    "R4": VerdictCategory.INSUFFICIENT_EVIDENCE,
    "R5": VerdictCategory.UNMEASURABLE_COMMITMENT,
}

_COMMITMENT_TYPES = frozenset({"latest_only", "explicit_range", "all_versions", "ambiguous"})
# Relative terms resolved against the release history; mirrors VersionNormalizer.
_TEMPORAL_EXPRESSIONS = RELATIVE_TERMS
_FAILED_SOURCE_STATUSES = frozenset({"FAILURE", "RATE_LIMITED"})
# Sources that supply the records of V or R; a failure there means V or R may be incomplete (R4).
_RECORD_SOURCES = frozenset({"osv", "github_advisory", "pypi"})
_VERSION_RANGE_TYPES = frozenset({"ECOSYSTEM", "SEMVER"})
_PROBE_RELEASES = [{"version": "0.0.1", "tag": "0.0.1", "published_at": "1970-01-01T00:00:00+00:00"}]
_MAX_LISTED = 5


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _parse_version(text: Any) -> Optional[Version]:
    try:
        return Version(clean_version_string(str(text)))
    except InvalidVersion:
        return None


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _day(moment: Optional[datetime]) -> str:
    return moment.date().isoformat() if moment else "date unknown"


def _brief(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= _MAX_LISTED:
        return ", ".join(items)
    return f"{', '.join(items[:_MAX_LISTED])} and {len(items) - _MAX_LISTED} more"


def _dated(versions: Sequence[str], dates: Dict[Version, datetime]) -> str:
    """List versions with their release dates (first few only), for the evidence trail."""
    parts = []
    for version in versions[:_MAX_LISTED]:
        released = dates.get(_parse_version(version))
        parts.append(f"{version} (released {_day(released)})" if released else version)
    extra = len(versions) - _MAX_LISTED
    return ", ".join(parts) + (f" and {extra} more" if extra > 0 else "")


def _identifiers(vuln: Dict[str, Any]) -> List[str]:
    found = list(vuln.get("cve_ids") or []) + list(vuln.get("ghsa_ids") or [])
    if not found:
        found = [i for i in (vuln.get("id"), vuln.get("osv_id")) if i]
    return list(dict.fromkeys(found))


def _label(vuln: Dict[str, Any]) -> str:
    return " / ".join(_identifiers(vuln)) or "unidentified vulnerability record"


def _is_minor_line_expression(expr: str) -> bool:
    clean = clean_version_string(expr)
    if re.match(r"^\d+\.\d+([\.\*xX]+)?$", clean):
        return True
    if clean.startswith("~=") or "==" in clean:
        parts = re.findall(r"\d+", clean)
        return len(parts) >= 2
    return False


def _release_line_key(parsed: Version, expressions: Sequence[str]) -> Tuple[int, ...]:
    """Return (major, minor) if covered by an explicit minor release line expression, else (major,)."""
    for expr in expressions:
        if _is_minor_line_expression(expr):
            specs = VersionNormalizer.resolve_specifiers([expr], [])
            if any(parsed in spec for spec in specs):
                return (parsed.major, parsed.minor)
    return (parsed.major,)


def _remediated(version: str, fixes: Sequence[Version], expressions: Sequence[str] = ()) -> bool:
    """True when a higher fixed release exists on the same release line (major or minor line)."""
    parsed = _parse_version(version)
    if parsed is None:
        return False
    line = _release_line_key(parsed, expressions)
    if len(line) == 2:
        return any(fix.major == line[0] and fix.minor == line[1] and fix > parsed for fix in fixes)
    return any(fix.major == line[0] and fix > parsed for fix in fixes)


def _single_interval_ranges(ranges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep ECOSYSTEM/SEMVER ranges and split multi-interval event lists into one interval each."""
    intervals: List[Dict[str, Any]] = []
    for entry in ranges:
        range_type = str(entry.get("type") or "ECOSYSTEM").upper()
        if range_type not in _VERSION_RANGE_TYPES:
            continue
        current: List[Dict[str, Any]] = []
        for event in entry.get("events") or []:
            if "introduced" in event and current:
                intervals.append({"type": range_type, "events": current})
                current = []
            current.append(event)
        if current:
            intervals.append({"type": range_type, "events": current})
    return intervals


# ---------------------------------------------------------------------------
# Evaluation context
# ---------------------------------------------------------------------------

@dataclass
class _Context:
    """Per-evaluation data shared by the per-vulnerability checks."""

    expressions: List[str]
    releases: List[Dict[str, Any]]
    temporal_reference: Optional[str]
    now: datetime
    release_dates: Dict[Version, datetime] = field(default_factory=dict)
    time_dependent: bool = False
    _scopes: Dict[Optional[datetime], NormalizedVersionScope] = field(default_factory=dict)
    _all_scope: Optional[NormalizedVersionScope] = None

    def scope_at(self, moment: Optional[datetime]) -> NormalizedVersionScope:
        """Normalised version scope with relative terms resolved at `moment` (cached)."""
        if moment not in self._scopes:
            self._scopes[moment] = NormalizedVersionScope(self.expressions, self.releases, moment)
        return self._scopes[moment]

    def all_versions_scope(self) -> NormalizedVersionScope:
        if self._all_scope is None:
            self._all_scope = NormalizedVersionScope(["all"], self.releases, None)
        return self._all_scope


def _release_dates(releases: Sequence[Dict[str, Any]]) -> Dict[Version, datetime]:
    dates: Dict[Version, datetime] = {}
    for release in releases:
        published = parse_iso_datetime(release.get("published_at"))
        if published is None:
            continue
        for key in (release.get("tag"), release.get("version")):
            parsed = _parse_version(key) if key else None
            if parsed is not None and (parsed not in dates or published < dates[parsed]):
                dates[parsed] = published
    return dates


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

class AssessmentRuleEngine:
    """Executes Algorithm 1 and returns a RuleAssessmentResult."""

    def evaluate(
        self,
        commitment: Dict[str, Any],
        vulnerabilities: Optional[List[Dict[str, Any]]],
        releases: Optional[List[Dict[str, Any]]],
        reference_time: Optional[datetime] = None,
        package_name: Optional[str] = None,
        source_diagnostics: Optional[Dict[str, Any]] = None,
        mapping_uncertain: bool = False,
    ) -> RuleAssessmentResult:
        """Assess `commitment` against V and R; reference_time is the evaluation clock (default: now, UTC)."""
        now = _as_utc(reference_time) if reference_time is not None else datetime.now(timezone.utc)
        commitment = commitment or {}
        vulns = list(vulnerabilities or [])
        base: Dict[str, Any] = {
            "package_name": package_name,
            "evaluated_vulns_count": len(vulns),
            "reasonable_timeframe_days": REASONABLE_TIMEFRAME_DAYS,
            "reference_time": now.isoformat(),
        }
        text = str(commitment.get("commitment_text") or "").strip()
        trail = [f'Commitment (t): "{text}"' if text else "Commitment (t): no policy statement supplied."]

        # Rule R5: measurability.
        expressions, unparsed, r5_reason = self._check_measurability(commitment)
        if r5_reason:
            return self._halt("R5", [r5_reason], trail, base, is_measurable=False)

        # Rule R4 pre-check: evidence sufficiency.
        r4_reasons = self._insufficiency_reasons(vulnerabilities, releases, source_diagnostics, mapping_uncertain)
        if r4_reasons:
            return self._halt("R4", r4_reasons, trail, base, is_measurable=True)

        temporal_reference = commitment.get("temporal_reference")
        ctx = _Context(
            expressions=expressions,
            releases=list(releases),
            temporal_reference=temporal_reference,
            now=now,
            release_dates=_release_dates(releases),
            time_dependent=any(e.lower() in _TEMPORAL_EXPRESSIONS for e in expressions),
        )
        trail.append(
            f"Version scope (v): {', '.join(expressions)}; commitment type {commitment.get('support_status')}; "
            f"temporal reference {temporal_reference}."
        )
        if unparsed:
            trail.append(f"Version token(s) that could not be normalised were ignored: {', '.join(unparsed)}.")

        # Per-vulnerability rules R1 to R4.
        entries: List[EvidenceLogEntry] = []
        out_of_scope: List[str] = []
        for vuln in vulns:
            entry = self._assess_vulnerability(vuln, ctx)
            if entry is None:
                out_of_scope.append(_label(vuln))
            else:
                entries.append(entry)
                trail.append(f"{entry.rule_id} ({entry.outcome}): {entry.message}")
        if out_of_scope:
            trail.append(
                f"{len(out_of_scope)} vulnerability record(s) affect no version covered by the commitment "
                f"and were not assessed: {_brief(out_of_scope)}."
            )

        # Aggregation (Table 3.1).
        counts = {rule: sum(1 for e in entries if e.rule_id == rule) for rule in ("R1", "R2", "R3", "R4")}
        if not entries:
            verdict = VerdictCategory.INSUFFICIENT_EVIDENCE
            message = (
                "No vulnerability record affects a version covered by the commitment, so the commitment was never "
                "tested (resolution of a case that Algorithm 1 leaves open)."
            )
            entries.append(EvidenceLogEntry(
                rule_id="R4", outcome=verdict.value, reason="no_vulnerability_in_scope", message=message,
            ))
            trail.append(f"R4 ({verdict.value}): {message}")
            triggered = ["R4"]
        else:
            verdict = self._aggregate(counts)
            triggered = [rule for rule in ("R1", "R2", "R3", "R4") if counts[rule]]
            summary = ", ".join(f"{counts[rule]} x {rule}" for rule in triggered)
            trail.append(f"Aggregation (Table 3.1): {summary} -> {verdict.value}.")

        return RuleAssessmentResult(
            verdict=verdict,
            is_measurable=True,
            rules_triggered=triggered,
            evidence_trail=trail,
            evidence_log=entries,
            assessed_vulns_count=sum(counts.values()),
            out_of_scope_vulns_count=len(out_of_scope),
            rule_counts=counts,
            **base,
        )

    # -- R5 and R4 pre-checks ------------------------------------------------

    @staticmethod
    def _check_measurability(commitment: Dict[str, Any]) -> Tuple[List[str], List[str], Optional[Tuple[str, str]]]:
        """Return (expressions, unparsed tokens, (reason code, message) when rule R5 applies)."""
        tau = commitment.get("support_status")
        if tau not in _COMMITMENT_TYPES or tau == "ambiguous":
            return [], [], (
                "ambiguous_commitment_type",
                f"The commitment type is '{tau or 'missing'}', so the statement cannot be mapped to concrete "
                "version identifiers.",
            )
        raw = commitment.get("version_expression") or []
        if isinstance(raw, str):
            raw = [raw]
        expressions = [str(token).strip() for token in raw if token is not None and str(token).strip()]
        if not expressions:
            expressions = {"latest_only": ["latest"], "all_versions": ["all"]}.get(tau, [])
        unparsed = [e for e in expressions if not VersionNormalizer.resolve_specifiers([e], _PROBE_RELEASES)]
        if commitment.get("is_measurable") is False or not expressions or len(unparsed) == len(expressions):
            return expressions, unparsed, (
                "version_scope_not_normalisable",
                f"The version expression {expressions} cannot be normalised into PEP 440 version specifiers.",
            )
        return expressions, unparsed, None

    @staticmethod
    def _insufficiency_reasons(
        vulnerabilities: Optional[List[Dict[str, Any]]],
        releases: Optional[List[Dict[str, Any]]],
        source_diagnostics: Optional[Dict[str, Any]],
        mapping_uncertain: bool,
    ) -> List[Tuple[str, str]]:
        reasons: List[Tuple[str, str]] = []
        if not vulnerabilities:
            reasons.append((
                "vulnerabilities_missing",
                "Vulnerability evidence V is missing or empty: no record exists against which the commitment "
                "can be tested.",
            ))
        if not releases:
            reasons.append(("releases_missing", "Release history R is missing or empty."))
        for source, diagnostic in sorted((source_diagnostics or {}).items()):
            # NVD only enriches records (publication dates); missing dates are handled per vulnerability.
            if source not in _RECORD_SOURCES:
                continue
            if isinstance(diagnostic, dict) and diagnostic.get("status") in _FAILED_SOURCE_STATUSES:
                detail = f": {diagnostic['error']}" if diagnostic.get("error") else ""
                reasons.append((
                    "source_failure",
                    f"Evidence source '{source}' reported {diagnostic['status']}{detail}, so V may be incomplete.",
                ))
        if mapping_uncertain:
            reasons.append((
                "mapping_uncertain",
                "The mapping of the evidence to the package was flagged as uncertain.",
            ))
        return reasons

    @staticmethod
    def _halt(
        rule_id: str,
        reasons: List[Tuple[str, str]],
        trail: List[str],
        base: Dict[str, Any],
        is_measurable: bool,
    ) -> RuleAssessmentResult:
        verdict = RULE_VERDICTS[rule_id]
        entries = [
            EvidenceLogEntry(rule_id=rule_id, outcome=verdict.value, reason=code, message=message)
            for code, message in reasons
        ]
        trail = trail + [f"{rule_id} ({verdict.value}): {message}" for _, message in reasons]
        return RuleAssessmentResult(
            verdict=verdict,
            is_measurable=is_measurable,
            rules_triggered=[rule_id],
            evidence_trail=trail,
            evidence_log=entries,
            **base,
        )

    # -- Per-vulnerability rules --------------------------------------------

    def _assess_vulnerability(self, vuln: Dict[str, Any], ctx: _Context) -> Optional[EvidenceLogEntry]:
        """Return the R1..R4 entry for one vulnerability, or None when it is out of scope."""
        identifiers = _identifiers(vuln)
        published = parse_iso_datetime(vuln.get("published_at"))
        label = _label(vuln)
        common: Dict[str, Any] = {
            "vulnerability_id": vuln.get("id") or (identifiers[0] if identifiers else None),
            "identifiers": identifiers,
            "published_at": published.isoformat() if published else None,
            "advisory_url": vuln.get("advisory_url"),
        }

        def entry(rule_id: str, reason: str, message: str, **extra: Any) -> EvidenceLogEntry:
            return EvidenceLogEntry(
                rule_id=rule_id, outcome=RULE_VERDICTS[rule_id].value, reason=reason,
                message=message, **common, **extra,
            )

        affected, mapped = self._affected_versions(vuln, ctx)
        if not mapped:
            return entry(
                "R4", "affected_versions_unmapped",
                f"{label} (published {_day(published)}): its affected versions cannot be mapped to known releases.",
            )

        covered, unresolved = self._covered_versions(vuln, affected, ctx, published)
        if not covered:
            if unresolved is None:
                return None
            return entry(
                "R4", unresolved,
                f"{label}: coverage at time T cannot be determined ({unresolved.replace('_', ' ')}).",
            )

        common["affected_in_scope"] = covered
        common["affected_release_dates"] = {
            version: ctx.release_dates[_parse_version(version)].isoformat()
            for version in covered if _parse_version(version) in ctx.release_dates
        }
        scope_text = f"{label} (published {_day(published)}) affects covered version(s) {_dated(covered, ctx.release_dates)}"
        fixed = [str(f) for f in (vuln.get("fixed_versions") or []) if f]

        if not fixed:
            if published is None:
                return entry(
                    "R4", "publication_date_missing",
                    f"{scope_text}; no fixed version is recorded and the publication date is missing, "
                    "so the elapsed time cannot be measured.",
                )
            elapsed = ctx.now - published
            days = max(math.ceil(elapsed.total_seconds() / 86400), 0)
            if elapsed > timedelta(days=REASONABLE_TIMEFRAME_DAYS):
                return entry(
                    "R3", "unfixed_beyond_timeframe",
                    f"{scope_text}; no fixed version is recorded {days} days after publication, beyond the "
                    f"{REASONABLE_TIMEFRAME_DAYS}-day reasonable timeframe.",
                    days_since_publication=days,
                )
            return entry(
                "R4", "unfixed_within_timeframe",
                f"{scope_text}; no fixed version is recorded, but only {days} days have elapsed since publication, "
                f"within the {REASONABLE_TIMEFRAME_DAYS}-day reasonable timeframe.",
                days_since_publication=days,
            )

        fixed_parsed = [(f, _parse_version(f)) for f in fixed]
        fixed_versions = [v for _, v in fixed_parsed if v is not None]
        if not fixed_versions:
            return entry(
                "R4", "fixed_identifiers_unmapped",
                f"{scope_text}; the fixed identifier(s) {_brief(fixed)} are not release versions and cannot be "
                "mapped to releases.",
                fixed_versions=fixed,
            )

        fix_dates = {f: ctx.release_dates[v].isoformat() for f, v in fixed_parsed if v in ctx.release_dates}
        fixed_text = _dated(fixed, ctx.release_dates)
        unremediated = [version for version in covered if not _remediated(version, fixed_versions, ctx.expressions)]
        fixed_fields = {"fixed_versions": fixed, "fix_release_dates": fix_dates}
        if unremediated:
            return entry(
                "R2", "fixed_on_some_release_lines",
                f"{scope_text}; fixed in {fixed_text}, but version(s) {_brief(unremediated)} have no later fixed "
                "release on their release line.",
                unremediated_versions=unremediated, **fixed_fields,
            )
        return entry("R1", "fixed_on_all_covered_versions", f"{scope_text}; fixed in {fixed_text}.", **fixed_fields)

    @staticmethod
    def _affected_versions(vuln: Dict[str, Any], ctx: _Context) -> Tuple[List[str], bool]:
        """Return (affected versions, mapped). Ranges are expanded to known releases when no list is given."""
        versions = [str(v) for v in (vuln.get("affected_versions") or []) if v]
        if versions:
            return versions, True
        ranges = _single_interval_ranges(vuln.get("affected_ranges") or [])
        if not ranges:
            return [], False
        _, expanded, uncertain = ctx.all_versions_scope().is_range_intersecting(ranges)
        return expanded, not uncertain

    @staticmethod
    def _covered_versions(
        vuln: Dict[str, Any], affected: List[str], ctx: _Context, published: Optional[datetime]
    ) -> Tuple[List[str], Optional[str]]:
        """Return (covered versions, reason code when coverage could not be determined)."""
        if not ctx.time_dependent:
            scope = ctx.scope_at(None)
            return [v for v in affected if scope.is_version_covered(v)], None
        if ctx.temporal_reference == "vulnerability_publication_time":
            if published is None:
                return [], "publication_date_missing"
            scope = ctx.scope_at(published)
            return [v for v in affected if scope.is_version_covered(v)], None
        if ctx.temporal_reference == "release_time":
            covered: List[str] = []
            missing = False
            for version in affected:
                parsed = _parse_version(version)
                if parsed is None:
                    continue
                released = ctx.release_dates.get(parsed)
                if released is None:
                    missing = True
                elif ctx.scope_at(released).is_version_covered(version):
                    covered.append(version)
            return covered, ("release_date_missing" if missing and not covered else None)
        return [], "temporal_reference_unknown"

    # -- Aggregation ----------------------------------------------------------

    @staticmethod
    def _aggregate(counts: Dict[str, int]) -> VerdictCategory:
        """Aggregate by highest severity (Algorithm 1, Table 3.1); see the module docstring."""
        if not any(counts.get(rule) for rule in ("R1", "R2", "R3", "R4")):
            return VerdictCategory.INSUFFICIENT_EVIDENCE
        if counts["R3"] and (counts["R1"] or counts["R2"]):
            return VerdictCategory.PARTIALLY_CONSISTENT
        if counts["R3"]:
            return VerdictCategory.POTENTIAL_INCONSISTENCY
        if counts["R2"]:
            return VerdictCategory.PARTIALLY_CONSISTENT
        if counts["R4"]:
            return VerdictCategory.INSUFFICIENT_EVIDENCE
        return VerdictCategory.CONSISTENT
