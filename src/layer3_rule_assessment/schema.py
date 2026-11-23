"""
Layer 3: Rule-Based Assessment Layer
Component: Result schema
====================================
Pydantic models for the Layer 3 output (thesis Section 3.2.4: assessment outcome y and
evidence trail E). The models carry no confidence value or ratio: the verdict follows
deterministically from the rule outcomes and the counts below.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .verdict_categories import VerdictCategory


class EvidenceLogEntry(BaseModel):
    """Structured justification of one rule outcome (one vulnerability, or one halt reason)."""

    rule_id: str = Field(..., description="One of R1, R2, R3, R4, R5")
    outcome: str = Field(..., description="Verdict category name associated with the rule")
    reason: str = Field(..., description="Machine-readable reason code, for example 'unfixed_beyond_timeframe'")
    message: str = Field(..., description="Human-readable justification built only from the evidence used")
    vulnerability_id: Optional[str] = Field(None, description="Canonical CVE or GHSA identifier of the record")
    identifiers: List[str] = Field(default_factory=list, description="All CVE and GHSA identifiers of the record")
    published_at: Optional[str] = Field(None, description="Vulnerability publication time (ISO 8601), None if missing")
    affected_in_scope: List[str] = Field(default_factory=list, description="Affected versions covered by the commitment")
    affected_release_dates: Dict[str, str] = Field(default_factory=dict, description="Release date (ISO 8601) per covered version, where known")
    unremediated_versions: List[str] = Field(default_factory=list, description="Covered versions without a later fix on their release line")
    fixed_versions: List[str] = Field(default_factory=list, description="Fixed versions listed by the evidence")
    fix_release_dates: Dict[str, str] = Field(default_factory=dict, description="Release date (ISO 8601) per fixed version, where known")
    days_since_publication: Optional[int] = Field(None, description="Whole days from publication to the reference time (unfixed records only)")
    advisory_url: Optional[str] = Field(None, description="Advisory URL when the evidence provides one")


class RuleAssessmentResult(BaseModel):
    """Layer 3 output: verdict y, evidence trail E and per-rule counts."""

    package_name: Optional[str] = Field(None, description="PyPI package name, if supplied")
    verdict: VerdictCategory = Field(..., description="One of the five categories of Table 3.1")
    is_measurable: bool = Field(..., description="False only when rule R5 halted the assessment")
    rules_triggered: List[str] = Field(default_factory=list, description="Rule identifiers (R1..R5) that fired at least once")
    evidence_trail: List[str] = Field(default_factory=list, description="Ordered statements naming the ids, versions and dates used")
    evidence_log: List[EvidenceLogEntry] = Field(default_factory=list, description="Structured counterpart of the evidence trail")

    evaluated_vulns_count: int = Field(0, description="Number of vulnerability records supplied (|V|)")
    assessed_vulns_count: int = Field(0, description="Records that received a per-vulnerability outcome R1 to R4")
    out_of_scope_vulns_count: int = Field(0, description="Records not affecting any version covered by the commitment")
    rule_counts: Dict[str, int] = Field(
        default_factory=lambda: {"R1": 0, "R2": 0, "R3": 0, "R4": 0},
        description="Number of per-vulnerability outcomes per rule",
    )
    reasonable_timeframe_days: int = Field(..., description="Reasonable remediation timeframe used by rule R3, in days")
    reference_time: str = Field(..., description="Evaluation clock (ISO 8601) used to compute elapsed time")

    def __getitem__(self, item: str) -> Any:
        return self.to_dict()[item]

    def get(self, item: str, default: Any = None) -> Any:
        return self.to_dict().get(item, default)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary with the verdict as its string value."""
        data = self.model_dump()
        data["verdict"] = self.verdict.value
        return data
