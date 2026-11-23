"""
Layer 4: Explanation and Output Layer
Component: Explanation generation (thesis Section 3.2.5)
=========================================================
The explanation is produced only by filling a Jinja2 template with facts taken from the
Layer 3 evidence trail, the commitment and the source references. There is no free-text
generation and no fallback text: a missing template or an undefined variable raises an error.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import jinja2

from src.layer3_rule_assessment.verdict_categories import VERDICT_CATEGORIES

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE_NAME = "explanation_template.j2"
LIST_LIMIT = 10
_MAX_LISTED_VERSIONS = 5


def _date_only(value: Optional[str]) -> str:
    """Date part (YYYY-MM-DD) of an ISO 8601 string, or a statement that it is unknown."""
    return str(value)[:10] if value else "date unknown"


def _brief(items: Optional[List[str]]) -> str:
    items = list(items or [])
    if len(items) <= _MAX_LISTED_VERSIONS:
        return ", ".join(items)
    return f"{', '.join(items[:_MAX_LISTED_VERSIONS])} and {len(items) - _MAX_LISTED_VERSIONS} more"


class ExplanationGenerator:
    """Renders the evidence-grounded explanation for one assessment."""

    def __init__(self, template_dir: Optional[Path] = None):
        self.environment = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir or TEMPLATE_DIR)),
            undefined=jinja2.StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.environment.filters["date_only"] = _date_only
        self.environment.filters["brief"] = _brief

    def generate_explanation(
        self,
        assessment: Any,
        commitment: Optional[Dict[str, Any]] = None,
        policy_citation: Optional[str] = None,
        source_references: Optional[List[str]] = None,
    ) -> str:
        """Render the explanation for `assessment` (RuleAssessmentResult or its dictionary)."""
        data = assessment.to_dict() if hasattr(assessment, "to_dict") else dict(assessment)
        verdict = data.get("verdict")
        if verdict not in VERDICT_CATEGORIES:
            raise ValueError(f"Unknown verdict: {verdict!r}")
        commitment = commitment or {}

        entries = [e.model_dump() if hasattr(e, "model_dump") else dict(e) for e in data.get("evidence_log") or []]
        vulnerability_entries = [e for e in entries if e.get("identifiers") or e.get("vulnerability_id")]
        package_entries = [e for e in entries if not (e.get("identifiers") or e.get("vulnerability_id"))]
        by_rule = {rule: [e for e in vulnerability_entries if e["rule_id"] == rule] for rule in ("R1", "R2", "R3", "R4")}
        counts = {rule: len(items) for rule, items in by_rule.items()}

        expression = commitment.get("version_expression") or []
        context = {
            "verdict": verdict,
            "package_name": data.get("package_name") or "not specified",
            "commitment_text": " ".join(str(commitment.get("commitment_text") or "no policy statement supplied").split()),
            "support_status": commitment.get("support_status") or "not stated",
            "version_expression": [expression] if isinstance(expression, str) else list(expression),
            "temporal_reference": commitment.get("temporal_reference") or "not stated",
            "policy_citation": policy_citation or "SECURITY.md",
            "reference_time": data.get("reference_time"),
            "timeframe_days": data.get("reasonable_timeframe_days"),
            "assessed": data.get("assessed_vulns_count", len(vulnerability_entries)),
            "out_of_scope": data.get("out_of_scope_vulns_count", 0),
            "counts": counts,
            "package_entries": package_entries,
            "source_references": list(source_references or []),
            "limit": LIST_LIMIT,
            **{rule.lower(): items for rule, items in by_rule.items()},
        }
        template = self.environment.get_template(TEMPLATE_NAME)
        return re.sub(r"\n{3,}", "\n\n", template.render(**context)).strip()
