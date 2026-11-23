"""
Layer 4: Explanation and Output Layer
Component: Final output builder (thesis Section 3.2.5)
=======================================================
Builds the structured JSON record (verdict y, template-generated explanation, source
references and the full trace of Layers 0 to 3) and stores it through ResultStorage.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .explanation_generation import ExplanationGenerator
from .result_storage import ResultStorage

RECORD_SCHEMA_VERSION = 1


class FinalOutputBuilder:
    """Builds and stores the final auditable record of one assessment."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.explainer = ExplanationGenerator()
        self.storage = ResultStorage(output_dir)

    def build_and_save(
        self,
        assessment_result: Any,
        package_info: Optional[Dict[str, Any]] = None,
        source_references: Optional[List[str]] = None,
        layer1_commitment: Optional[Dict[str, Any]] = None,
        layer2_evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate the explanation, store the record under data/reports/ and return it.

        `package_info` may carry github_repo, security_policy_url, security_policy_path,
        line_references and version_section_detail (Layer 0). `layer2_evidence` may carry
        layer_traces, a dictionary of per-layer traces (layer0, layer2, ...) stored under
        `trace`; the commitment (Layer 1) and the assessment (Layer 3) have their own keys.
        """
        assessment = assessment_result.to_dict() if hasattr(assessment_result, "to_dict") else dict(assessment_result)
        package_info = package_info or {}
        commitment = layer1_commitment or {}
        layer2_evidence = layer2_evidence or {}
        package_name = assessment.get("package_name") or package_info.get("package_name")

        policy = self._policy_provenance(package_info)
        sources = self._collect_sources(package_name, package_info, policy, assessment, source_references)
        explanation = self.explainer.generate_explanation(
            assessment, commitment, policy["citation"], sources
        )

        layer_traces = layer2_evidence.get("layer_traces") or {}
        layer2_summary = {k: v for k, v in layer2_evidence.items() if k != "layer_traces"}
        # Layer 1 and Layer 3 are stored as `commitment` and `assessment`; the trace holds the other layers.
        trace: Dict[str, Any] = {k: v for k, v in layer_traces.items() if k not in ("layer1", "layer3")}
        if layer2_summary and "layer2" not in trace:
            trace["layer2"] = layer2_summary

        record: Dict[str, Any] = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "package_name": package_name,
            "verdict": assessment["verdict"],
            "explanation": explanation,
            "source_references": sources,
            "policy": policy,
            "commitment": commitment,
            "assessment": assessment,
            "trace": trace,
        }
        stored = self.storage.save_report(record)
        record = {"report_id": stored.report_id, "created_at": stored.created_at, **record}
        record["saved_report_path"] = str(stored.path)
        return record

    @staticmethod
    def _policy_lines(package_info: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        """Line range of the supported-versions section of SECURITY.md, if Layer 0 located it."""
        detail = package_info.get("version_section_detail") or {}
        if detail.get("line_start"):
            return detail["line_start"], detail.get("line_end") or detail["line_start"]
        for heading, reference in (package_info.get("line_references") or {}).items():
            if not isinstance(reference, dict) or not reference.get("line_start"):
                continue
            if reference.get("section_type") == "supported_versions" or \
                    str(heading).lower().replace(" ", "_") == "supported_versions":
                return reference["line_start"], reference.get("line_end") or reference["line_start"]
        return None, None

    def _policy_provenance(self, package_info: Dict[str, Any]) -> Dict[str, Any]:
        url = package_info.get("security_policy_url")
        path = package_info.get("security_policy_path")
        start, end = self._policy_lines(package_info)
        annotated = f"{url}#L{start}-L{end}" if url and start else url
        citation = path or "SECURITY.md"
        if start:
            citation += f", supported-versions section, lines {start}-{end}"
        if annotated:
            citation += f", {annotated}"
        return {
            "repository": package_info.get("github_repo"),
            "path": path,
            "url": url,
            "line_start": start,
            "line_end": end,
            "line_annotated_url": annotated,
            "citation": citation,
        }

    @staticmethod
    def _collect_sources(
        package_name: Optional[str],
        package_info: Dict[str, Any],
        policy: Dict[str, Any],
        assessment: Dict[str, Any],
        source_references: Optional[List[str]],
    ) -> List[str]:
        """Policy, repository, PyPI page, advisory URLs and caller-supplied references, without duplicates."""
        sources: List[str] = []
        if policy["line_annotated_url"]:
            sources.append(policy["line_annotated_url"])
        if package_info.get("github_repo"):
            sources.append(f"https://github.com/{package_info['github_repo']}")
        if package_name:
            sources.append(f"https://pypi.org/project/{package_name}/")
        sources.extend(e["advisory_url"] for e in assessment.get("evidence_log") or [] if e.get("advisory_url"))
        sources.extend(source_references or [])
        return list(dict.fromkeys(sources))
