"""
SupportLens: end-to-end five-layer pipeline (thesis 3.1.3.4 and 3.2).

f(S, V, R) -> (y, epsilon): a PyPI package name passes through
  Layer 0  data acquisition and preprocessing (S, R),
  Layer 1  AI extraction of the commitment tuple c = <t, tau, v, T>,
  Layer 2  vulnerability evidence V and the normalised version scope,
  Layer 3  rule-based assessment (Algorithm 1) giving the verdict y,
  Layer 4  template-based explanation epsilon and the stored JSON record.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.layer0_data_acquisition import DataAcquisitionPipeline
from src.layer0_data_acquisition.policy_preprocessing import PolicyPreprocessor
from src.layer1_ai_extraction import QwenCommitmentExtractor
from src.layer2_evidence_layer import EvidenceCollector, NormalizedVersionScope
from src.layer3_rule_assessment import AssessmentRuleEngine
from src.layer4_explanation_output import FinalOutputBuilder


class SupportLensPipeline:
    """Runs Layers 0 to 4 for one PyPI package."""

    def __init__(self, github_token: Optional[str] = None, extractor: Any = None, strict: bool = False):
        self.layer0 = DataAcquisitionPipeline(github_token)
        self.layer1 = extractor if extractor is not None else QwenCommitmentExtractor(strict=strict)
        self.layer2 = EvidenceCollector()
        self.layer3 = AssessmentRuleEngine()
        self.layer4 = FinalOutputBuilder()
        self.github_token = github_token

    def audit_package(self, package_name: str, reference_time: Optional[datetime] = None) -> Dict[str, Any]:
        reference_time = reference_time or datetime.now(timezone.utc)

        l0 = self.layer0.process_package(package_name)
        if l0.get("status") == "EXCLUDED":
            return {
                "package_name": package_name,
                "status": "EXCLUDED",
                "exclusion_reason": l0.get("exclusion_reason"),
                "layer0": l0,
            }
        releases = l0.get("releases", [])

        # Layer 1 input is the Layer 0 preprocessed text (thesis 3.2.1/3.2.2), the same text that
        # is stored as policy_text in the dataset records used for annotation and training.
        policy_text = l0.get("preprocessed_text")
        if policy_text is None:
            policy_text = PolicyPreprocessor.clean_text(l0.get("raw_policy_text") or "")
        commitment = self.layer1.extract(policy_text).to_dict()

        # Algorithm 1 R4 "mapping_uncertain": a forked or archived repository may not be the
        # authoritative source of the package's policy.
        repo_meta = l0.get("repo_metadata") or {}
        mapping_uncertain = bool(repo_meta.get("fork")) or bool(repo_meta.get("archived"))

        evidence = self.layer2.collect_unified_evidence(package_name, pypi_releases=releases, github_token=self.github_token)
        vulns = evidence.get("vulnerabilities", [])
        scope = NormalizedVersionScope(commitment.get("version_expression", []), releases, reference_time)

        assessment = self.layer3.evaluate(
            commitment=commitment,
            vulnerabilities=vulns,
            releases=releases,
            reference_time=reference_time,
            package_name=package_name,
            source_diagnostics=evidence.get("source_diagnostics"),
            mapping_uncertain=mapping_uncertain,
        )

        traces = {
            "status": "INCLUDED",
            "reference_time": reference_time.isoformat(),
            "layer0": l0,
            "layer1": commitment,
            "mapping_uncertain": mapping_uncertain,
            "layer2": {
                "total_releases": len(releases),
                "total_vulnerabilities": len(vulns),
                "normalized_scope": [str(s) for s in scope.specifiers],
                "supported_releases_count": scope.get_supported_releases_count(),
                "source_diagnostics": evidence.get("source_diagnostics", {}),
                "vulnerabilities": vulns,
            },
            "layer3": assessment.to_dict(),
        }
        package_info = {
            "package_name": package_name,
            "github_repo": l0.get("github_repo"),
            "security_policy_url": l0.get("security_policy_url"),
            "security_policy_path": l0.get("security_policy_path"),
            "line_references": l0.get("line_references", {}),
            "version_section_detail": l0.get("version_section_detail", {}),
        }
        source_refs = [
            f"PyPI JSON API: https://pypi.org/pypi/{package_name}/json ({len(releases)} releases)",
            f"OSV / GitHub Advisory / NVD: {len(vulns)} de-duplicated vulnerability records",
            f"Security policy: {l0.get('security_policy_url')}",
        ]
        result = self.layer4.build_and_save(
            assessment_result=assessment,
            package_info=package_info,
            source_references=source_refs,
            layer1_commitment=commitment,
            layer2_evidence={"total_releases": len(releases), "total_vulnerabilities": len(vulns), "layer_traces": traces},
        )
        result.update(traces)
        return result
