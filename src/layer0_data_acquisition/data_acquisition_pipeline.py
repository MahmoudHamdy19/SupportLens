"""
Layer 0: Data Acquisition and Preprocessing Pipeline
====================================================
Processes:
  1. Input: PyPI package name
  2. PyPI Metadata: Package metadata, Release versions, Release dates (Missing kept as None/Unknown)
  3. Repository Mapping: Strict domain scope check, GitHub API verification, default branch extraction
  4. SECURITY.md Retrieval: Dynamic default branch priority, distinct MISSING_FILE vs ACCESS_FAILURE
  5. Policy Preprocessing: Parse Markdown, extract version support, segment text with line references
Outputs:
  Preprocessed text, structured sections, line references, and verified package metadata for Layer 1.
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from .pypi_metadata import PyPIMetadataCollector
from .repository_mapping import RepositoryMapper
from .security_md_retrieval import SecurityPolicyRetriever
from .policy_preprocessing import PolicyPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXCLUSION_LOG_PATH = PROJECT_ROOT / "data" / "audit" / "exclusion_log.jsonl"

class DataAcquisitionPipeline:
    """Executes the Layer 0 data collection and preprocessing pipeline."""

    def __init__(self, github_token: Optional[str] = None, exclusion_log_path: Optional[str] = None):
        self.pypi_collector = PyPIMetadataCollector()
        self.policy_retriever = SecurityPolicyRetriever(github_token=github_token)
        self.github_token = github_token
        self.exclusion_log_path = Path(exclusion_log_path) if exclusion_log_path else EXCLUSION_LOG_PATH

    def _record_exclusion(self, package_name: str, reason: str, retrieval_status: Optional[str] = None, repo: Optional[str] = None):
        """Persists the exclusion reason to the audit log (thesis 3.2.1, missing-data handling)."""
        self.exclusion_log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "package_name": package_name,
            "github_repo": repo,
            "retrieval_status": retrieval_status,
            "exclusion_reason": reason
        }
        with open(self.exclusion_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def process_package(self, package_name: str, require_version_support: bool = False) -> Dict[str, Any]:
        """
        Takes a PyPI package name, extracts metadata, maps repo, downloads and preprocesses policy.
        Optionally enforces the Section 3.3.1 version-support statement gate for research cohort.
        """
        result = {
            "package_name": package_name,
            "status": "PENDING",
            "exclusion_reason": None,
            "pypi_metadata": {},
            "releases": [],
            "github_repo": None,
            "default_branch": None,
            "repo_verification": None,
            "security_policy_path": None,
            "security_policy_url": None,
            "retrieval_status": None,
            "retrieval_reason": None,
            "raw_policy_text": None,
            "preprocessed_text": None,
            "prioritized_text": None,
            "policy_sections": {},
            "line_references": {},
            "version_section_detail": {},
            "has_version_support_statement": False,
            "repo_metadata": {},
            "policy_word_count": None
        }

        # Step 2: Fetch PyPI Metadata & Release Timeline (Requirement 6: missing dates kept as None/Unknown)
        pypi_data, pypi_err = self.pypi_collector.fetch_metadata(package_name)
        if not pypi_data:
            result["status"] = "EXCLUDED"
            result["exclusion_reason"] = pypi_err
            self._record_exclusion(package_name, pypi_err)
            return result

        result["pypi_metadata"] = pypi_data.get("info", {})
        result["releases"] = self.pypi_collector.extract_releases(pypi_data)

        # Step 3: Repository Mapping (Requirement 1 & 2: Domain scope check + GitHub API verification)
        repo_map = RepositoryMapper.map_repository(
            pypi_data,
            verify_api=True,
            github_token=self.github_token
        )
        if not repo_map["is_valid"]:
            result["status"] = "EXCLUDED"
            result["exclusion_reason"] = repo_map["evidence"]
            self._record_exclusion(package_name, repo_map["evidence"])
            return result

        repo_full = repo_map["github_repo"]
        default_branch = repo_map["default_branch"] or "main"
        repo_meta = repo_map.get("metadata") or {}
        result["github_repo"] = repo_full
        result["default_branch"] = default_branch
        result["repo_verification"] = repo_map["evidence"]
        result["repo_metadata"] = {
            "stars": repo_meta.get("stars"),
            "created_at": repo_meta.get("created_at"),
            "pushed_at": repo_meta.get("pushed_at"),
            "archived": repo_meta.get("archived"),
            "fork": repo_meta.get("fork"),
            "html_url": repo_meta.get("html_url"),
        }

        # Step 4: SECURITY.md Retrieval (Requirement 2 & 3: Actual default branch + Missing file vs Access failure)
        retrieval = self.policy_retriever.retrieve_policy_detailed(
            repo_full_name=repo_full,
            default_branch=default_branch
        )
        result["retrieval_status"] = retrieval["retrieval_status"]
        result["retrieval_reason"] = retrieval["retrieval_reason"]

        if retrieval["retrieval_status"] != "FOUND":
            result["status"] = "EXCLUDED"
            # Explicit distinction between missing file and access failure
            if retrieval["retrieval_status"] == "ACCESS_FAILURE":
                reason = f"EXCLUDED: {retrieval['retrieval_reason']}"
            else:
                reason = f"EXCLUDED: No SECURITY.md found in repository '{repo_full}' (HTTP 404 across candidate paths)."
            result["exclusion_reason"] = reason
            self._record_exclusion(package_name, reason, retrieval["retrieval_status"], repo_full)
            return result

        raw_text = retrieval["content"]
        result["raw_policy_text"] = raw_text
        result["security_policy_path"] = retrieval["path"]
        result["security_policy_url"] = retrieval["url"]

        # Step 5: Policy Preprocessing (Requirement 4 & 5: Markdown parsing, version sections, line references)
        cleaned_text = PolicyPreprocessor.clean_text(raw_text)
        sections = PolicyPreprocessor.extract_relevant_sections(raw_text)
        prioritized_text = PolicyPreprocessor.prioritize_version_support(raw_text)

        has_version_support = bool(
            sections.get("version_section_detail", {}).get("found", False)
            or sections.get("supported_versions", "").strip()
        )

        result["preprocessed_text"] = cleaned_text
        result["prioritized_text"] = prioritized_text
        result["policy_sections"] = sections
        result["line_references"] = sections.get("line_references", {})
        result["version_section_detail"] = sections.get("version_section_detail", {})
        result["has_version_support_statement"] = has_version_support
        result["policy_word_count"] = len(raw_text.split())

        if require_version_support and not has_version_support:
            result["status"] = "EXCLUDED"
            reason = f"EXCLUDED: SECURITY.md in repository '{repo_full}' contains no explicit version-support statement (Section 3.3.1 Gate)."
            result["exclusion_reason"] = reason
            self._record_exclusion(package_name, reason, "NO_VERSION_SUPPORT_STATEMENT", repo_full)
            return result

        result["status"] = "INCLUDED"

        return result
