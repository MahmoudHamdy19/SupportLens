"""
Layer 0: Data Acquisition and Preprocessing
Component: SECURITY.md Retrieval
=========================================
Performs:
1. Dynamic Branch Resolution:
   - Uses the repository's actual default branch (default_branch) first.
   - Falls back to common branches (main, master).
2. Distinction between Missing File vs. Access Failure:
   - MISSING_FILE: When the file legitimately returns HTTP 404 across candidate locations.
   - ACCESS_FAILURE: When blocked by HTTP 403 (rate-limit), 5xx (server error), or timeout.
   - Auditable diagnostic logging for exact root-cause tracking.
"""

import os
import logging
import requests
from typing import Optional, Tuple, Dict, Any, List

log = logging.getLogger(__name__)

class SecurityPolicyRetriever:
    """Retrieves SECURITY.md policies from GitHub with rigorous status diagnostics."""

    CANDIDATE_PATHS = [
        ".github/SECURITY.md",
        "SECURITY.md",
        "docs/SECURITY.md",
        ".github/security.md",
        "security.md",
        "docs/security.md"
    ]

    def __init__(self, github_token: Optional[str] = None, timeout: int = 10):
        token = github_token or os.environ.get("GITHUB_TOKEN")
        self.timeout = timeout
        self.headers = {"User-Agent": "SupportLens-Layer0-PolicyRetriever/2.0"}
        if token:
            self.headers["Authorization"] = f"token {token}"
            self.headers["Accept"] = "application/vnd.github.v3+json"
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def retrieve_policy_detailed(
        self,
        repo_full_name: str,
        default_branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves raw SECURITY.md content while using the repository's actual default branch
        and distinguishing between missing file (404) vs. access failure (403, 5xx, timeout).
        """
        # Determine candidate branches, prioritizing the actual default_branch
        branches_to_try: List[str] = []
        if default_branch and default_branch.strip():
            branches_to_try.append(default_branch.strip())
        else:
            for fallback in ["main", "master"]:
                branches_to_try.append(fallback)

        attempts_log = []
        encountered_access_failures = []
        all_404 = True

        for branch in branches_to_try:
            for path in self.CANDIDATE_PATHS:
                raw_url = f"https://raw.githubusercontent.com/{repo_full_name}/{branch}/{path}"
                try:
                    resp = self.session.get(raw_url, timeout=self.timeout)
                    attempts_log.append(f"{branch}/{path} -> HTTP {resp.status_code}")

                    if resp.status_code == 200:
                        content = resp.text
                        if len(content.strip()) > 10:
                            view_url = f"https://github.com/{repo_full_name}/blob/{branch}/{path}"
                            return {
                                "content": content,
                                "path": path,
                                "url": view_url,
                                "branch": branch,
                                "retrieval_status": "FOUND",
                                "retrieval_reason": f"Successfully retrieved from {branch}/{path}",
                                "attempts": attempts_log
                            }
                    elif resp.status_code == 404:
                        # Legitimate not found
                        pass
                    elif resp.status_code in [403, 429]:
                        all_404 = False
                        encountered_access_failures.append(f"HTTP {resp.status_code} (Rate Limit or Forbidden)")
                    elif resp.status_code >= 500:
                        all_404 = False
                        encountered_access_failures.append(f"HTTP {resp.status_code} (GitHub Server Error)")
                    else:
                        all_404 = False
                        encountered_access_failures.append(f"HTTP {resp.status_code}")

                except requests.Timeout:
                    all_404 = False
                    encountered_access_failures.append(f"Timeout connecting to {raw_url}")
                except requests.RequestException as e:
                    all_404 = False
                    encountered_access_failures.append(f"Network error: {str(e)}")

        # Distinguish root cause
        if encountered_access_failures and all_404 is False:
            reason = f"ACCESS_FAILURE: Failed to reach GitHub ({'; '.join(set(encountered_access_failures))})"
            status = "ACCESS_FAILURE"
        else:
            reason = f"MISSING_FILE: No SECURITY.md found across branches [{', '.join(branches_to_try)}] and standard paths (HTTP 404)"
            status = "MISSING_FILE"

        return {
            "content": None,
            "path": None,
            "url": None,
            "branch": None,
            "retrieval_status": status,
            "retrieval_reason": reason,
            "attempts": attempts_log
        }

    def retrieve_security_policy(
        self,
        repo_full_name: str,
        default_branch: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Backward-compatible signature: returns (raw_content, file_path_in_repo, html_url).
        """
        res = self.retrieve_policy_detailed(repo_full_name, default_branch=default_branch)
        return res["content"], res["path"], res["url"]
