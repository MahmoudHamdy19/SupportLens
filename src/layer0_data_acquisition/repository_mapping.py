"""
Layer 0: Data Acquisition and Preprocessing
Component: Repository Mapping
=========================================
Performs:
1. Strict URL Domain Scope Verification:
   - Validates that project URLs point strictly to github.com
   - Extracts canonical owner/repo while rejecting spoofed or malformed domains
2. GitHub API Verification & Default Branch Discovery (thesis 3.2.1, tool: PyGithub):
   - Validates repository existence via the GitHub API
   - Detects repository renames, visibility, and status
   - Retrieves the repository's ACTUAL default branch (e.g. main, master, trunk)
   - Returns repository metadata (stars, creation and push dates) used by the robustness analysis (RQ3)
"""

import os
import re
import logging
from urllib.parse import urlparse
from typing import Dict, Any, Optional, Tuple

log = logging.getLogger(__name__)


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value else None)

class RepositoryMapper:
    """Identifies, validates domain scope, and verifies canonical GitHub repository via API."""

    PRIORITY_KEYS = [
        "Source", "Source Code", "Repository", "GitHub", "Code",
        "Home", "Homepage", "Bug Tracker", "Issues", "Documentation"
    ]

    RESERVED_GITHUB_NAMES = {
        "features", "topics", "site", "organizations", "marketplace",
        "pulls", "issues", "settings", "explore", "search", "pricing",
        "about", "enterprise", "security", "login", "join"
    }

    @classmethod
    def validate_github_url_scope(cls, url: str) -> Optional[Tuple[str, str]]:
        """
        Strict URL domain scope check:
        Ensures the host is strictly github.com and extracts (owner, repo).
        Rejects non-github domains, lookalikes (e.g. github.com.evil.com), and reserved paths.
        """
        if not url or not isinstance(url, str):
            return None

        url_clean = url.strip()
        # Add scheme if missing for proper parsing
        if not url_clean.startswith(("http://", "https://")):
            url_clean = "https://" + url_clean

        try:
            parsed = urlparse(url_clean)
        except Exception:
            return None

        # Check domain scope: must be exactly github.com or www.github.com
        hostname = (parsed.hostname or "").lower()
        if hostname not in ["github.com", "www.github.com"]:
            return None

        # Path must be in the format /owner/repo[/...]
        path_segments = [seg for seg in parsed.path.strip("/").split("/") if seg]
        if len(path_segments) < 2:
            return None

        owner, repo = path_segments[0], path_segments[1]

        # Sanitize repo name: remove .git suffix and query/fragment leftovers
        repo = repo.removesuffix(".git").strip()

        # Validate owner and repo syntax
        valid_ident = re.compile(r"^[A-Za-z0-9_.-]+$")
        if not valid_ident.match(owner) or not valid_ident.match(repo):
            return None

        if owner.lower() in cls.RESERVED_GITHUB_NAMES:
            return None

        return owner, repo

    @classmethod
    def verify_via_github_api(
        cls,
        owner: str,
        repo: str,
        github_token: Optional[str] = None,
        timeout: int = 10,
        max_retries: int = 3
    ) -> Tuple[bool, Optional[str], str, Optional[Dict[str, Any]], str]:
        """
        Verifies repository existence and the actual default branch through the GitHub API (PyGithub).
        Distinguishes VERIFIED, NOT_FOUND (404), and VERIFICATION_FAILED (rate limit, network, other errors).
        Returns: (is_valid, default_branch, message, repo_metadata, verification_status)
        """
        from github import Auth, Github, GithubException, UnknownObjectException

        token = github_token or os.environ.get("GITHUB_TOKEN")
        client = Github(
            auth=Auth.Token(token) if token else None,
            timeout=timeout,
            retry=max_retries,
            user_agent="SupportLens-Layer0-RepoMapper",
        )
        try:
            gh_repo = client.get_repo(f"{owner}/{repo}")
            meta = {
                "full_name": gh_repo.full_name,
                "default_branch": gh_repo.default_branch,
                "stars": gh_repo.stargazers_count,
                "forks": gh_repo.forks_count,
                "created_at": _iso(gh_repo.created_at),
                "pushed_at": _iso(gh_repo.pushed_at),
                "archived": gh_repo.archived,
                "fork": gh_repo.fork,
                "html_url": gh_repo.html_url,
            }
            msg = f"Verified via GitHub API: {meta['full_name']} (default branch: '{meta['default_branch']}')"
            return True, meta["default_branch"], msg, meta, "VERIFIED"
        except UnknownObjectException:
            return False, None, f"EXCLUDED: GitHub repository '{owner}/{repo}' does not exist or is private (HTTP 404).", None, "NOT_FOUND"
        except GithubException as e:
            reason = f"GitHub API error HTTP {e.status}"
        except Exception as e:  # network errors raised by the underlying HTTP client
            reason = f"Connection/Network error: {e}"
        finally:
            client.close()
        fail_msg = f"EXCLUDED: GitHub verification could not be confirmed for '{owner}/{repo}' ({reason}). Branch and repository unverified."
        return False, None, fail_msg, None, "VERIFICATION_FAILED"

    @classmethod
    def map_repository(
        cls,
        pypi_data: Dict[str, Any],
        verify_api: bool = True,
        github_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete mapping workflow:
        1. Parse candidate URLs from PyPI metadata.
        2. Strictly validate domain scope (github.com).
        3. Verify existence and obtain actual default_branch via GitHub API.
        """
        info = pypi_data.get("info", {})
        urls = info.get("project_urls") or {}

        candidates = []
        for key in cls.PRIORITY_KEYS:
            if key in urls and urls[key]:
                candidates.append((key, urls[key]))

        if info.get("home_page"):
            candidates.append(("home_page", info["home_page"]))

        for k, v in urls.items():
            if k not in cls.PRIORITY_KEYS and v:
                candidates.append((k, v))

        checked_urls = []
        for source_key, url in candidates:
            checked_urls.append(url)
            parsed_scope = cls.validate_github_url_scope(url)
            if not parsed_scope:
                continue

            owner, repo = parsed_scope
            repo_candidate = f"{owner}/{repo}"

            if verify_api:
                is_valid, default_branch, evidence, meta, v_status = cls.verify_via_github_api(
                    owner, repo, github_token=github_token
                )
                if is_valid:
                    return {
                        "is_valid": True,
                        "verification_status": v_status,
                        "github_repo": meta.get("full_name", repo_candidate) if meta else repo_candidate,
                        "default_branch": default_branch or "main",
                        "evidence": f"Verified via PyPI project_urls['{source_key}'] -> {url} | {evidence}",
                        "metadata": meta
                    }
                else:
                    # If verification failed due to rate limits or API outage, don't silently pretend it's valid
                    if v_status == "VERIFICATION_FAILED":
                        return {
                            "is_valid": False,
                            "verification_status": "VERIFICATION_FAILED",
                            "github_repo": repo_candidate,
                            "default_branch": None,
                            "evidence": evidence,
                            "metadata": None
                        }
                    # If 404, check next candidate URL
                    continue
            else:
                return {
                    "is_valid": True,
                    "verification_status": "DOMAIN_ONLY",
                    "github_repo": repo_candidate,
                    "default_branch": "main",
                    "evidence": f"Verified via PyPI project_urls['{source_key}'] -> {url} (domain verified)",
                    "metadata": None
                }

        return {
            "is_valid": False,
            "verification_status": "NOT_FOUND",
            "github_repo": None,
            "default_branch": None,
            "evidence": "EXCLUDED: No verifiable GitHub repository link found in PyPI metadata.",
            "metadata": None
        }

    @classmethod
    def identify_github_repo(
        cls,
        pypi_data: Dict[str, Any],
        verify_api: bool = True,
        github_token: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Backward-compatible interface returning (repo_full_name, link_evidence_source).
        """
        res = cls.map_repository(pypi_data, verify_api=verify_api, github_token=github_token)
        if res["is_valid"]:
            return res["github_repo"], res["evidence"]
        return None, res["evidence"]
