"""
Layer 0: Data Acquisition and Preprocessing
Component: PyPI Metadata
=========================================
Extracts:
- Package metadata
- Release versions
- Release dates (Preserves missing dates as None/Unknown; never substitutes current date)
"""

import requests
from typing import Dict, List, Any, Optional, Tuple

PYPI_API_BASE = "https://pypi.org/pypi"

class PyPIMetadataCollector:
    """Queries PyPI JSON API for package metadata and chronological release timeline."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.headers = {"User-Agent": "SupportLens-Layer0/2.0"}

    def fetch_metadata(self, package_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Fetch package metadata and release history from PyPI JSON API."""
        url = f"{PYPI_API_BASE}/{package_name}/json"
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 404:
                return None, f"EXCLUDED: Package '{package_name}' not found on PyPI."
            if resp.status_code != 200:
                return None, f"EXCLUDED: PyPI API returned HTTP {resp.status_code}."
            return resp.json(), None
        except Exception as e:
            return None, f"EXCLUDED: Connection error to PyPI API ({str(e)})."

    def extract_releases(self, pypi_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract sorted list of releases with versions and release dates.
        Strict requirement: missing release dates are kept as None ('Unknown'),
        NEVER replaced with datetime.now() or current timestamp.
        """
        releases_raw = pypi_data.get("releases", {})
        parsed_releases = []

        for version_str, files in releases_raw.items():
            if not files:
                parsed_releases.append({
                    "tag": version_str,
                    "version": version_str,
                    "published_at": None,
                    "source": "PyPI JSON API (unknown date)"
                })
                continue

            upload_times = [f.get("upload_time_iso_8601") for f in files if f.get("upload_time_iso_8601")]
            if upload_times:
                earliest_time = min(upload_times)
            else:
                raw_times = [f.get("upload_time") for f in files if f.get("upload_time")]
                earliest_time = min(raw_times) if raw_times else None

            # Keep missing dates as None (auditable Unknown), never replace with current date
            parsed_releases.append({
                "tag": version_str,
                "version": version_str,
                "published_at": earliest_time,
                "source": "PyPI JSON API" if earliest_time else "PyPI JSON API (unknown date)"
            })

        # Sort safely: known timestamps sorted chronologically, missing timestamps sorted at the end
        parsed_releases.sort(
            key=lambda x: (
                x["published_at"] is None or str(x["published_at"]).lower() in ["unknown", "none", ""],
                x["published_at"] or ""
            )
        )
        return parsed_releases
