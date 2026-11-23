"""
Layer 2: Evidence Layer
Component: Version Normalization
=================================
Component Specification:
- Parses version expressions using packaging (PEP 440).
- Converts wildcards (e.g. 5.2.x, 3.x, ~=2.1) into standard PEP 440 specifiers.
- A bare release-line token denotes the whole line: "1.2" -> >=1.2.0,<1.3.0 and "3" -> >=3,<4.
  Only a full N.M.P token (e.g. "1.2.3") is an exact pin (==1.2.3).
- Resolves relative temporal terms (RELATIVE_TERMS, e.g. 'latest', 'newest') strictly against the
  release timeline at Time T. The timeline is a pandas DataFrame (version, published_at) filtered at
  T (thesis 3.2.3, tools: pandas, packaging). Pre-releases and dev releases never resolve a relative term.
- Strict temporal integrity: missing dates remain None; never substitute with current or artificial dates.
  When T is given, releases without a publication date are excluded from the timeline.
"""

import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import pandas as pd
from packaging.version import Version, InvalidVersion
from packaging.specifiers import SpecifierSet, InvalidSpecifier

def parse_iso_datetime(date_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime safely into a timezone-aware UTC datetime. Returns None on missing data."""
    if not date_str or str(date_str).lower() in ["none", "unknown", ""]:
        return None
    try:
        clean = str(date_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        try:
            dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

def clean_version_string(v_str: str) -> str:
    """Normalize version string by stripping 'v', quotes, and spaces."""
    if not v_str:
        return ""
    return str(v_str).strip().lstrip("vV").strip("`'\" ")

def convert_wildcard_to_specifier(expr: str) -> Optional[str]:
    """
    Converts expressions like '5.2.x', '5.2.*', '3.x', '~=2.1' into standard PEP 440 specifiers.
    """
    clean = clean_version_string(expr)
    if not clean:
        return None

    # 5.2.x or 5.2.*
    match_minor = re.match(r"^(\d+)\.(\d+)[\.\*xX]+$", clean)
    if match_minor:
        major, minor = match_minor.group(1), match_minor.group(2)
        next_minor = int(minor) + 1
        return f">={major}.{minor}.0, <{major}.{next_minor}.0"

    # 3.x or 3.*
    match_major = re.match(r"^(\d+)[\.\*xX]+$", clean)
    if match_major:
        major = match_major.group(1)
        next_major = int(major) + 1
        return f">={major}.0.0, <{next_major}.0.0"

    if any(clean.startswith(op) for op in [">=", "<=", ">", "<", "==", "~=", "!="]):
        return clean

    # Bare N.M (e.g. "1.2" in a support table) means the whole 1.2 release line.
    match_line = re.match(r"^(\d+)\.(\d+)$", clean)
    if match_line:
        major, minor = int(match_line.group(1)), int(match_line.group(2))
        return f">={major}.{minor}.0, <{major}.{minor + 1}.0"

    # Bare N means the whole major release line.
    if re.match(r"^\d+$", clean):
        return f">={int(clean)}, <{int(clean) + 1}"

    # Full N.M.P (or any other concrete version) is an exact pin.
    return f"=={clean}"


# Relative temporal terms resolved against the release timeline at time T.
RELATIVE_TERMS = frozenset({
    "latest", "current", "most recent", "newest", "latest stable", "current stable",
})


def _release_version(release: Dict[str, Any]) -> Optional[Version]:
    for key in ("tag", "version"):
        raw = release.get(key)
        if raw:
            try:
                return Version(clean_version_string(str(raw)))
            except InvalidVersion:
                continue
    return None


def release_timeline(releases: List[Dict[str, Any]], reference_time: Optional[datetime] = None) -> pd.DataFrame:
    """Release timeline as a DataFrame with columns (version, published_at), filtered at time T.

    `version` holds packaging Version objects; `published_at` is a UTC timestamp (NaT when unknown).
    With a reference time T only releases published at or before T are kept, so releases with an
    unknown date are dropped rather than assumed. Unparseable versions are dropped.
    """
    rows = []
    for release in releases or []:
        version = _release_version(release)
        if version is None:
            continue
        published = parse_iso_datetime(release.get("published_at"))
        rows.append({"version": version, "published_at": pd.Timestamp(published) if published else pd.NaT})
    frame = pd.DataFrame(rows, columns=["version", "published_at"])
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    if reference_time is not None:
        moment = pd.Timestamp(reference_time)
        moment = moment.tz_localize("UTC") if moment.tzinfo is None else moment.tz_convert("UTC")
        frame = frame[frame["published_at"].notna() & (frame["published_at"] <= moment)]
    return frame.reset_index(drop=True)


def latest_stable_version(releases: List[Dict[str, Any]], reference_time: Optional[datetime] = None) -> Optional[Version]:
    """Highest final release (no pre-release, no dev release) in the timeline at time T."""
    frame = release_timeline(releases, reference_time)
    stable = [v for v in frame["version"] if not (v.is_prerelease or v.is_devrelease)]
    return max(stable) if stable else None

class VersionNormalizer:
    """Normalizes version expressions and resolves them using temporal information."""

    @classmethod
    def resolve_specifiers(
        cls,
        raw_expressions: List[str],
        releases: List[Dict[str, Any]],
        reference_time: Optional[datetime] = None
    ) -> List[SpecifierSet]:
        """
        Convert list of raw expressions into SpecifierSet objects.
        Resolves 'latest' strictly against releases that existed at or before reference_time T.
        """
        specifiers = []

        for expr in raw_expressions:
            if not expr:
                continue
            expr_lower = str(expr).strip().lower()

            if expr_lower in RELATIVE_TERMS:
                # Temporal resolution against the release timeline at T; pre/dev releases excluded.
                latest_ver = latest_stable_version(releases, reference_time)
                if latest_ver is not None:
                    specifiers.append(SpecifierSet(f"=={latest_ver}"))
                continue

            if expr_lower in ["all", "all versions", "every release", "all releases"]:
                specifiers.append(SpecifierSet(">=0.0.0"))
                continue

            spec_str = convert_wildcard_to_specifier(expr)
            if spec_str:
                try:
                    specifiers.append(SpecifierSet(spec_str))
                except InvalidSpecifier:
                    pass

        return specifiers
