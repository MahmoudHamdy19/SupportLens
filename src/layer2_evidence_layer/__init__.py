"""
SupportLens — Layer 2: Evidence Layer
======================================
"""

from .evidence_sources import EvidenceCollector
from .version_normalization import VersionNormalizer, convert_wildcard_to_specifier, clean_version_string, parse_iso_datetime
from .normalized_scope import NormalizedVersionScope

__all__ = [
    "EvidenceCollector",
    "VersionNormalizer",
    "NormalizedVersionScope",
    "convert_wildcard_to_specifier",
    "clean_version_string",
    "parse_iso_datetime",
]

