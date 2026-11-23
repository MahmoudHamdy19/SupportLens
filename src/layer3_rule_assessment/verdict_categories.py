"""
Layer 3: Rule-Based Assessment
Component: Assessment Outcome Categories
========================================
Component Specification:
- Consistent: The declared commitment is fully supported by the evidence.
- Partially Consistent: The commitment is supported for some vulnerabilities but not others.
- Potential Inconsistency: The commitment is not supported by the evidence, and sufficient evidence exists.
- Insufficient Evidence: The available evidence is incomplete or ambiguous.
- Unmeasurable Commitment: The commitment text cannot be mapped to concrete version identifiers.
"""

from enum import Enum
from typing import List

class VerdictCategory(str, Enum):
    CONSISTENT = "Consistent"
    PARTIALLY_CONSISTENT = "Partially Consistent"
    POTENTIAL_INCONSISTENCY = "Potential Inconsistency"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence"
    UNMEASURABLE_COMMITMENT = "Unmeasurable Commitment"

VERDICT_CATEGORIES: List[str] = [
    VerdictCategory.CONSISTENT.value,
    VerdictCategory.PARTIALLY_CONSISTENT.value,
    VerdictCategory.POTENTIAL_INCONSISTENCY.value,
    VerdictCategory.INSUFFICIENT_EVIDENCE.value,
    VerdictCategory.UNMEASURABLE_COMMITMENT.value,
]
