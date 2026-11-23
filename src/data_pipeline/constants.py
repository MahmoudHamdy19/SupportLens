"""Shared vocabulary of the thesis (Chapter 3) used by the data pipeline and the evaluation."""

SEED = 42

# Table 3.1, in the order used for every confusion matrix.
VERDICTS = (
    "Consistent",
    "Partially Consistent",
    "Potential Inconsistency",
    "Insufficient Evidence",
    "Unmeasurable Commitment",
)

# Commitment type (tau) and temporal reference (T), Section 3.1.3.2 and docs/INTERFACES.md.
TAU_VALUES = ("latest_only", "explicit_range", "all_versions", "ambiguous")
TEMPORAL_VALUES = ("release_time", "vulnerability_publication_time")

# Keys of the commitment tuple c = <t, tau, v, T> (docs/INTERFACES.md, section 1).
COMMITMENT_KEYS = ("commitment_text", "version_expression", "support_status", "temporal_reference")

# Target of the inter-annotator agreement (Section 3.3.1).
KAPPA_TARGET = 0.80

# Stratified split proportions (Section 3.3.1).
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

ANNOTATORS = ("annotator1", "annotator2")
