"""SupportLens Layer 1: AI extraction of the version-support commitment c = <t, tau, v, T>.

Heavy libraries (torch, transformers, peft) are imported lazily, so importing this package
works without them.
"""

from .baselines import RuleBasedRegexBaseline
from .errors import ExtractionFormatError, ModelUnavailableError
from .qwen_extractor import QwenCommitmentExtractor
from .roberta_baseline import RoBERTaCommitmentExtractor
from .schema import (
    SUPPORT_STATUSES,
    TEMPORAL_REFERENCES,
    CommitmentExtractionOutput,
    compute_is_measurable,
)
from .seeding import set_global_seed

__all__ = [
    "CommitmentExtractionOutput",
    "ExtractionFormatError",
    "ModelUnavailableError",
    "QwenCommitmentExtractor",
    "RoBERTaCommitmentExtractor",
    "RuleBasedRegexBaseline",
    "SUPPORT_STATUSES",
    "TEMPORAL_REFERENCES",
    "compute_is_measurable",
    "set_global_seed",
]
