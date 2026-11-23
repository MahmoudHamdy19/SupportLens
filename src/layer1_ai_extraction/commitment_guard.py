"""Deterministic consistency check of a model-extracted commitment (live audits only).

The check is applied by `QwenCommitmentExtractor` when `strict=False` (CLI and web interface). It is
never applied in an evaluation, which always runs with `strict=True`, so the reported Qwen metrics
measure the model output itself. Every correction is named in `extractor_engine`.

Rules, in order:
  1. A non-empty commitment_text that is not a verbatim substring of the policy (whitespace
     normalised) is replaced by the regex baseline result.
  2. A commitment_text without any support cue (`SUPPORT_CUE_RE`, or a table marker of
     `POSITIVE_RE`), for example a vulnerability reporting instruction such as "confirm the issue
     on the latest version", is replaced by the regex baseline result when the baseline finds a
     commitment that has one.
  3. tau `latest_only` or `all_versions` together with a numeric version token (for example "2.x")
     is corrected to `explicit_range`, as `commitment_rules.derive_support_status` would derive.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from .baselines import RuleBasedRegexBaseline
from .commitment_rules import POSITIVE_RE, SUPPORT_CUE_RE
from .schema import CommitmentExtractionOutput


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _has_support_cue(text: str) -> bool:
    return bool(SUPPORT_CUE_RE.search(text) or POSITIVE_RE.search(text))


def check_commitment(policy_text: str, output: CommitmentExtractionOutput) -> Tuple[CommitmentExtractionOutput, Optional[str]]:
    """Return the (possibly corrected) output and a short note naming the correction, or None."""
    text = output.commitment_text
    if not text:
        return output, None

    replacement_reason: Optional[str] = None
    if _squash(text) not in _squash(policy_text):
        replacement_reason = "commitment_text not verbatim in the policy"
    elif not _has_support_cue(text):
        replacement_reason = "commitment_text has no support statement"

    if replacement_reason:
        baseline = RuleBasedRegexBaseline().extract(policy_text)
        usable = bool(baseline.commitment_text) and (
            replacement_reason.startswith("commitment_text not verbatim")
            or _has_support_cue(baseline.commitment_text)
        )
        if usable:
            corrected = output.model_copy(update=baseline.commitment())
            return corrected, f"regex baseline used: {replacement_reason}"

    numeric = any(re.search(r"\d", token) for token in output.version_expression)
    if numeric and output.support_status in ("latest_only", "all_versions"):
        corrected = output.model_copy(update={"support_status": "explicit_range"})
        return corrected, f"support_status {output.support_status} -> explicit_range (numeric version token)"
    return output, None
