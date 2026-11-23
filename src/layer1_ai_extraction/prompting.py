"""Prompt and target format shared by training, the zero-shot baseline and inference.

The zero-shot baseline and the fine-tuned model receive exactly the same system prompt and
schema instructions (thesis 3.3.2, item 2). The training target is the canonical JSON below,
which is also the only text the constrained decoder accepts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .schema import COMMITMENT_KEYS, CommitmentExtractionOutput

# Policies longer than this are cut at a line boundary, in training and at inference alike.
MAX_POLICY_CHARS = 6000

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

SYSTEM_PROMPT = (
    "You are the extraction component of SupportLens. Read the SECURITY.md policy supplied by the user "
    "and extract its version-support commitment as one JSON object c = <t, tau, v, T> with exactly these "
    "keys, in this order:\n"
    '"commitment_text": the verbatim sentence(s) copied from the policy that state which versions receive '
    "security support (t). Vulnerability reporting instructions (such as confirming reproduction against latest) "
    "are not support commitments. Use an empty string if the policy contains no such statement.\n"
    '"version_expression": a list of the raw version tokens named in that commitment, copied as written, '
    'for example ["5.2.x", ">=3.0"] or ["latest"] (v). Use an empty list if none is named.\n'
    '"support_status": the commitment type (tau), one of\n'
    '  "latest_only": only the latest release is supported;\n'
    '  "explicit_range": specific versions or version ranges are named as supported;\n'
    '  "all_versions": every version is supported;\n'
    '  "ambiguous": the commitment is vague, conditional, or cannot be mapped to concrete versions.\n'
    '"temporal_reference": the point in time at which the commitment is evaluated (T), one of\n'
    '  "release_time": at the time a version is released;\n'
    '  "vulnerability_publication_time": at the time a vulnerability is published.\n'
    "Answer with the JSON object only."
)


# A Markdown heading that announces the version-support section ("## Supported Versions", "## Versions").
SUPPORT_HEADING_RE = re.compile(
    r"^#{1,6}[ \t][^\n]*\b(?:supported|support|versions?|maintenance|lifecycle|end[- ]of[- ]life|eol)\b[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_ANY_HEADING_RE = re.compile(r"^#{1,6}[ \t]", re.MULTILINE)
# An ASCII-armoured PGP block (for example a reporting key): many tokens and never a commitment.
PGP_BLOCK_RE = re.compile(r"-----BEGIN PGP [A-Z ]+-----.*?-----END PGP [A-Z ]+-----", re.DOTALL)
PGP_PLACEHOLDER = "[PGP block removed]"


def _cut_at_line(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    return text[: cut if cut > 0 else max_chars]


def prepare_policy_text(text: str, max_chars: int = MAX_POLICY_CHARS) -> str:
    """Make the policy safe and short enough for the prompt.

    A policy comes from an arbitrary repository, so the control-token opener `<|` is rewritten to
    `< |`; otherwise the tokenizer would read a literal `<|im_end|>` in the file as a real control
    token. ASCII-armoured PGP blocks are replaced by PGP_PLACEHOLDER. Text longer than `max_chars` is cut at the last line break before the limit. When the
    first version-support heading (SUPPORT_HEADING_RE) and its section do not fit in that leading
    window, the window starts at that heading instead, so a support section placed late in a long
    policy is not cut away. The same function is used in training and at inference.
    """
    text = PGP_BLOCK_RE.sub(PGP_PLACEHOLDER, text.replace("<|", "< |"))
    if len(text) <= max_chars:
        return text
    head = _cut_at_line(text, max_chars)
    match = SUPPORT_HEADING_RE.search(text)
    if match:
        following = _ANY_HEADING_RE.search(text, match.end())
        section_end = following.start() if following else len(text)
        if section_end > len(head):
            return _cut_at_line(text[match.start():], max_chars)
    return head


def build_chatml_prompt(policy_text: str) -> str:
    """ChatML prompt that ends where the assistant answer begins."""
    return (
        f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n"
        f"{IM_START}user\n{policy_text}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def serialize_commitment(commitment: Mapping[str, Any]) -> str:
    """Canonical JSON of the four thesis keys (validated, fixed key order, default separators)."""
    checked = CommitmentExtractionOutput(**{key: commitment[key] for key in COMMITMENT_KEYS}).commitment()
    return json.dumps({key: checked[key] for key in COMMITMENT_KEYS}, ensure_ascii=False)
