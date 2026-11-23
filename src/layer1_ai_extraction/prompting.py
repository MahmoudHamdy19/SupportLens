"""Prompt and target format shared by training, the zero-shot baseline and inference.

The zero-shot baseline and the fine-tuned model receive exactly the same system prompt and
schema instructions (thesis 3.3.2, item 2). The training target is the canonical JSON below,
which is also the only text the constrained decoder accepts.
"""

from __future__ import annotations

import json
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


def prepare_policy_text(text: str, max_chars: int = MAX_POLICY_CHARS) -> str:
    """Make the policy safe and short enough for the prompt.

    A policy comes from an arbitrary repository, so the control-token opener `<|` is rewritten to
    `< |`; otherwise the tokenizer would read a literal `<|im_end|>` in the file as a real control
    token. Text longer than `max_chars` is cut at the last line break before the limit.
    """
    text = text.replace("<|", "< |")
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    return text[: cut if cut > 0 else max_chars]


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
