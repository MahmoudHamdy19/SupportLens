"""Baseline 1: deterministic regular-expression and keyword extractor (thesis 3.3.2, item 1).

The extractor splits the policy into candidate units (sentences, table blocks, list blocks),
picks the unit that best matches the support vocabulary and derives the tuple from it:

    t    the selected unit, copied verbatim from the input
    v    version tokens (`\\d+\\.\\d+\\.\\d+`, `2.x`, `>= 3.9`) and the token "latest"
    tau  from `commitment_rules.derive_support_status`
    T    from `commitment_rules.temporal_from_sentence`

There is no learning and no randomness. It outputs only the thesis tuple plus provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .commitment_rules import (
    ALL_VERSIONS_RE,
    HEADING_CUE_RE,
    LATEST_RE,
    LATEST_TOKEN,
    SUPPORT_CUE_RE,
    VERSION_WORD_RE,
    derive_support_status,
    extract_version_tokens,
    is_negative,
    temporal_from_sentence,
)
from .schema import CommitmentExtractionOutput

ENGINE_NAME = "RuleBasedRegexBaseline"

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[(*_`])")
_CLAUSE_BOUNDARY_RE = re.compile(r";\s+")


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    kind: str  # "prose" | "table" | "list"
    heading: str


def _trim(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_sentences(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    """Sentence offsets inside text[start:end]."""
    segment = text[start:end]
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for boundary in _SENTENCE_BOUNDARY_RE.finditer(segment):
        spans.append((cursor, boundary.start()))
        cursor = boundary.end()
    spans.append((cursor, len(segment)))
    result = []
    for a, b in spans:
        a2, b2 = _trim(text, start + a, start + b)
        if b2 > a2:
            result.append((a2, b2))
    return result


def _iter_units(text: str) -> List[_Unit]:
    """Candidate units with the heading in force. Offsets index into `text`."""
    units: List[_Unit] = []
    heading = ""
    block_kind: Optional[str] = None
    block_start = 0
    block_end = 0

    def flush() -> None:
        nonlocal block_kind
        if block_kind is None:
            return
        start, end = _trim(text, block_start, block_end)
        if end > start:
            if block_kind == "prose":
                for s, e in _split_sentences(text, start, end):
                    units.append(_Unit(s, e, "prose", heading))
            else:
                units.append(_Unit(start, end, block_kind, heading))
        block_kind = None

    position = 0
    for raw_line in text.splitlines(keepends=True):
        line_start = position
        position += len(raw_line)
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            flush()
            continue
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush()
            heading = heading_match.group(1).strip()
            continue
        if _TABLE_ROW_RE.match(line):
            kind = "table"
        elif _LIST_ITEM_RE.match(line):
            kind = "list"
        else:
            kind = "prose"
        if kind != block_kind:
            flush()
            block_kind = kind
            block_start = line_start
        block_end = line_start + len(line)
    flush()
    return units


@dataclass(frozen=True)
class _Candidate:
    score: int
    tokens: Tuple[str, ...]
    has_latest: bool


def _pieces(unit_text: str, tabular: bool) -> List[str]:
    """Table and list units are read line by line, prose units clause by clause."""
    return unit_text.splitlines() if tabular else _CLAUSE_BOUNDARY_RE.split(unit_text)


def _score(unit: _Unit, unit_text: str) -> Optional[_Candidate]:
    """Score a unit, or None when it cannot state a commitment.

    Lines or clauses that declare versions unsupported or end-of-life are ignored when reading
    version tokens; a unit made only of such pieces is not a commitment.
    """
    tabular = unit.kind != "prose"
    positive = [piece for piece in _pieces(unit_text, tabular) if not is_negative(piece, tabular=tabular)]
    if not positive:
        return None
    positive_text = "\n".join(positive)
    tokens: List[str] = []
    for piece in positive:
        for token in extract_version_tokens(piece):
            if token not in tokens:
                tokens.append(token)
    has_latest = bool(LATEST_RE.search(positive_text))
    has_all = bool(ALL_VERSIONS_RE.search(positive_text))
    anchor = 2 if (tokens or has_latest or has_all) else 0
    own_cue = bool(SUPPORT_CUE_RE.search(unit_text))
    heading_cue = bool(HEADING_CUE_RE.search(unit.heading))
    mentions_version = bool(VERSION_WORD_RE.search(unit_text))

    if anchor and (own_cue or heading_cue):
        score = anchor + (2 if own_cue else 0) + (1 if heading_cue else 0) + (1 if mentions_version else 0)
    elif own_cue and mentions_version:
        score = 2  # a support statement without any concrete version: kept as a vague commitment
    else:
        return None
    return _Candidate(score, tuple(tokens), has_latest)


class RuleBasedRegexBaseline:
    """Deterministic regex and keyword baseline."""

    def extract(self, text: str) -> CommitmentExtractionOutput:
        provenance = dict(
            extractor_engine=ENGINE_NAME,
            actual_model_used="none",
            lora_applied=False,
            execution_device="cpu",
            degraded=False,
        )
        if not text or not text.strip():
            return CommitmentExtractionOutput.empty(**provenance)

        best: Optional[Tuple[Tuple[int, int], _Unit, _Candidate]] = None
        for unit in _iter_units(text):
            candidate = _score(unit, text[unit.start : unit.end])
            if candidate is None:
                continue
            rank = (candidate.score, -unit.start)  # highest score, earliest position on ties
            if best is None or rank > best[0]:
                best = (rank, unit, candidate)
        if best is None:
            return CommitmentExtractionOutput.empty(**provenance)

        _, unit, candidate = best
        commitment_text = text[unit.start : unit.end]
        tokens = list(candidate.tokens)
        if candidate.has_latest:
            tokens.append(LATEST_TOKEN)
        return CommitmentExtractionOutput(
            commitment_text=commitment_text,
            version_expression=tokens,
            support_status=derive_support_status(tokens, commitment_text),
            temporal_reference=temporal_from_sentence(commitment_text),
            **provenance,
        )
