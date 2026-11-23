"""Deterministic lexical rules shared by the rule-based and RoBERTa baselines (thesis 3.3.2).

Everything here is pure string processing. The rules are interpretations of the thesis, which
names the pattern families (`\\d+\\.\\d+\\.\\d+`, `>= 3.9`, "latest", "supported", "end-of-life")
but not their exact form.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from .schema import DEFAULT_TEMPORAL_REFERENCE

_VERSION_CORE = r"v?(?:\d+(?:\.\d+)+(?:\.(?:[xX]|\*))?|\d+\.(?:[xX]|\*))\+?"
_SPECIFIER = r"(?:>=|<=|==|~=|!=|≥|≤|>|<)\s*" + _VERSION_CORE

# A specifier is tried first, so ">= 3.9" is one token and not ">=" plus "3.9".
VERSION_TOKEN_RE = re.compile(r"(?<![\w.])(?:" + _SPECIFIER + "|" + _VERSION_CORE + r")(?!\w)")

LATEST_RE = re.compile(
    r"\b(?:latest|newest|most\s+recent|current(?:ly)?\s+(?:stable\s+)?(?:release|version))\b", re.IGNORECASE
)
ALL_VERSIONS_RE = re.compile(r"\b(?:all|every|any)\s+(?:previous\s+|released\s+)?(?:versions?|releases?)\b", re.IGNORECASE)

SUPPORT_CUE_RE = re.compile(
    r"\b(?:support(?:s|ed|ing)?|maintain(?:s|ed|ing)?|maintenance|backport(?:s|ed)?|patch(?:es|ed)?"
    r"|security\s+(?:updates?|fix(?:es)?|patch(?:es)?|releases?)|end[-\s]of[-\s]life|eol)\b",
    re.IGNORECASE,
)
VERSION_WORD_RE = re.compile(r"\b(?:versions?|releases?|branches?)\b", re.IGNORECASE)
HEADING_CUE_RE = re.compile(r"support|version|release|maintenance|lifecycle|end[-\s]of[-\s]life|eol", re.IGNORECASE)

_NEGATIVE_PROSE = (
    r"\bunsupported\b|\bnot\s+(?:currently\s+)?(?:supported|maintained)\b|\bno\s+longer\s+(?:supported|maintained)\b"
    r"|\bend[-\s]of[-\s]life\b|\beol\b|\bdeprecated\b|:x:|❌|✗|✘"
)
NEGATIVE_PROSE_RE = re.compile(_NEGATIVE_PROSE, re.IGNORECASE)
NEGATIVE_TABULAR_RE = re.compile(_NEGATIVE_PROSE + r"|\bno\b", re.IGNORECASE)
POSITIVE_RE = re.compile(
    r":white_check_mark:|✅|✔|✓|\byes\b|\bsupported\b|\bactive\b|\bmaintained\b"
    r"|security\s+(?:fixes|updates|patches)",
    re.IGNORECASE,
)

# T anchored at the release of a version, versus at the publication of a vulnerability.
RELEASE_TIME_PHRASE_RE = re.compile(
    r"\b(?:at|since|upon|from)\s+(?:the\s+)?(?:time\s+of\s+)?(?:its\s+|their\s+|each\s+)?release\b"
    r"|\bwhen\s+(?:it\s+(?:is|was)|they\s+(?:are|were))\s+released\b|\brelease\s+date\b",
    re.IGNORECASE,
)
VULNERABILITY_TIME_PHRASE_RE = re.compile(
    r"\b(?:at|upon|since|from)\s+(?:the\s+)?(?:time\s+of\s+)?(?:the\s+)?(?:vulnerability\s+|cve\s+|advisory\s+)?"
    r"(?:publication|disclosure|report)\b"
    r"|\bwhen\s+(?:a\s+|the\s+)?(?:vulnerabilit(?:y|ies)|issue|cve)\s+(?:is|are|was|were)\s+(?:published|disclosed|reported)\b",
    re.IGNORECASE,
)
_RELEASE_WORD_RE = re.compile(r"\breleas\w*\b", re.IGNORECASE)
_VULNERABILITY_WORD_RE = re.compile(
    r"\b(?:publication|published|disclos\w*|reported|advisor\w*|vulnerab\w*|cve)\b", re.IGNORECASE
)

LATEST_TOKEN = "latest"


def normalise_version_token(token: str) -> str:
    """Remove whitespace inside a token (">= 3.9" -> ">=3.9") and trim surrounding punctuation."""
    cleaned = re.sub(r"\s+", "", token)
    return cleaned.strip(".,;:()[]{}`'\"*_|")


def extract_version_tokens(text: str) -> List[str]:
    """Raw version tokens in order of appearance, without duplicates."""
    tokens: List[str] = []
    for match in VERSION_TOKEN_RE.finditer(text):
        token = normalise_version_token(match.group(0))
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def is_negative(text: str, tabular: bool = False) -> bool:
    """True when the text declares versions unsupported or end-of-life."""
    pattern = NEGATIVE_TABULAR_RE if tabular else NEGATIVE_PROSE_RE
    return bool(pattern.search(text))


def derive_support_status(version_tokens: Sequence[str], commitment_text: str) -> str:
    """tau from the version tokens and the commitment text.

    Numeric token -> explicit_range; "latest"-like token -> latest_only; "all versions" phrase
    -> all_versions; otherwise ambiguous (including an empty commitment).
    """
    if not commitment_text.strip():
        return "ambiguous"
    if any(re.search(r"\d", token) for token in version_tokens):
        return "explicit_range"
    if any(LATEST_RE.fullmatch(token.strip()) for token in version_tokens):
        return "latest_only"
    if ALL_VERSIONS_RE.search(commitment_text):
        return "all_versions"
    return "ambiguous"


def temporal_from_sentence(text: str) -> str:
    """T from a full commitment sentence: release_time only on an explicit release-time phrase.

    The default is vulnerability_publication_time, because the assessment compares support
    against vulnerabilities and the word "release" occurs in most commitments ("latest release").
    """
    if RELEASE_TIME_PHRASE_RE.search(text) and not VULNERABILITY_TIME_PHRASE_RE.search(text):
        return "release_time"
    return DEFAULT_TEMPORAL_REFERENCE


def temporal_from_span_texts(span_texts: Iterable[str]) -> str:
    """T from short predicted temporal spans (RoBERTa baseline).

    A span that mentions a release and no vulnerability term gives release_time; no span, or any
    other span, gives the default.
    """
    joined = " ".join(span_texts)
    if _RELEASE_WORD_RE.search(joined) and not _VULNERABILITY_WORD_RE.search(joined):
        return "release_time"
    return DEFAULT_TEMPORAL_REFERENCE
