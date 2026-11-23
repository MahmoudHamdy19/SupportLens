"""Rule-based baseline on small SYNTHETIC policy snippets.

The snippets below are written for these unit tests only. They are fixtures, not data, and are not
part of any dataset or evaluation.
"""

import pytest

from src.layer1_ai_extraction.baselines import RuleBasedRegexBaseline
from src.layer1_ai_extraction.schema import OUTPUT_KEYS

SYNTHETIC_TABLE = """# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 5.1.x   | :white_check_mark: |
| 5.0.x   | :x:                |
| 4.0.x   | :white_check_mark: |
| < 4.0   | :x:                |

## Reporting a Vulnerability

Please email security@example.invalid. We reply within 48 hours.
"""

SYNTHETIC_LATEST = (
    "# Security\n\nOnly the latest release is supported with security updates. "
    "Please report issues by email.\n"
)

SYNTHETIC_LIST = "## Supported versions\n\n- 2.x: security fixes\n- 1.x: security fixes\n\nReport by email.\n"

SYNTHETIC_ALL = "We support all versions of this library with security patches.\n"

SYNTHETIC_SPECIFIER = "We provide security fixes for Python >= 3.9 and for version 4.2.1 and 4.1.x.\n"

SYNTHETIC_VAGUE = "We will try to support recent releases when possible.\n"

SYNTHETIC_NO_COMMITMENT = "# Security\nPlease email security@example.invalid to report vulnerabilities.\n"

SYNTHETIC_RELEASE_TIME = "Each version is supported for one year from the time of release.\n"


@pytest.fixture
def extractor():
    return RuleBasedRegexBaseline()


def test_output_has_only_contract_keys(extractor):
    assert tuple(extractor.extract(SYNTHETIC_TABLE).to_dict()) == OUTPUT_KEYS


def test_table_keeps_supported_rows_only(extractor):
    out = extractor.extract(SYNTHETIC_TABLE)
    assert out.version_expression == ["5.1.x", "4.0.x"]
    assert out.support_status == "explicit_range"
    assert out.is_measurable
    assert "5.1.x" in out.commitment_text and "Reporting" not in out.commitment_text


def test_only_latest_release(extractor):
    out = extractor.extract(SYNTHETIC_LATEST)
    assert out.support_status == "latest_only"
    assert out.version_expression == ["latest"]
    assert out.commitment_text == "Only the latest release is supported with security updates."


def test_bullet_list(extractor):
    out = extractor.extract(SYNTHETIC_LIST)
    assert out.version_expression == ["2.x", "1.x"]
    assert out.support_status == "explicit_range"


def test_all_versions_has_no_version_token(extractor):
    out = extractor.extract(SYNTHETIC_ALL)
    assert out.support_status == "all_versions"
    assert out.version_expression == []
    assert out.is_measurable


def test_specifier_and_dotted_versions(extractor):
    out = extractor.extract(SYNTHETIC_SPECIFIER)
    assert out.version_expression == [">=3.9", "4.2.1", "4.1.x"]
    assert out.support_status == "explicit_range"


def test_vague_statement_is_ambiguous_but_quoted(extractor):
    out = extractor.extract(SYNTHETIC_VAGUE)
    assert out.support_status == "ambiguous"
    assert out.version_expression == []
    assert out.commitment_text == SYNTHETIC_VAGUE.strip()
    assert not out.is_measurable


def test_no_commitment_gives_the_empty_ambiguous_tuple(extractor):
    out = extractor.extract(SYNTHETIC_NO_COMMITMENT)
    assert (out.commitment_text, out.version_expression, out.support_status) == ("", [], "ambiguous")


@pytest.mark.parametrize("text", ["", "   \n  "])
def test_empty_input(extractor, text):
    out = extractor.extract(text)
    assert out.support_status == "ambiguous" and out.commitment_text == "" and not out.is_measurable


def test_temporal_reference_rule(extractor):
    assert extractor.extract(SYNTHETIC_RELEASE_TIME).temporal_reference == "release_time"
    assert extractor.extract(SYNTHETIC_LATEST).temporal_reference == "vulnerability_publication_time"


@pytest.mark.parametrize(
    "text",
    [SYNTHETIC_TABLE, SYNTHETIC_LATEST, SYNTHETIC_LIST, SYNTHETIC_ALL, SYNTHETIC_SPECIFIER, SYNTHETIC_VAGUE],
)
def test_commitment_text_is_verbatim(extractor, text):
    assert extractor.extract(text).commitment_text in text


def test_deterministic(extractor):
    assert extractor.extract(SYNTHETIC_TABLE) == extractor.extract(SYNTHETIC_TABLE)


def test_provenance_is_honest(extractor):
    data = extractor.extract(SYNTHETIC_LATEST).to_dict()
    assert data["extractor_engine"] == "RuleBasedRegexBaseline"
    assert data["actual_model_used"] == "none"
    assert data["lora_applied"] is False
    assert data["degraded"] is False
