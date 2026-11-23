"""Consistency check of model-extracted commitments (synthetic fixtures, no model)."""

from src.layer1_ai_extraction import qwen_extractor
from src.layer1_ai_extraction.commitment_guard import check_commitment
from src.layer1_ai_extraction.schema import CommitmentExtractionOutput

POLICY = (
    "# Security Policy\n\n"
    "## Supported Versions\n\n"
    "Only the main branch (the 2.x release line) receives updates, including security fixes.\n\n"
    "## Reporting a Vulnerability\n\n"
    "Please confirm the issue on the latest version before you report it.\n"
)


def output(text, versions, tau):
    return CommitmentExtractionOutput(
        commitment_text=text,
        version_expression=versions,
        support_status=tau,
        temporal_reference="vulnerability_publication_time",
        extractor_engine="Qwen (synthetic)",
    )


def test_reporting_instruction_is_replaced_by_the_support_statement():
    wrong = output("Please confirm the issue on the latest version before you report it.", ["latest"], "latest_only")
    checked, note = check_commitment(POLICY, wrong)
    assert "no support statement" in note
    assert checked.support_status == "explicit_range"
    assert "2.x" in checked.version_expression
    assert "security fixes" in checked.commitment_text


def test_numeric_version_with_latest_only_becomes_explicit_range():
    text = "Only the main branch (the 2.x release line) receives updates, including security fixes."
    checked, note = check_commitment(POLICY, output(text, ["2.x"], "latest_only"))
    assert checked.support_status == "explicit_range" and "explicit_range" in note


def test_non_verbatim_text_falls_back_to_the_baseline():
    checked, note = check_commitment(POLICY, output("We support 9.x with security fixes.", ["9.x"], "explicit_range"))
    assert "not verbatim" in note and "9.x" not in checked.version_expression


def test_a_consistent_output_is_unchanged():
    text = "Only the main branch (the 2.x release line) receives updates, including security fixes."
    original = output(text, ["2.x"], "explicit_range")
    checked, note = check_commitment(POLICY, original)
    assert note is None and checked == original


def _extractor(strict, monkeypatch, generated):
    extractor = qwen_extractor.QwenCommitmentExtractor(strict=strict)
    monkeypatch.setattr(extractor, "availability_problems", lambda: [])
    monkeypatch.setattr(extractor, "generate_json", lambda text: generated)
    monkeypatch.setattr(extractor, "_provenance", lambda: {"extractor_engine": "Qwen (synthetic)"})
    return extractor


def test_check_runs_only_outside_strict_mode(monkeypatch):
    generated = (
        '{"commitment_text": "Please confirm the issue on the latest version before you report it.", '
        '"version_expression": ["latest"], "support_status": "latest_only", '
        '"temporal_reference": "vulnerability_publication_time"}'
    )
    strict = _extractor(True, monkeypatch, generated).extract(POLICY)
    assert strict.support_status == "latest_only" and "consistency check" not in strict.extractor_engine
    live = _extractor(False, monkeypatch, generated).extract(POLICY)
    assert live.support_status == "explicit_range" and "consistency check" in live.extractor_engine


def test_a_supported_versions_table_row_counts_as_a_support_statement():
    policy = "# Security\n\n| Version | Status |\n| --- | --- |\n| 2.5.x | :white_check_mark: |\n"
    original = output("| 2.5.x | :white_check_mark: |", ["2.5.x"], "explicit_range")
    assert check_commitment(policy, original) == (original, None)
