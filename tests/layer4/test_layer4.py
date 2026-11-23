"""
Layer 4 unit tests: template explanations, JSON record and report storage.

All fixtures below are SYNTHETIC (invented packages, identifiers and dates) and exist only to
exercise the code paths. Reports are written under pytest's tmp_path, never under data/reports.
No network access is used.
"""

import json
from datetime import datetime, timedelta, timezone

import jinja2
import pytest

from src.layer3_rule_assessment import VERDICT_CATEGORIES, AssessmentRuleEngine
from src.layer4_explanation_output import ExplanationGenerator, FinalOutputBuilder, ResultStorage

NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)
POLICY_TEXT = "Versions 1.x and 2.x receive security fixes."

PACKAGE_INFO = {
    "github_repo": "example-org/synthetic-pkg",
    "security_policy_url": "https://github.com/example-org/synthetic-pkg/blob/main/SECURITY.md",
    "security_policy_path": "SECURITY.md",
    "line_references": {
        "Reporting": {"line_start": 11, "line_end": 20, "section_type": "vulnerability_reporting"},
        "Supported Versions": {"line_start": 3, "line_end": 9, "section_type": "supported_versions"},
    },
}


def release(version, day):
    return {"version": version, "tag": version, "published_at": f"{day}T00:00:00Z", "source": "pypi"}


RELEASES = [release("1.0.0", "2022-01-01"), release("2.0.0", "2023-01-01"), release("2.0.1", "2023-03-01")]


def commitment(tau="explicit_range", expression=("1.x", "2.x")):
    return {
        "commitment_text": POLICY_TEXT,
        "support_status": tau,
        "version_expression": list(expression),
        "temporal_reference": "vulnerability_publication_time",
        "is_measurable": tau != "ambiguous",
    }


def vulnerability(vid, published, affected, fixed=(), **extra):
    record = {
        "id": vid, "cve_ids": [vid], "ghsa_ids": [], "published_at": published,
        "affected_versions": list(affected), "affected_ranges": [], "fixed_versions": list(fixed),
        "advisory_url": None,
    }
    record.update(extra)
    return record


R1 = vulnerability("CVE-2024-0001", "2023-02-01T00:00:00Z", ["2.0.0"], ["2.0.1"],
                   advisory_url="https://example.org/advisories/synthetic-1")
R3 = vulnerability("CVE-2024-0003", "2023-02-01T00:00:00Z", ["2.0.0"])
R4 = vulnerability("CVE-2024-0004", (NOW - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ"), ["2.0.0"])


def assess(c, vulns, releases=RELEASES):
    return AssessmentRuleEngine().evaluate(c, vulns, releases, reference_time=NOW, package_name="synthetic-pkg")


CASES = {
    "Consistent": lambda: assess(commitment(), [R1]),
    "Partially Consistent": lambda: assess(commitment(), [R1, R3]),
    "Potential Inconsistency": lambda: assess(commitment(), [R3]),
    "Insufficient Evidence": lambda: assess(commitment(), [R4]),
    "Unmeasurable Commitment": lambda: assess(commitment(tau="ambiguous", expression=()), [R1]),
}


# ---------------------------------------------------------------------------
# Explanation templates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("verdict", VERDICT_CATEGORIES)
def test_every_verdict_renders_a_grounded_explanation(verdict):
    result = CASES[verdict]()
    assert result.verdict.value == verdict
    text = ExplanationGenerator().generate_explanation(
        result, commitment(), "SECURITY.md, supported-versions section, lines 3-9",
        ["https://pypi.org/project/synthetic-pkg/"],
    )
    assert text.startswith(f"**Verdict: {verdict}**")
    assert POLICY_TEXT in text
    assert "lines 3-9" in text
    assert "https://pypi.org/project/synthetic-pkg/" in text
    assert text.isascii()
    for banned in ("SLA", "Non-PyPI", "{{", "{%"):
        assert banned not in text


@pytest.mark.parametrize(
    "verdict, facts",
    [
        ("Consistent", ["CVE-2024-0001", "2023-02-01", "2.0.1 (released 2023-03-01)", "https://example.org/advisories/synthetic-1"]),
        ("Partially Consistent", ["CVE-2024-0001", "CVE-2024-0003", "2023-02-01", "2023-03-01", "days after publication"]),
        ("Potential Inconsistency", ["CVE-2024-0003", "2023-02-01", "2.0.0 (released 2023-01-01)", "90 days", "no fixed version"]),
        ("Insufficient Evidence", ["CVE-2024-0004", str(NOW.date() - timedelta(days=10)), "10 days"]),
    ],
)
def test_explanation_cites_ids_and_dates(verdict, facts):
    text = ExplanationGenerator().generate_explanation(CASES[verdict](), commitment(), None, [])
    for fact in facts:
        assert fact in text


def test_unmeasurable_explanation_names_rule_and_reason():
    text = ExplanationGenerator().generate_explanation(CASES["Unmeasurable Commitment"](), commitment("ambiguous", ()), None, [])
    assert "rule R5" in text and "ambiguous" in text


def test_insufficient_evidence_without_vulnerability_data_explains_the_gap():
    result = assess(commitment(), None)
    text = ExplanationGenerator().generate_explanation(result, commitment(), None, [])
    assert "Vulnerability evidence V is missing" in text


def test_partially_consistent_lists_unremediated_versions():
    vuln = vulnerability("CVE-2024-0002", "2023-02-01T00:00:00Z", ["1.0.0", "2.0.0"], ["2.0.1"])
    text = ExplanationGenerator().generate_explanation(assess(commitment(), [vuln]), commitment(), None, [])
    assert "CVE-2024-0002" in text and "1.0.0" in text and "major release line" in text


def test_long_record_lists_are_truncated_with_a_pointer():
    vulns = [vulnerability(f"CVE-2024-{n:04d}", "2023-02-01T00:00:00Z", ["2.0.0"], ["2.0.1"]) for n in range(15)]
    text = ExplanationGenerator().generate_explanation(assess(commitment(), vulns), commitment(), None, [])
    assert text.count("- CVE-2024-") == 10
    assert "5 further record(s) are listed in the stored report" in text


def test_explanation_is_deterministic():
    generator = ExplanationGenerator()
    result = CASES["Partially Consistent"]()
    first = generator.generate_explanation(result, commitment(), "SECURITY.md", ["a", "b"])
    second = generator.generate_explanation(result, commitment(), "SECURITY.md", ["a", "b"])
    assert first == second


def test_no_fallback_text_when_the_template_is_missing(tmp_path):
    generator = ExplanationGenerator(template_dir=tmp_path)
    with pytest.raises(jinja2.TemplateNotFound):
        generator.generate_explanation(CASES["Consistent"](), commitment(), None, [])


def test_unknown_verdict_is_rejected():
    with pytest.raises(ValueError):
        ExplanationGenerator().generate_explanation({"verdict": "Compliant"}, commitment(), None, [])


# ---------------------------------------------------------------------------
# Final record
# ---------------------------------------------------------------------------

def build(tmp_path, result=None, package_info=PACKAGE_INFO, **kwargs):
    builder = FinalOutputBuilder(output_dir=tmp_path)
    record = builder.build_and_save(
        result or CASES["Partially Consistent"](), package_info,
        kwargs.pop("source_references", ["PyPI: 3 releases"]), commitment(), kwargs.pop("layer2_evidence", None),
    )
    return builder, record


def test_record_contains_verdict_explanation_sources_and_trace(tmp_path):
    traces = {"status": "INCLUDED", "layer0": {"github_repo": "example-org/synthetic-pkg"},
              "layer2": {"total_vulnerabilities": 2, "fetched_at": NOW}, "layer3": {"duplicate": True}}
    _, record = build(tmp_path, layer2_evidence={"total_releases": 3, "layer_traces": traces})

    assert record["verdict"] == "Partially Consistent"
    assert POLICY_TEXT in record["explanation"]
    assert record["commitment"]["support_status"] == "explicit_range"
    assert record["assessment"]["rules_triggered"] == ["R1", "R3"]
    assert record["assessment"]["evidence_trail"] and record["assessment"]["evidence_log"]
    assert record["trace"]["layer0"] == traces["layer0"]
    assert record["trace"]["layer2"]["total_vulnerabilities"] == 2
    assert "layer3" not in record["trace"]

    assert record["policy"]["line_start"] == 3 and record["policy"]["line_end"] == 9
    assert record["source_references"][0] == PACKAGE_INFO["security_policy_url"] + "#L3-L9"
    for reference in (
        "https://github.com/example-org/synthetic-pkg",
        "https://pypi.org/project/synthetic-pkg/",
        "https://example.org/advisories/synthetic-1",
        "PyPI: 3 releases",
    ):
        assert reference in record["source_references"]
    assert len(record["source_references"]) == len(set(record["source_references"]))
    assert "lines 3-9" in record["explanation"]


def test_record_is_stored_as_reproducible_json(tmp_path):
    traces = {"layer2": {"fetched_at": NOW}}
    _, record = build(tmp_path, layer2_evidence={"layer_traces": traces})
    stored = json.loads((tmp_path / f"{record['report_id']}.json").read_text(encoding="utf-8"))
    expected = {k: v for k, v in record.items() if k != "saved_report_path"}
    expected["trace"] = {"layer2": {"fetched_at": NOW.isoformat()}}
    assert stored == expected
    assert record["saved_report_path"] == str(tmp_path / f"{record['report_id']}.json")


def test_same_inputs_give_the_same_record_apart_from_id_and_time(tmp_path):
    _, first = build(tmp_path)
    _, second = build(tmp_path)
    volatile = {"report_id", "created_at", "saved_report_path"}
    assert {k: v for k, v in first.items() if k not in volatile} == {k: v for k, v in second.items() if k not in volatile}
    assert first["report_id"] != second["report_id"]


def test_builder_accepts_a_plain_assessment_dictionary(tmp_path):
    _, record = build(tmp_path, result=CASES["Consistent"]().to_dict())
    assert record["verdict"] == "Consistent"


def test_record_without_layer0_information_still_cites_the_policy(tmp_path):
    _, record = build(tmp_path, package_info={})
    assert record["policy"]["line_start"] is None
    assert "SECURITY.md" in record["explanation"]
    assert "https://pypi.org/project/synthetic-pkg/" in record["source_references"]


def test_each_verdict_can_be_stored(tmp_path):
    for verdict in VERDICT_CATEGORIES:
        _, record = build(tmp_path, result=CASES[verdict]())
        assert record["verdict"] == verdict and record["explanation"]


# ---------------------------------------------------------------------------
# Report storage and index
# ---------------------------------------------------------------------------

def test_index_is_appended_never_rewritten(tmp_path):
    builder, first = build(tmp_path)
    index = tmp_path / "reports_index.jsonl"
    first_line = index.read_text(encoding="utf-8").splitlines()[0]

    builder.build_and_save(CASES["Consistent"](), PACKAGE_INFO, [], commitment(), None)
    lines = index.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == first_line
    entry = json.loads(lines[0])
    assert entry["report_id"] == first["report_id"]
    assert entry["package_name"] == "synthetic-pkg"
    assert entry["verdict"] == "Partially Consistent"
    assert entry["rules_triggered"] == ["R1", "R3"]
    assert entry["report_file"] == f"{first['report_id']}.json"
    assert json.loads(lines[1])["verdict"] == "Consistent"


def test_index_timestamps_are_real_and_ids_unique(tmp_path):
    storage = ResultStorage(tmp_path)
    before = datetime.now(timezone.utc)
    saved = [storage.save_report({"package_name": "synthetic-pkg", "verdict": "Consistent"}) for _ in range(5)]
    after = datetime.now(timezone.utc)

    assert len({s.report_id for s in saved}) == 5
    assert len(list(tmp_path.glob("*.json"))) == 5
    entries = storage.list_reports()
    assert [e["report_id"] for e in entries] == [s.report_id for s in saved]
    for entry in entries:
        assert before <= datetime.fromisoformat(entry["created_at"]) <= after


def test_load_and_list_helpers(tmp_path):
    storage = ResultStorage(tmp_path)
    a = storage.save_report({"package_name": "alpha", "verdict": "Consistent", "explanation": "x"})
    storage.save_report({"package_name": "beta", "verdict": "Insufficient Evidence"})
    assert [e["package_name"] for e in storage.list_reports()] == ["alpha", "beta"]
    assert [e["package_name"] for e in storage.list_reports("beta")] == ["beta"]
    loaded = storage.load_report(a.report_id)
    assert loaded["report_id"] == a.report_id and loaded["explanation"] == "x"
    assert loaded["created_at"] == a.created_at


def test_load_report_rejects_unknown_and_unsafe_ids(tmp_path):
    storage = ResultStorage(tmp_path)
    with pytest.raises(FileNotFoundError):
        storage.load_report("missing_report")
    for bad in ("../secret", "a/b", "..", ""):
        with pytest.raises(ValueError):
            storage.load_report(bad)


def test_empty_storage_lists_nothing(tmp_path):
    assert ResultStorage(tmp_path / "new" / "reports").list_reports() == []


def test_corrupt_index_line_is_reported(tmp_path):
    storage = ResultStorage(tmp_path)
    storage.save_report({"package_name": "alpha", "verdict": "Consistent"})
    with open(storage.index_file, "a", encoding="utf-8") as handle:
        handle.write("not json\n")
    with pytest.raises(ValueError, match="line 2"):
        storage.list_reports()


def test_package_name_is_sanitised_in_file_names(tmp_path):
    saved = ResultStorage(tmp_path).save_report({"package_name": "../evil name", "verdict": "Consistent"})
    assert saved.path.parent == tmp_path
    assert saved.path.name.startswith("evil_name_")


R2 = vulnerability("CVE-2024-0002", "2023-02-01T00:00:00Z", ["1.0.0", "2.0.0"], ["2.0.1"])


@pytest.mark.parametrize(
    "vulns, verdict, listed",
    [
        ([R3, R4], "Potential Inconsistency", ["CVE-2024-0003", "CVE-2024-0004", "rule R3", "rule R4"]),
        ([R1, R4], "Insufficient Evidence", ["CVE-2024-0001", "CVE-2024-0004", "rule R1", "rule R4"]),
        ([R2, R4], "Partially Consistent", ["CVE-2024-0002", "CVE-2024-0004", "rule R2", "rule R4"]),
        ([R1, R2, R3, R4], "Partially Consistent",
         ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003", "CVE-2024-0004",
          "rule R1", "rule R2", "rule R3", "rule R4"]),
        ([R1], "Consistent", ["CVE-2024-0001", "rule R1"]),
    ],
)
def test_every_verdict_lists_all_existing_rule_records(vulns, verdict, listed):
    result = assess(commitment(), vulns)
    assert result.verdict.value == verdict
    text = ExplanationGenerator().generate_explanation(result, commitment(), None, [])
    for fact in listed:
        assert fact in text, fact


def test_potential_inconsistency_with_r4_records_mentions_them():
    text = ExplanationGenerator().generate_explanation(assess(commitment(), [R3, R4]), commitment(), None, [])
    assert "does not outweigh the confirmed R3 finding" in text
