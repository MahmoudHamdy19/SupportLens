"""
Layer 3 unit tests: Algorithm 1 (rules R1 to R5), priority order and aggregation (Table 3.1).

All fixtures below are SYNTHETIC. They are built inline to exercise one rule at a time and
are not real packages, advisories or ground-truth records. No network access is used.
"""

import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.layer3_rule_assessment import (
    REASONABLE_TIMEFRAME_DAYS,
    VERDICT_CATEGORIES,
    AssessmentRuleEngine,
)

NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------

def release(version, day):
    return {"version": version, "tag": version, "published_at": f"{day}T00:00:00Z", "source": "pypi"}


RELEASES = [
    release("1.0.0", "2022-01-01"),
    release("1.1.0", "2022-06-01"),
    release("2.0.0", "2023-01-01"),
    release("2.0.1", "2023-03-01"),
    release("3.0.0", "2023-09-01"),
]


def commitment(tau="explicit_range", expression=("1.x", "2.x"), temporal="vulnerability_publication_time",
               text="Versions 1.x and 2.x receive security fixes."):
    return {
        "commitment_text": text,
        "support_status": tau,
        "version_expression": list(expression),
        "temporal_reference": temporal,
        "is_measurable": tau != "ambiguous" and not (tau == "explicit_range" and not expression),
    }


def vulnerability(vid, published, affected, fixed=(), **extra):
    record = {
        "id": vid,
        "cve_ids": [vid] if vid.startswith("CVE-") else [],
        "ghsa_ids": [],
        "published_at": published,
        "affected_versions": list(affected),
        "affected_ranges": [],
        "fixed_versions": list(fixed),
        "is_patched": bool(fixed),
        "advisory_url": None,
    }
    record.update(extra)
    return record


def iso(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def evaluate(c, vulns, releases=RELEASES, **kwargs):
    kwargs.setdefault("reference_time", NOW)
    kwargs.setdefault("package_name", "synthetic-pkg")
    return AssessmentRuleEngine().evaluate(c, vulns, releases, **kwargs)


def reasons(result):
    return [entry.reason for entry in result.evidence_log]


# ---------------------------------------------------------------------------
# R5: Unmeasurable Commitment
# ---------------------------------------------------------------------------

def test_r5_ambiguous_commitment_type():
    result = evaluate(commitment(tau="ambiguous", expression=()), [vulnerability("CVE-2024-0001", "2023-02-01", ["2.0.0"])])
    assert result.verdict.value == "Unmeasurable Commitment"
    assert result.rules_triggered == ["R5"]
    assert result.is_measurable is False
    assert reasons(result) == ["ambiguous_commitment_type"]


def test_r5_explicit_range_without_version_expression():
    result = evaluate(commitment(tau="explicit_range", expression=()), [vulnerability("CVE-2024-0001", "2023-02-01", ["2.0.0"])])
    assert result.verdict.value == "Unmeasurable Commitment"
    assert reasons(result) == ["version_scope_not_normalisable"]


def test_r5_version_expression_not_normalisable():
    result = evaluate(commitment(expression=("see the table below",)), [vulnerability("CVE-2024-0001", "2023-02-01", ["2.0.0"])])
    assert result.verdict.value == "Unmeasurable Commitment"
    assert reasons(result) == ["version_scope_not_normalisable"]


def test_r5_missing_commitment_type():
    result = evaluate({"commitment_text": "x", "version_expression": ["1.x"]}, [vulnerability("CVE-2024-0001", "2023-02-01", ["1.0.0"])])
    assert result.verdict.value == "Unmeasurable Commitment"


def test_r5_trail_starts_with_the_policy_statement():
    result = evaluate(commitment(tau="ambiguous", expression=(), text="We support what we can."), None, None)
    assert result.evidence_trail[0] == 'Commitment (t): "We support what we can."'


def test_one_unparsable_token_does_not_trigger_r5():
    result = evaluate(commitment(expression=("2.x", "LTS-ish")), [vulnerability("CVE-2024-0001", "2023-02-01", ["2.0.0"], ["2.0.1"])])
    assert result.verdict.value == "Consistent"
    assert any("LTS-ish" in line for line in result.evidence_trail)


@pytest.mark.parametrize("tau", ["all_versions", "latest_only"])
def test_empty_expression_is_completed_from_the_commitment_type(tau):
    result = evaluate(commitment(tau=tau, expression=()), [vulnerability("CVE-2024-0001", "2023-02-01", ["2.0.0"], ["2.0.1"])])
    assert result.verdict.value != "Unmeasurable Commitment"
    assert result.assessed_vulns_count == 1


# ---------------------------------------------------------------------------
# Priority order: R5 before R4
# ---------------------------------------------------------------------------

def test_priority_r5_before_r4():
    failing = {"osv": {"status": "FAILURE", "count": 0, "error": "HTTP 500"}}
    result = evaluate(
        commitment(tau="ambiguous", expression=()), None, None,
        source_diagnostics=failing, mapping_uncertain=True,
    )
    assert result.verdict.value == "Unmeasurable Commitment"
    assert result.rules_triggered == ["R5"]
    assert reasons(result) == ["ambiguous_commitment_type"]


# ---------------------------------------------------------------------------
# R4 pre-check: evidence sufficiency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing", [None, []])
def test_r4_empty_vulnerability_list_means_v_is_missing(missing):
    result = evaluate(commitment(), missing)
    assert result.verdict.value == "Insufficient Evidence"
    assert result.rules_triggered == ["R4"]
    assert reasons(result) == ["vulnerabilities_missing"]
    assert any("Vulnerability evidence V" in line for line in result.evidence_trail)


@pytest.mark.parametrize("missing", [None, []])
def test_r4_missing_release_history(missing):
    result = evaluate(commitment(), [vulnerability("CVE-2024-0001", "2023-02-01", ["2.0.0"])], releases=missing)
    assert result.verdict.value == "Insufficient Evidence"
    assert reasons(result) == ["releases_missing"]


@pytest.mark.parametrize("status", ["FAILURE", "RATE_LIMITED"])
def test_r4_source_failure(status):
    diagnostics = {"osv": {"status": "SUCCESS", "count": 3}, "github_advisory": {"status": status, "error": "HTTP 403"}}
    result = evaluate(commitment(), [vulnerability("CVE-2024-0001", "2023-02-01", ["2.0.0"], ["2.0.1"])], source_diagnostics=diagnostics)
    assert result.verdict.value == "Insufficient Evidence"
    assert reasons(result) == ["source_failure"]
    assert "github_advisory" in result.evidence_log[0].message


def test_non_failing_diagnostics_do_not_halt():
    diagnostics = {
        "pypi": {"status": "NOT_PROVIDED"}, "osv": {"status": "SUCCESS"},
        "github_advisory": {"status": "SUCCESS"}, "nvd": {"status": "NO_CVES"},
    }
    result = evaluate(commitment(), [vulnerability("CVE-2024-0001", "2023-02-01", ["2.0.0"], ["2.0.1"])], source_diagnostics=diagnostics)
    assert result.verdict.value == "Consistent"


def test_r4_mapping_uncertain_flag():
    result = evaluate(commitment(), [vulnerability("CVE-2024-0001", "2023-02-01", ["2.0.0"], ["2.0.1"])], mapping_uncertain=True)
    assert result.verdict.value == "Insufficient Evidence"
    assert reasons(result) == ["mapping_uncertain"]


def test_r4_precheck_names_every_reason():
    failing = {"osv": {"status": "FAILURE", "error": "timeout"}}
    result = evaluate(commitment(), None, None, source_diagnostics=failing, mapping_uncertain=True)
    assert reasons(result) == ["vulnerabilities_missing", "releases_missing", "source_failure", "mapping_uncertain"]
    assert len(result.evidence_trail) == 1 + 4


# ---------------------------------------------------------------------------
# R1, R2, R3 per vulnerability
# ---------------------------------------------------------------------------

def test_r1_fixed_on_every_covered_version():
    advisory = "https://example.org/advisories/synthetic-1"
    vuln = vulnerability("CVE-2024-0001", "2023-02-01T00:00:00Z", ["2.0.0"], ["2.0.1"],
                         ghsa_ids=["GHSA-aaaa-bbbb-cccc"], advisory_url=advisory)
    result = evaluate(commitment(), [vuln])
    assert result.verdict.value == "Consistent"
    assert result.rules_triggered == ["R1"]
    assert result.rule_counts == {"R1": 1, "R2": 0, "R3": 0, "R4": 0}
    entry = result.evidence_log[0]
    assert (entry.rule_id, entry.reason) == ("R1", "fixed_on_all_covered_versions")
    assert entry.advisory_url == advisory
    assert entry.fix_release_dates == {"2.0.1": "2023-03-01T00:00:00+00:00"}
    assert entry.affected_release_dates == {"2.0.0": "2023-01-01T00:00:00+00:00"}
    trail = " ".join(result.evidence_trail)
    for fact in ("CVE-2024-0001", "GHSA-aaaa-bbbb-cccc", "2.0.0 (released 2023-01-01)",
                 "2.0.1 (released 2023-03-01)", "published 2023-02-01"):
        assert fact in trail


def test_r2_fixed_on_some_release_lines_only():
    vuln = vulnerability("CVE-2024-0002", "2023-02-01T00:00:00Z", ["1.0.0", "1.1.0", "2.0.0"], ["2.0.1"])
    result = evaluate(commitment(), [vuln])
    assert result.verdict.value == "Partially Consistent"
    assert result.rules_triggered == ["R2"]
    entry = result.evidence_log[0]
    assert entry.reason == "fixed_on_some_release_lines"
    assert entry.unremediated_versions == ["1.0.0", "1.1.0"]


def test_r2_fixed_on_some_minor_release_lines_only():
    vuln = vulnerability("CVE-2024-0002-minor", "2023-02-01T00:00:00Z", ["2.1.0", "2.2.0"], ["2.2.5"])
    comm = commitment(expression=("2.1.x", "2.2.x"), text="Versions 2.1.x and 2.2.x receive security fixes.")
    rels = [
        release("2.1.0", "2023-01-01"),
        release("2.2.0", "2023-01-01"),
        release("2.2.5", "2023-03-01"),
    ]
    result = evaluate(comm, [vuln], releases=rels)
    assert result.verdict.value == "Partially Consistent"
    assert result.rules_triggered == ["R2"]
    entry = result.evidence_log[0]
    assert entry.reason == "fixed_on_some_release_lines"
    assert entry.unremediated_versions == ["2.1.0"]


def test_r2_bounded_range_inside_one_major_uses_minor_lines():
    vuln = vulnerability("CVE-2024-0002-range", "2023-02-01T00:00:00Z", ["2.1.0", "2.2.0"], ["2.2.5"])
    comm = commitment(expression=(">=2.1,<2.3",), text="Versions >=2.1,<2.3 receive security fixes.")
    rels = [
        release("2.1.0", "2023-01-01"),
        release("2.2.0", "2023-01-01"),
        release("2.2.5", "2023-03-01"),
    ]
    result = evaluate(comm, [vuln], releases=rels)
    assert result.verdict.value == "Partially Consistent"
    assert result.evidence_log[0].unremediated_versions == ["2.1.0"]


def test_open_or_cross_major_ranges_stay_on_major_lines():
    from src.layer3_rule_assessment.rule_engine import _is_minor_line_expression

    assert not _is_minor_line_expression(">=2.1")
    assert not _is_minor_line_expression(">=1.0,<3.0")
    assert _is_minor_line_expression(">=2.1, <2.3")


def test_r3_unfixed_beyond_timeframe():
    vuln = vulnerability("CVE-2024-0003", "2023-02-01T00:00:00Z", ["2.0.0"])
    result = evaluate(commitment(), [vuln])
    assert result.verdict.value == "Potential Inconsistency"
    assert result.rules_triggered == ["R3"]
    entry = result.evidence_log[0]
    assert entry.reason == "unfixed_beyond_timeframe"
    assert entry.days_since_publication == (NOW - datetime(2023, 2, 1, tzinfo=timezone.utc)).days


def test_unfixed_but_within_timeframe_is_r4():
    vuln = vulnerability("CVE-2024-0004", iso(NOW - timedelta(days=10)), ["2.0.0"])
    result = evaluate(commitment(), [vuln])
    assert result.verdict.value == "Insufficient Evidence"
    assert result.evidence_log[0].reason == "unfixed_within_timeframe"
    assert result.evidence_log[0].days_since_publication == 10


@pytest.mark.parametrize(
    "age, rule",
    [
        (timedelta(days=89), "R4"),
        (timedelta(days=90), "R4"),
        (timedelta(days=90, seconds=1), "R3"),
        (timedelta(days=91), "R3"),
    ],
)
def test_ninety_day_boundary(age, rule):
    assert REASONABLE_TIMEFRAME_DAYS == 90
    vuln = vulnerability("CVE-2024-0005", iso(NOW - age), ["2.0.0"])
    result = evaluate(commitment(), [vuln])
    assert result.evidence_log[0].rule_id == rule
    assert result.reasonable_timeframe_days == 90


def test_unfixed_without_publication_date_is_r4():
    vuln = vulnerability("CVE-2024-0006", None, ["2.0.0"])
    result = evaluate(commitment(), [vuln])
    assert result.verdict.value == "Insufficient Evidence"
    assert result.evidence_log[0].reason == "publication_date_missing"


def test_fixed_vulnerability_without_publication_date_is_still_r1_for_a_fixed_range():
    vuln = vulnerability("CVE-2024-0007", None, ["2.0.0"], ["2.0.1"])
    result = evaluate(commitment(), [vuln])
    assert result.verdict.value == "Consistent"


def test_fixed_identifier_that_is_not_a_release_version_is_r4():
    vuln = vulnerability("CVE-2024-0008", "2023-02-01T00:00:00Z", ["2.0.0"], ["3f2a9c1d7e5b"])
    result = evaluate(commitment(), [vuln])
    assert result.verdict.value == "Insufficient Evidence"
    assert result.evidence_log[0].reason == "fixed_identifiers_unmapped"


def test_vulnerability_outside_the_commitment_is_ignored():
    inside = vulnerability("CVE-2024-0009", "2023-02-01T00:00:00Z", ["2.0.0"], ["2.0.1"])
    outside = vulnerability("CVE-2024-0010", "2021-02-01T00:00:00Z", ["0.9.0"])
    result = evaluate(commitment(), [inside, outside])
    assert result.verdict.value == "Consistent"
    assert (result.evaluated_vulns_count, result.assessed_vulns_count, result.out_of_scope_vulns_count) == (2, 1, 1)
    assert any("CVE-2024-0010" in line and "not assessed" in line for line in result.evidence_trail)


# ---------------------------------------------------------------------------
# Version coverage at time T
# ---------------------------------------------------------------------------

def test_latest_is_resolved_at_the_vulnerability_publication_time():
    latest = commitment(tau="latest_only", expression=("latest",))
    # Published 2023-02-01: the latest release at that time is 2.0.0 (2.0.1 follows on 2023-03-01, 3.0.0 later).
    at_head = vulnerability("CVE-2024-0011", "2023-02-01T00:00:00Z", ["2.0.0"], ["2.0.1"])
    # Published 2022-07-01: the latest release at that time is 1.1.0.
    earlier_head = vulnerability("CVE-2024-0012", "2022-07-01T00:00:00Z", ["1.1.0"], ["2.0.0"])
    # 1.1.0 was no longer the latest release on 2023-02-01.
    not_head = vulnerability("CVE-2024-0013", "2023-02-01T00:00:00Z", ["1.1.0"])
    result = evaluate(latest, [at_head, earlier_head, not_head])
    covered = {e.vulnerability_id: e.affected_in_scope for e in result.evidence_log}
    assert covered == {"CVE-2024-0011": ["2.0.0"], "CVE-2024-0012": ["1.1.0"]}
    assert result.out_of_scope_vulns_count == 1


def test_latest_is_not_resolved_against_the_newest_release_overall():
    latest = commitment(tau="latest_only", expression=("latest",))
    only_old_head = vulnerability("CVE-2024-0014", "2023-02-01T00:00:00Z", ["3.0.0"], ["3.0.1"])
    result = evaluate(latest, [only_old_head])
    # 3.0.0 is the newest release overall but did not exist on 2023-02-01.
    assert result.out_of_scope_vulns_count == 1
    assert result.evidence_log[0].reason == "no_vulnerability_in_scope"


def test_latest_without_publication_date_cannot_be_resolved():
    latest = commitment(tau="latest_only", expression=("latest",))
    result = evaluate(latest, [vulnerability("CVE-2024-0015", None, ["2.0.0"], ["2.0.1"])])
    assert result.verdict.value == "Insufficient Evidence"
    assert result.evidence_log[0].reason == "publication_date_missing"


def test_release_time_uses_the_release_date_of_each_affected_version():
    releases = RELEASES + [release("1.1.1", "2023-06-01")]
    latest = commitment(tau="latest_only", expression=("latest",), temporal="release_time")
    # 2.0.1 was the latest release when it was published; 1.1.1 (a backport) was not.
    vuln = vulnerability("CVE-2024-0016", "2023-07-01T00:00:00Z", ["2.0.1", "1.1.1"], ["3.0.0"])
    result = evaluate(latest, [vuln], releases=releases)
    assert result.evidence_log[0].affected_in_scope == ["2.0.1"]


def test_release_time_with_unknown_release_date_is_r4():
    latest = commitment(tau="latest_only", expression=("latest",), temporal="release_time")
    result = evaluate(latest, [vulnerability("CVE-2024-0017", "2023-07-01T00:00:00Z", ["9.9.9"], ["10.0.0"])])
    assert result.verdict.value == "Insufficient Evidence"
    assert result.evidence_log[0].reason == "release_date_missing"


def test_unknown_temporal_reference_cannot_resolve_latest():
    latest = commitment(tau="latest_only", expression=("latest",), temporal="end_of_life")
    result = evaluate(latest, [vulnerability("CVE-2024-0018", "2023-02-01T00:00:00Z", ["2.0.0"], ["2.0.1"])])
    assert result.evidence_log[0].reason == "temporal_reference_unknown"


def test_all_versions_covers_everything():
    result = evaluate(commitment(tau="all_versions", expression=("all",)),
                      [vulnerability("CVE-2024-0019", "2021-02-01T00:00:00Z", ["0.9.0"], ["0.9.1"])])
    assert result.verdict.value == "Consistent"
    assert result.evidence_log[0].affected_in_scope == ["0.9.0"]


# ---------------------------------------------------------------------------
# Affected ranges
# ---------------------------------------------------------------------------

def test_ranges_are_expanded_to_known_releases():
    vuln = vulnerability("CVE-2024-0020", "2023-02-01T00:00:00Z", [], ["2.0.1"],
                         affected_ranges=[{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "2.0.1"}]}])
    result = evaluate(commitment(expression=("2.x",)), [vuln])
    assert result.verdict.value == "Consistent"
    assert result.evidence_log[0].affected_in_scope == ["2.0.0"]


def test_multi_interval_range_is_split_into_intervals():
    events = [{"introduced": "0"}, {"fixed": "1.1.0"}, {"introduced": "2.0.0"}, {"fixed": "2.0.1"}]
    vuln = vulnerability("CVE-2024-0021", "2023-02-01T00:00:00Z", [], ["1.1.0", "2.0.1"],
                         affected_ranges=[{"type": "ECOSYSTEM", "events": events}])
    result = evaluate(commitment(), [vuln])
    assert result.evidence_log[0].affected_in_scope == ["1.0.0", "2.0.0"]
    assert result.verdict.value == "Consistent"


def test_git_ranges_cannot_be_mapped_to_releases():
    vuln = vulnerability("CVE-2024-0022", "2023-02-01T00:00:00Z", [], [],
                         affected_ranges=[{"type": "GIT", "events": [{"introduced": "abc123"}, {"fixed": "def456"}]}])
    result = evaluate(commitment(), [vuln])
    assert result.verdict.value == "Insufficient Evidence"
    assert result.evidence_log[0].reason == "affected_versions_unmapped"


# ---------------------------------------------------------------------------
# Aggregation (Table 3.1)
# ---------------------------------------------------------------------------

OLD = "2023-02-01T00:00:00Z"

BUILDERS = {
    "r1": lambda n: vulnerability(f"CVE-2024-1{n:03d}", OLD, ["2.0.0"], ["2.0.1"]),
    "r2": lambda n: vulnerability(f"CVE-2024-1{n:03d}", OLD, ["1.0.0", "2.0.0"], ["2.0.1"]),
    "r3": lambda n: vulnerability(f"CVE-2024-1{n:03d}", OLD, ["2.0.0"]),
    "r4": lambda n: vulnerability(f"CVE-2024-1{n:03d}", iso(NOW - timedelta(days=10)), ["2.0.0"]),
    "out": lambda n: vulnerability(f"CVE-2024-1{n:03d}", OLD, ["0.9.0"], ["1.0.0"]),
}


@pytest.mark.parametrize(
    "outcomes, expected",
    [
        (["r1"], "Consistent"),
        (["r1", "r1"], "Consistent"),
        (["r1", "out"], "Consistent"),
        (["r2"], "Partially Consistent"),
        (["r1", "r2"], "Partially Consistent"),
        (["r2", "r3"], "Partially Consistent"),
        (["r2", "r4"], "Partially Consistent"),
        (["r1", "r3"], "Partially Consistent"),
        (["r1", "r2", "r3"], "Partially Consistent"),
        (["r3"], "Potential Inconsistency"),
        (["r3", "r3"], "Potential Inconsistency"),
        (["r3", "out"], "Potential Inconsistency"),
        (["r3", "r4"], "Potential Inconsistency"),
        (["r3", "r4", "r4"], "Potential Inconsistency"),
        (["r3", "r4", "out"], "Potential Inconsistency"),
        (["r1", "r3", "r4"], "Partially Consistent"),
        (["r2", "r3", "r4"], "Partially Consistent"),
        (["r1", "r2", "r4"], "Partially Consistent"),
        (["r1", "r4"], "Insufficient Evidence"),
        (["r4"], "Insufficient Evidence"),
        (["out"], "Insufficient Evidence"),
        (["out", "out"], "Insufficient Evidence"),
    ],
)
def test_aggregation_follows_table_3_1(outcomes, expected):
    vulns = [BUILDERS[name](index) for index, name in enumerate(outcomes)]
    result = evaluate(commitment(), vulns)
    assert result.verdict.value == expected
    assert result.verdict.value in VERDICT_CATEGORIES
    expected_counts = {rule: outcomes.count(rule.lower()) for rule in ("R1", "R2", "R3", "R4")}
    if outcomes == ["out"] or outcomes == ["out", "out"]:
        expected_counts["R4"] = 0
    assert result.rule_counts == expected_counts
    assert result.assessed_vulns_count == sum(expected_counts.values())
    assert result.evaluated_vulns_count == len(outcomes)


def test_no_vulnerability_in_scope_documents_the_open_case():
    result = evaluate(commitment(), [BUILDERS["out"](0)])
    assert result.verdict.value == "Insufficient Evidence"
    assert result.rules_triggered == ["R4"]
    entry = result.evidence_log[0]
    assert entry.reason == "no_vulnerability_in_scope"
    assert "never tested" in entry.message
    assert result.assessed_vulns_count == 0


def test_rules_triggered_lists_each_fired_rule_once():
    result = evaluate(commitment(), [BUILDERS["r1"](0), BUILDERS["r1"](1), BUILDERS["r3"](2)])
    assert result.rules_triggered == ["R1", "R3"]
    assert result.evidence_trail[-1] == "Aggregation (Table 3.1): 2 x R1, 1 x R3 -> Partially Consistent."


# ---------------------------------------------------------------------------
# Determinism and input handling
# ---------------------------------------------------------------------------

def test_output_is_deterministic():
    vulns = [BUILDERS["r1"](0), BUILDERS["r2"](1), BUILDERS["r3"](2), BUILDERS["out"](3)]
    first = evaluate(commitment(), vulns).to_dict()
    second = evaluate(commitment(), vulns).to_dict()
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_inputs_are_not_modified():
    vulns = [BUILDERS["r1"](0), BUILDERS["r2"](1)]
    c = commitment()
    snapshot = copy.deepcopy((c, vulns, RELEASES))
    evaluate(c, vulns)
    assert (c, vulns, RELEASES) == snapshot


def test_naive_reference_time_is_treated_as_utc():
    vulns = [BUILDERS["r3"](0)]
    naive = evaluate(commitment(), vulns, reference_time=NOW.replace(tzinfo=None))
    aware = evaluate(commitment(), vulns, reference_time=NOW)
    assert naive.to_dict() == aware.to_dict()


def test_reference_time_defaults_to_the_current_utc_time():
    before = datetime.now(timezone.utc)
    result = AssessmentRuleEngine().evaluate(commitment(), [BUILDERS["r3"](0)], RELEASES)
    stamp = datetime.fromisoformat(result.reference_time)
    assert before <= stamp <= datetime.now(timezone.utc)


def test_result_exposes_no_confidence_or_ratio():
    data = evaluate(commitment(), [BUILDERS["r1"](0)]).to_dict()
    assert "confidence" not in data and "consistency_ratio" not in data
    assert data["verdict"] in VERDICT_CATEGORIES
    json.dumps(data)


@pytest.mark.parametrize(
    "counts, expected",
    [
        ({"R1": 0, "R2": 0, "R3": 0, "R4": 0}, "Insufficient Evidence"),
        ({"R1": 1, "R2": 0, "R3": 1, "R4": 3}, "Partially Consistent"),
        ({"R1": 0, "R2": 1, "R3": 1, "R4": 0}, "Partially Consistent"),
        ({"R1": 0, "R2": 0, "R3": 1, "R4": 5}, "Potential Inconsistency"),
        ({"R1": 0, "R2": 2, "R3": 0, "R4": 4}, "Partially Consistent"),
        ({"R1": 3, "R2": 0, "R3": 0, "R4": 1}, "Insufficient Evidence"),
        ({"R1": 3, "R2": 0, "R3": 0, "R4": 0}, "Consistent"),
    ],
)
def test_aggregate_uses_the_exact_severity_order(counts, expected):
    assert AssessmentRuleEngine._aggregate(counts).value == expected


def test_confirmed_r3_is_not_hidden_by_r4():
    result = evaluate(commitment(), [BUILDERS["r3"](0), BUILDERS["r4"](1)])
    assert result.verdict.value == "Potential Inconsistency"
    assert result.rule_counts["R3"] == 1 and result.rule_counts["R4"] == 1


def test_nvd_failure_alone_does_not_halt_with_r4():
    """Synthetic: NVD only enriches dates, so its failure must not trigger the R4 pre-check."""
    from datetime import datetime, timezone
    from src.layer3_rule_assessment import AssessmentRuleEngine
    commitment = {"commitment_text": "2.x is supported", "support_status": "explicit_range",
                  "version_expression": ["2.x"], "temporal_reference": "vulnerability_publication_time"}
    releases = [{"version": v, "tag": v, "published_at": d, "source": "synthetic"}
                for v, d in [("2.0.0", "2024-01-01T00:00:00Z"), ("2.0.1", "2024-03-01T00:00:00Z")]]
    vulns = [{"id": "CVE-2024-0001", "osv_id": "GHSA-x", "published_at": "2024-02-01T00:00:00Z",
              "affected_versions": ["2.0.0"], "affected_ranges": [], "fixed_versions": ["2.0.1"], "is_patched": True}]
    diags = {"osv": {"status": "SUCCESS"}, "github_advisory": {"status": "SUCCESS"},
             "pypi": {"status": "SUCCESS"}, "nvd": {"status": "FAILURE", "error": "HTTP 503"}}
    out = AssessmentRuleEngine().evaluate(commitment, vulns, releases, datetime(2025, 1, 1, tzinfo=timezone.utc),
                                          "demo", diags).to_dict()
    assert out["verdict"] == "Consistent"
