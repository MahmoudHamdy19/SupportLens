"""Cohen's kappa and the agreement report against hand-computed values (synthetic annotations)."""

import pytest

from src.data_pipeline.agreement import build_agreement_report, cohens_kappa, confusion_matrix, find_disagreements, token_f1
from src.data_pipeline.annotation import Annotation
from src.data_pipeline.constants import TAU_VALUES, TEMPORAL_VALUES, VERDICTS

A, B = "Consistent", "Insufficient Evidence"


def test_kappa_two_by_two_hand_computed():
    # 50 items: 20 both A, 5 (A,B), 10 (B,A), 15 both B.
    first = [A] * 20 + [A] * 5 + [B] * 10 + [B] * 15
    second = [A] * 20 + [B] * 5 + [A] * 10 + [B] * 15
    result = cohens_kappa(first, second, [A, B])
    assert result["observed_agreement"] == pytest.approx(0.70)
    assert result["expected_agreement"] == pytest.approx(0.50)  # 0.5*0.6 + 0.5*0.4
    assert result["cohens_kappa"] == pytest.approx(0.40)
    assert result["confusion_matrix"] == [[20, 5], [10, 15]]
    assert not result["degenerate"]


def test_kappa_perfect_and_chance_level():
    labels = list(VERDICTS[:3])
    items = [labels[i % 3] for i in range(30)]
    assert cohens_kappa(items, items, labels)["cohens_kappa"] == pytest.approx(1.0)
    # Second rater cycles with a shift so that the label marginals are equal but no item agrees.
    shifted = [labels[(i + 1) % 3] for i in range(30)]
    result = cohens_kappa(items, shifted, labels)
    assert result["observed_agreement"] == 0.0
    assert result["cohens_kappa"] == pytest.approx(-0.5)  # (0 - 1/3) / (1 - 1/3)


def test_kappa_degenerate_single_label_is_flagged():
    result = cohens_kappa([A] * 4, [A] * 4, [A, B])
    assert result["cohens_kappa"] == 1.0 and result["degenerate"]


def test_kappa_rejects_unknown_label_and_length_mismatch():
    with pytest.raises(ValueError):
        cohens_kappa([A], ["Other"], [A, B])
    with pytest.raises(ValueError):
        cohens_kappa([A, A], [A], [A, B])


def test_confusion_matrix_orientation():
    matrix = confusion_matrix([A, A, B], [B, A, B], [A, B])
    assert matrix == [[1, 1], [0, 1]]  # rows: first rater, columns: second rater


def test_token_f1():
    assert token_f1("only version 2.x", "only version 2.x") == 1.0
    assert token_f1("a b", "c d") == 0.0
    assert token_f1("a b c", "a b") == pytest.approx(0.8)


def _annotation(name, verdict, tau="explicit_range", temporal="vulnerability_publication_time", versions=("2.x",), text="t"):
    return Annotation(
        name,
        {"commitment_text": text, "version_expression": list(versions), "support_status": tau, "temporal_reference": temporal},
        verdict,
    )


def test_agreement_report_fields_and_disagreements():
    verdicts1 = [A, A, A, B, B, B, A, B]
    verdicts2 = [A, A, B, B, B, A, A, B]
    first = {f"p{i}": _annotation(f"p{i}", v) for i, v in enumerate(verdicts1)}
    second = {f"p{i}": _annotation(f"p{i}", v) for i, v in enumerate(verdicts2)}
    # field-level differences: p0 differs in tau, p1 in the temporal reference, p2 in the version tokens
    second["p0"] = _annotation("p0", A, tau="latest_only")
    second["p1"] = _annotation("p1", A, temporal="release_time")
    second["p2"] = _annotation("p2", B, versions=("3.x",))

    report = build_agreement_report(first, second, tau_values=TAU_VALUES, temporal_values=TEMPORAL_VALUES)

    # verdicts: 6 of 8 agree; marginals A:4/4 -> pe = (4/8*4/8) + (4/8*4/8) = 0.5; kappa = (0.75 - 0.5) / 0.5
    assert report["n_records"] == 8
    assert report["observed_agreement"] == pytest.approx(0.75)
    assert report["cohens_kappa"] == pytest.approx(0.5)
    assert report["percent_agreement"] == pytest.approx(75.0)
    assert report["meets_target"] is False
    assert report["fields"]["support_status"]["percent_agreement"] == pytest.approx(87.5)
    assert report["fields"]["temporal_reference"]["percent_agreement"] == pytest.approx(87.5)
    assert report["fields"]["version_expression"]["percent_agreement"] == pytest.approx(87.5)
    matrix = report["confusion_matrix"]["matrix"]
    assert matrix[VERDICTS.index(A)][VERDICTS.index(A)] == 3
    assert matrix[VERDICTS.index(A)][VERDICTS.index(B)] == 1
    disagreeing = {d["package_name"]: set(d["fields"]) for d in report["disagreements"]}
    assert disagreeing == {
        "p0": {"support_status"},
        "p1": {"temporal_reference"},
        "p2": {"verdict", "version_expression"},
        "p5": {"verdict"},
    }
    assert report["n_disagreements"] == 4


def test_whitespace_and_case_only_differences_are_agreement():
    first = {"p": _annotation("p", A, versions=("Latest",), text="Only  the latest")}
    second = {"p": _annotation("p", A, versions=("latest",), text="Only the latest")}
    assert find_disagreements(first, second) == {}
