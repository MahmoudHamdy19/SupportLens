"""Krippendorff's alpha, rating-sheet preparation and ingestion (synthetic ratings and explanations)."""

import csv
import json

import pytest

from src.data_pipeline.constants import VERDICTS
from src.data_pipeline.io_utils import DataError, write_jsonl
from src.evaluation import expert
from src.evaluation.cli import expert_main
from supportlens_synthetic import gold_record

# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": four observers, twelve units,
# nominal alpha = 0.743 and interval alpha = 0.849 (published values).
TEXTBOOK = {
    "A": "1 2 3 3 2 1 4 1 2 * * *",
    "B": "1 2 3 3 2 2 4 1 2 5 * 3",
    "C": "* 3 3 3 2 3 4 2 2 5 1 *",
    "D": "1 2 3 3 2 4 4 1 2 5 1 *",
}


def _units(rows):
    columns = zip(*[[None if x == "*" else float(x) for x in row.split()] for row in rows.values()])
    return [list(c) for c in columns]


def test_krippendorff_matches_the_published_example():
    units = _units(TEXTBOOK)
    assert expert.krippendorff_alpha(units, "nominal") == pytest.approx(0.743, abs=5e-4)
    assert expert.krippendorff_alpha(units, "interval") == pytest.approx(0.849, abs=5e-4)


def test_krippendorff_simple_cases():
    assert expert.krippendorff_alpha([[1, 1], [3, 3], [5, 5]]) == pytest.approx(1.0)
    # Two raters who disagree maximally and systematically: alpha below zero.
    assert expert.krippendorff_alpha([[1, 5], [5, 1], [1, 5], [5, 1]]) < 0
    # By hand: units [1,2] and [2,3]. D_o = (1/4) * (2*1/1 + 2*1/1) = 1; D_e = (1/12) * sum over ordered pairs.
    # Values 1,2,2,3: squared differences summed over ordered pairs = 2 * (1+1+4+0+1+1) = 16 -> D_e = 16/12.
    assert expert.krippendorff_alpha([[1, 2], [2, 3]]) == pytest.approx(1 - 1 / (16 / 12))


def test_krippendorff_handles_missing_values_and_undefined_cases():
    complete = [[3, 3, 4], [2, 2, 2], [5, 4, 5], [1, 1, 2]]
    with_missing = [row[:] for row in complete] + [[4, None, None], [None, None, None]]
    assert expert.krippendorff_alpha(with_missing) == pytest.approx(expert.krippendorff_alpha(complete))
    assert expert.krippendorff_alpha([[3, 3], [3, 3]]) is None  # no variation: undefined
    assert expert.krippendorff_alpha([[1, None], [None, 2]]) is None  # nothing pairable


# ----------------------------------------------------------------------------- sheets


def _predictions(system="qwen_finetuned_lora"):
    rows, policies = [], {}
    n = 0
    for verdict, count in ((VERDICTS[0], 8), (VERDICTS[1], 2), (VERDICTS[2], 7), (VERDICTS[3], 9), (VERDICTS[4], 0)):
        for _ in range(count):
            n += 1
            gold = gold_record(n, "Consistent")
            policies[gold["package_name"]] = gold["policy_text"]
            rows.append(
                {
                    "package_name": gold["package_name"],
                    "system": system,
                    "verdict": verdict,
                    "commitment": {"commitment_text": gold["commitment"]["commitment_text"]},
                    "explanation": f"Synthetic explanation {n}: {verdict}.",
                    "evidence_trail": [f"synthetic evidence {n}"],
                }
            )
    rows.append({**rows[0], "system": "regex_baseline"})  # another system must be ignored
    return rows, policies


def test_selection_samples_up_to_six_per_predicted_category_and_reports_shortfalls():
    rows, policies = _predictions()
    items, availability = expert.select_items(rows, policies, system="qwen_finetuned_lora", per_category=6, seed=42)
    counts = {label: sum(1 for i in items if i["predicted_verdict"] == label) for label in VERDICTS}
    assert counts == {VERDICTS[0]: 6, VERDICTS[1]: 2, VERDICTS[2]: 6, VERDICTS[3]: 6, VERDICTS[4]: 0}
    assert availability[VERDICTS[1]] == {"requested": 6, "available": 2, "sampled": 2, "shortfall": 4}
    assert availability[VERDICTS[4]]["sampled"] == 0
    assert len({i["item_id"] for i in items}) == len(items) == 20
    again, _ = expert.select_items(rows, policies, system="qwen_finetuned_lora", per_category=6, seed=42)
    assert [i["package_name"] for i in again] == [i["package_name"] for i in items]


def test_sheets_are_blind_and_independent(tmp_path):
    rows, policies = _predictions()
    items, _ = expert.select_items(rows, policies, system="qwen_finetuned_lora")
    paths = expert.write_rating_sheets(items, tmp_path, 2)
    contents = [p.read_text(encoding="utf-8-sig") for p in paths]
    for text in contents:
        assert "Insufficient Evidence" in text or "Consistent" in text  # explanations mention their own verdict
        for column in ("item_id", "explanation", "policy_snippet", "cited_evidence", "correctness", "completeness", "traceability"):
            assert column in text.splitlines()[0]
        assert "gold" not in text.lower() and "ground" not in text.lower()
    order = [[r["item_id"] for r in csv.DictReader(open(p, encoding="utf-8-sig", newline=""))] for p in paths]
    assert sorted(order[0]) == sorted(order[1]) and order[0] != order[1]


def test_snippet_surrounds_the_commitment_or_falls_back():
    policy = "A" * 1000 + "COMMITMENT" + "B" * 1000
    snippet = expert.snippet_for(policy, "COMMITMENT")
    assert "COMMITMENT" in snippet and snippet.startswith("...") and snippet.endswith("...") and len(snippet) < 900
    assert expert.snippet_for("short policy", None) == "short policy"


# ----------------------------------------------------------------------------- ingest


def _prepared(tmp_path, n_raters=3):
    rows, policies = _predictions()
    items, availability = expert.select_items(rows, policies, system="qwen_finetuned_lora")
    paths = expert.write_rating_sheets(items, tmp_path / "sheets", n_raters)
    manifest = {
        "availability": availability,
        "items": [{k: i[k] for k in ("item_id", "package_name", "system", "predicted_verdict")} for i in items],
    }
    return manifest, paths


def _fill(path, scores):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows, names = list(reader), reader.fieldnames
    for row in rows:
        row.update(scores(row["item_id"]))
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _score(offset):
    def scores(item_id):
        base = int(item_id[1:]) % 3
        return {"correctness": str(3 + base), "completeness": str(2 + base + offset), "traceability": str(4)}
    return scores


def test_ingest_computes_statistics_and_agreement(tmp_path):
    manifest, paths = _prepared(tmp_path, 3)
    for number, path in enumerate(paths):
        _fill(path, _score(0 if number < 2 else 1))
    summary = expert.ingest_ratings(paths, manifest)
    assert summary["n_raters"] == 3 and summary["n_items"] == 20
    assert summary["dimensions"]["traceability"]["mean"] == 4.0
    assert summary["dimensions"]["traceability"]["std"] == 0.0
    assert summary["dimensions"]["traceability"]["krippendorff_alpha_interval"] is None  # no variation at all
    assert summary["dimensions"]["correctness"]["krippendorff_alpha_interval"] == pytest.approx(1.0)
    assert summary["dimensions"]["completeness"]["krippendorff_alpha_interval"] < 1.0
    expected_mean = sum(3 + int(i["item_id"][1:]) % 3 for i in manifest["items"]) / 20
    assert summary["dimensions"]["correctness"]["mean"] == pytest.approx(expected_mean)
    assert summary["dimensions"]["correctness"]["n_ratings"] == 60
    json.dumps(summary)  # serialisable


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"correctness": "6", "completeness": "3", "traceability": "3"}, "outside 1-5"),
        ({"correctness": "0", "completeness": "3", "traceability": "3"}, "outside 1-5"),
        ({"correctness": "3.5", "completeness": "3", "traceability": "3"}, "non-integer"),
        ({"correctness": "", "completeness": "3", "traceability": "3"}, "missing"),
        ({"correctness": "good", "completeness": "3", "traceability": "3"}, "non-integer"),
    ],
)
def test_ingest_refuses_missing_or_out_of_range_ratings(tmp_path, bad, message):
    manifest, paths = _prepared(tmp_path, 2)
    for path in paths:
        _fill(path, _score(0))
    first_id = manifest["items"][0]["item_id"]
    _fill(paths[1], lambda item_id: bad if item_id == first_id else _score(0)(item_id))
    with pytest.raises(DataError, match=message):
        expert.ingest_ratings(paths, manifest)


def test_ingest_requires_two_raters_all_items_and_distinct_ids(tmp_path):
    manifest, paths = _prepared(tmp_path, 2)
    for path in paths:
        _fill(path, _score(0))
    with pytest.raises(DataError, match="At least 2"):
        expert.ingest_ratings(paths[:1], manifest)
    lines = paths[1].read_text(encoding="utf-8-sig").splitlines()
    paths[1].write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8-sig")  # drops the last item (rows may span lines)
    with pytest.raises(DataError):
        expert.ingest_ratings(paths, manifest)
    manifest2, paths2 = _prepared(tmp_path / "again", 2)
    for path in paths2:
        _fill(path, _score(0))
    text = paths2[1].read_text(encoding="utf-8-sig").replace("rater2", "rater1")
    paths2[1].write_text(text, encoding="utf-8-sig")
    with pytest.raises(DataError, match="more than one sheet"):
        expert.ingest_ratings(paths2, manifest2)


def test_cli_prepare_then_ingest(tmp_path, capsys):
    rows, policies = _predictions()
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, rows)
    ground_truth = tmp_path / "test.jsonl"
    write_jsonl(ground_truth, [{"package_name": name, "policy_text": text} for name, text in policies.items()])
    out_dir = tmp_path / "expert"
    assert expert_main(["--prepare", "--predictions", str(predictions), "--ground-truth", str(ground_truth),
                        "--out-dir", str(out_dir), "--raters", "2"]) == 0
    printed = capsys.readouterr().out
    assert "fewer than requested" in printed  # the shortfall is reported
    sheets = sorted(out_dir.glob("rating_sheet_rater*.csv"))
    assert len(sheets) == 2
    for sheet in sheets:
        _fill(sheet, _score(0))
    output = tmp_path / "expert_evaluation.json"
    assert expert_main(["--ingest", "--out-dir", str(out_dir), "--ratings", *map(str, sheets), "--output", str(output)]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["n_raters"] == 2 and "ratings" in document and document["sampling"][VERDICTS[1]]["shortfall"] == 4
    # an incomplete sheet makes the ingestion fail without writing anything
    _fill(sheets[0], lambda item_id: {"correctness": ""})
    output.unlink()
    assert expert_main(["--ingest", "--out-dir", str(out_dir), "--ratings", *map(str, sheets), "--output", str(output)]) == 2
    assert not output.exists()
