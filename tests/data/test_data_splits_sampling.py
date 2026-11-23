"""Stratified splits, small-class handling and the seed-42 sample (synthetic data only)."""

import json
from collections import Counter

import numpy as np
import pytest

from src.data_pipeline.constants import VERDICTS
from src.data_pipeline.cli import sample_main, splits_main
from src.data_pipeline.io_utils import DataError, read_jsonl, write_jsonl
from src.data_pipeline.sampling import (
    InsufficientDataError,
    allocate_proportionally,
    star_tercile_cutoffs,
    stratified_sample,
    stratum_of,
)
from src.data_pipeline.splits import class_allocation, stratified_split
from supportlens_synthetic import dataset, gold_record


# ----------------------------------------------------------------------------- splits


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, (0, 0, 0)),
        (1, (1, 0, 0)),
        (2, (1, 0, 1)),
        (3, (1, 1, 1)),
        (4, (2, 1, 1)),
        (10, (6, 2, 2)),
        (100, (70, 15, 15)),
        (300, (210, 45, 45)),
    ],
)
def test_class_allocation(n, expected):
    train, val, test, _ = class_allocation(n)
    assert (train, val, test) == expected
    assert train + val + test == n


def _ground_truth(counts):
    records, n = [], 0
    for verdict, count in counts.items():
        for _ in range(count):
            n += 1
            records.append(gold_record(n, verdict))
    return records


def test_split_is_stratified_disjoint_and_deterministic():
    counts = {VERDICTS[0]: 40, VERDICTS[1]: 20, VERDICTS[2]: 20, VERDICTS[3]: 80, VERDICTS[4]: 40}
    records = _ground_truth(counts)
    splits, report = stratified_split(records, seed=42)
    again, _ = stratified_split(list(reversed(records)), seed=42)

    assert [r["package_name"] for r in splits["test"]] == [r["package_name"] for r in again["test"]]
    names = [r["package_name"] for part in splits.values() for r in part]
    assert len(names) == len(set(names)) == 200
    assert report["split_sizes"] == {"train": 140, "val": 30, "test": 30}
    for label, count in counts.items():
        assert report["class_counts"]["test"][label] == round(0.15 * count)
        assert report["class_counts"]["val"][label] == round(0.15 * count)
        assert report["class_counts"]["train"][label] == count - 2 * round(0.15 * count)
    assert not report["warnings"] or all("unstable" in w for w in report["warnings"])


def test_split_changes_with_the_seed():
    records = _ground_truth({VERDICTS[0]: 60, VERDICTS[3]: 60})
    a, _ = stratified_split(records, seed=42)
    b, _ = stratified_split(records, seed=7)
    assert [r["package_name"] for r in a["test"]] != [r["package_name"] for r in b["test"]]


def test_small_classes_are_handled_explicitly_and_reported():
    counts = {VERDICTS[0]: 30, VERDICTS[1]: 1, VERDICTS[2]: 2, VERDICTS[3]: 3, VERDICTS[4]: 0}
    splits, report = stratified_split(_ground_truth(counts), seed=42)
    per_class = report["per_class"]
    assert per_class[VERDICTS[1]]["train"] == 1 and per_class[VERDICTS[1]]["test"] == 0
    assert per_class[VERDICTS[2]]["train"] == 1 and per_class[VERDICTS[2]]["test"] == 1 and per_class[VERDICTS[2]]["val"] == 0
    assert (per_class[VERDICTS[3]]["train"], per_class[VERDICTS[3]]["val"], per_class[VERDICTS[3]]["test"]) == (1, 1, 1)
    assert per_class[VERDICTS[4]]["rule"] == "absent"
    text = " ".join(report["warnings"])
    assert VERDICTS[1] in text and "cannot be evaluated" in text
    assert VERDICTS[4] in text and "no record" in text
    realised = Counter(r["verdict"] for r in splits["test"])
    assert realised[VERDICTS[2]] == 1 and realised[VERDICTS[3]] == 1


def test_splits_cli_writes_files_and_report(tmp_path):
    records = _ground_truth({VERDICTS[0]: 20, VERDICTS[3]: 20})
    source = tmp_path / "ground_truth.jsonl"
    write_jsonl(source, records)
    assert splits_main(["--input", str(source), "--out-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "split_report.json").read_text(encoding="utf-8"))
    sizes = {name: len(read_jsonl(tmp_path / f"{name}.jsonl")) for name in ("train", "val", "test")}
    assert sizes == report["split_sizes"] == {"train": 28, "val": 6, "test": 6}
    assert report["input_sha256"] and report["seed"] == 42


def test_splits_cli_refuses_ground_truth_without_provenance(tmp_path):
    records = _ground_truth({VERDICTS[0]: 5})
    del records[0]["annotation"]
    source = tmp_path / "ground_truth.jsonl"
    write_jsonl(source, records)
    assert splits_main(["--input", str(source), "--out-dir", str(tmp_path)]) == 2
    assert not (tmp_path / "train.jsonl").exists()


# ----------------------------------------------------------------------------- sampling


def test_tercile_cutoffs_and_strata():
    stars = list(range(1, 10))  # 1..9: quantiles 3.67 and 6.33
    q1, q2 = star_tercile_cutoffs(stars)
    assert (round(q1, 2), round(q2, 2)) == (3.67, 6.33)
    low = dataset(1)[0] | {"stars": 3}
    mid = dataset(1)[0] | {"stars": 5}
    high = dataset(1)[0] | {"stars": 9, "vulnerabilities": [{"id": "x"}]}
    assert stratum_of(low, (q1, q2)) == "T1_low|no_vuln"
    assert stratum_of(mid, (q1, q2)) == "T2_mid|no_vuln"
    assert stratum_of(high, (q1, q2)) == "T3_high|vuln"


def test_proportional_allocation_sums_and_respects_sizes():
    allocation = allocate_proportionally({"a": 50, "b": 30, "c": 20}, 33)
    assert sum(allocation.values()) == 33
    assert allocation == {"a": 16, "b": 10, "c": 7}  # floors 16, 9, 6; remainders .9 (b), .6 (c), .5 (a)
    with pytest.raises(InsufficientDataError):
        allocate_proportionally({"a": 2}, 3)


def test_sample_is_deterministic_and_independent_of_input_order():
    records = dataset(120)
    first, meta = stratified_sample(records, 30, seed=42)
    second, _ = stratified_sample(list(reversed(records)), 30, seed=42)
    assert [r["package_name"] for r in first] == [r["package_name"] for r in second]
    assert len(first) == 30 and meta["realised_size"] == 30 and meta["shortfall"] == 0
    assert sum(meta["sample_by_stratum"].values()) == 30
    assert all("sampling_stratum" in r for r in first)
    other, _ = stratified_sample(records, 30, seed=7)
    assert [r["package_name"] for r in other] != [r["package_name"] for r in first]


def test_sample_keeps_stratum_proportions():
    records = dataset(240)
    sample, meta = stratified_sample(records, 60, seed=42)
    for stratum, population in meta["population_by_stratum"].items():
        expected = 60 * population / 240
        assert abs(meta["sample_by_stratum"].get(stratum, 0) - expected) < 1.0 + 1e-9


def test_sample_fails_loudly_when_too_small_unless_allowed():
    records = dataset(10)
    with pytest.raises(InsufficientDataError, match="Only 10 eligible records"):
        stratified_sample(records, 300)
    sample, meta = stratified_sample(records, 300, allow_smaller=True)
    assert len(sample) == 10 and meta["realised_size"] == 10 and meta["requested_size"] == 300
    assert meta["allow_smaller"] is True and meta["shortfall"] == 290


def test_sample_rejects_duplicates():
    records = dataset(5) + [dataset(1)[0]]
    with pytest.raises(DataError, match="Duplicate"):
        stratified_sample(records, 3)


def test_sample_cli(tmp_path):
    source = tmp_path / "dataset.jsonl"
    write_jsonl(source, dataset(40))
    out = tmp_path / "sample_300.jsonl"
    assert sample_main(["--input", str(source), "--output", str(out)]) == 2  # fewer than 300
    assert not out.exists()
    assert sample_main(["--input", str(source), "--output", str(out), "--allow-smaller"]) == 0
    meta = json.loads((tmp_path / "sample_300.meta.json").read_text(encoding="utf-8"))
    assert meta["realised_size"] == 40 and meta["requested_size"] == 300 and meta["seed"] == 42
    assert meta["input_sha256"]
    assert len(read_jsonl(out)) == 40


def test_numpy_quantile_matches_the_definition():
    assert np.quantile([1, 2, 3, 4], [1 / 3, 2 / 3]).tolist() == pytest.approx([2.0, 3.0])
