"""scripts/package_dataset.py and the release metadata files (synthetic data only)."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from src.data_pipeline.constants import VERDICTS
from src.data_pipeline.io_utils import write_jsonl
from supportlens_synthetic import gold_record

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("package_dataset", ROOT / "scripts" / "package_dataset.py")
package_dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package_dataset)


@pytest.fixture
def dataset(tmp_path):
    gt, ann = tmp_path / "ground_truth", tmp_path / "annotations"
    records = [gold_record(n, VERDICTS[n % 5]) for n in range(1, 11)]
    write_jsonl(gt / "ground_truth.jsonl", records)
    write_jsonl(gt / "train.jsonl", records[:7])
    write_jsonl(gt / "val.jsonl", records[7:8])
    write_jsonl(gt / "test.jsonl", records[8:])
    (gt / "split_report.json").write_text(json.dumps({"seed": 42}), encoding="utf-8")
    ann.mkdir()
    (ann / "annotation_guidelines.md").write_text("# synthetic codebook\n", encoding="utf-8")
    (ann / "label_studio_config.xml").write_text("<View/>\n", encoding="utf-8")
    (ann / "agreement_report.json").write_text(json.dumps({"cohens_kappa": 0.85}), encoding="utf-8")
    meta = tmp_path / ".zenodo.json"
    meta.write_text(json.dumps({"upload_type": "dataset"}), encoding="utf-8")
    return {"gt": gt, "ann": ann, "meta": meta, "tmp": tmp_path}


def test_archive_contains_every_file_with_a_matching_manifest(dataset):
    out = dataset["tmp"] / "dist" / "bundle.zip"
    manifest = package_dataset.build_archive(out, dataset["gt"], dataset["ann"], dataset["meta"])
    with zipfile.ZipFile(out) as archive:
        names = {Path(n).name for n in archive.namelist()}
        folder = package_dataset.FOLDER
        assert all(n.startswith(folder + "/") for n in archive.namelist())
        assert names == set(package_dataset.GROUND_TRUTH_FILES) | set(package_dataset.ANNOTATION_FILES) | {
            "DATASET_CARD.md", "MANIFEST.json", ".zenodo.json"}
        stored = json.loads(archive.read(f"{folder}/MANIFEST.json"))
        for name, info in stored["files"].items():
            assert hashlib.sha256(archive.read(f"{folder}/{name}")).hexdigest() == info["sha256"]
        card = archive.read(f"{folder}/DATASET_CARD.md").decode("utf-8")
    assert stored == manifest and "to be assigned" in stored["doi"]
    assert "Records: 10 (train 7, validation 1, test 2)" in card and "0.8500" in card
    assert "DOI: to be assigned by Zenodo on deposit." in card
    first = out.read_bytes()
    package_dataset.build_archive(out, dataset["gt"], dataset["ann"], dataset["meta"])
    assert out.read_bytes() == first  # deterministic


def test_refuses_without_ground_truth(dataset, capsys):
    (dataset["gt"] / "ground_truth.jsonl").unlink()
    out = dataset["tmp"] / "bundle.zip"
    with pytest.raises(package_dataset.PackagingError, match="ground_truth.jsonl does not exist"):
        package_dataset.build_archive(out, dataset["gt"], dataset["ann"], dataset["meta"])
    code = package_dataset.main(["--ground-truth-dir", str(dataset["gt"]), "--annotations-dir", str(dataset["ann"]),
                                 "--output", str(out)])
    assert code == 2 and "never creates labels" in capsys.readouterr().err
    assert not out.exists()


def test_refuses_when_another_bundled_file_is_missing(dataset):
    (dataset["ann"] / "agreement_report.json").unlink()
    with pytest.raises(package_dataset.PackagingError, match="agreement_report.json"):
        package_dataset.build_archive(dataset["tmp"] / "b.zip", dataset["gt"], dataset["ann"], dataset["meta"])
    assert not (dataset["tmp"] / "b.zip").exists()


def test_release_metadata_files():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License") and "Copyright (c) 2026 SupportLens authors" in license_text
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for expected in ("cff-version: 1.2.0", "license: MIT", "Prince Sattam bin Abdulaziz University",
                     "Master of Applied Artificial Intelligence", "Torkey", '"Student 1"', '"Student 2"', '"Student 3"',
                     "Assessing Version-Support Commitments Against Vulnerability and Release Evidence"):
        assert expected in citation
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert zenodo["upload_type"] == "dataset" and "doi" not in zenodo
    assert "to be assigned" in zenodo["notes"]
    assert [c["name"] for c in zenodo["creators"]] == [f"Full Name, Student {i}" for i in (1, 2, 3)]
