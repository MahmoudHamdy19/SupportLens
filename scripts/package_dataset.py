"""Bundle the annotated reference dataset into a zip for the Zenodo deposit (thesis Section 3.4.1).

The archive contains, under one top-level folder:

    ground_truth.jsonl, train.jsonl, val.jsonl, test.jsonl, split_report.json   data/ground_truth/
    annotation_guidelines.md, label_studio_config.xml (the codebook)            data/annotations/
    agreement_report.json                                                       data/annotations/
    DATASET_CARD.md   generated from the files above (counts, agreement, licence, provenance)
    MANIFEST.json     size and SHA-256 of every bundled file
    .zenodo.json      the deposit metadata from the project root, when present

The script never creates or edits labels. It refuses to run when ground_truth.jsonl does not
exist, and equally when any other bundled file is missing, because a partial deposit cannot be
cited. Entries carry a fixed timestamp, so the same inputs give a byte-identical archive.

Run from the project root:  python scripts/package_dataset.py [--output dist/supportlens-dataset-v1.0.0.zip]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"
FOLDER = f"supportlens-dataset-v{VERSION}"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)

GROUND_TRUTH_FILES = ("ground_truth.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "split_report.json")
ANNOTATION_FILES = ("annotation_guidelines.md", "label_studio_config.xml", "agreement_report.json")


class PackagingError(RuntimeError):
    """The dataset cannot be packaged; nothing has been written."""


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_inputs(ground_truth_dir: Path, annotations_dir: Path) -> Dict[str, Path]:
    """Archive name -> source path; raises PackagingError when a required file is missing."""
    ground_truth = Path(ground_truth_dir) / "ground_truth.jsonl"
    if not ground_truth.is_file():
        raise PackagingError(
            f"{ground_truth} does not exist. Build the ground truth first (scripts/adjudicate.py, "
            "scripts/make_splits.py); this script never creates labels."
        )
    inputs = {name: Path(ground_truth_dir) / name for name in GROUND_TRUTH_FILES}
    inputs.update({name: Path(annotations_dir) / name for name in ANNOTATION_FILES})
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise PackagingError("Required dataset files are missing: " + ", ".join(missing))
    return inputs


def _table(counter: Mapping[str, int]) -> str:
    lines = ["| Value | Records |", "|---|---|"]
    lines += [f"| {key} | {counter[key]} |" for key in sorted(counter, key=str)]
    return "\n".join(lines)


def dataset_card(inputs: Mapping[str, Path]) -> str:
    """Markdown dataset card; every number is read from the bundled files."""
    records = _read_jsonl(inputs["ground_truth.jsonl"])
    splits = {name: len(_read_jsonl(inputs[f"{name}.jsonl"])) for name in ("train", "val", "test")}
    agreement = json.loads(inputs["agreement_report.json"].read_text(encoding="utf-8"))
    verdicts = Counter(str(r.get("verdict")) for r in records)
    taus = Counter(str((r.get("commitment") or {}).get("support_status")) for r in records)
    adjudicated = sum(1 for r in records if (r.get("annotation") or {}).get("adjudicated_by"))
    kappa = agreement.get("cohens_kappa")
    kappa_text = f"{kappa:.4f}" if isinstance(kappa, (int, float)) else "not reported"
    return f"""# SupportLens annotated reference dataset (version {VERSION})

Manually annotated reference dataset of the master research project *SupportLens: An Explainable
AI Framework for Assessing Version-Support Commitments Against Vulnerability and Release
Evidence* (thesis Section 3.3.1), Master of Applied Artificial Intelligence, Prince Sattam bin
Abdulaziz University. Supervisor: Hanaa Torkey.

DOI: to be assigned by Zenodo on deposit.

## Contents

| File | Meaning |
|---|---|
| `ground_truth.jsonl` | one record per package: policy text, commitment tuple c = <t, tau, v, T>, verdict, releases, vulnerabilities and both annotators' labels |
| `train.jsonl`, `val.jsonl`, `test.jsonl` | stratified 70/15/15 split on the verdict, seed 42 |
| `split_report.json` | realised class counts per split |
| `annotation_guidelines.md`, `label_studio_config.xml` | the annotation codebook and labelling interface |
| `agreement_report.json` | inter-annotator agreement (Cohen's kappa) measured before adjudication |
| `MANIFEST.json` | size and SHA-256 of every file |

## Summary

* Records: {len(records)} (train {splits['train']}, validation {splits['val']}, test {splits['test']})
* Records resolved by the supervisor's adjudication: {adjudicated}
* Cohen's kappa on the verdict: {kappa_text}

### Verdicts

{_table(verdicts)}

### Commitment type tau

{_table(taus)}

## Provenance

Two annotators labelled every record independently, without model output or each other's
labels; the supervisor resolved every disagreement. No label was produced by a heuristic, a
random number or a model. Releases come from the PyPI JSON API and vulnerabilities from OSV, the
GitHub Advisory Database and NVD, captured at the time given in each record's `collected_at`.

## Licence

The annotations and the dataset structure are released under CC BY 4.0. The security policy
texts are reproduced from public repositories and remain under their own licences. The code that
produced the dataset is released under the MIT licence.
"""


def build_archive(
    output: Path,
    ground_truth_dir: Path = ROOT / "data" / "ground_truth",
    annotations_dir: Path = ROOT / "data" / "annotations",
    zenodo_metadata: Optional[Path] = ROOT / ".zenodo.json",
) -> Dict[str, Any]:
    """Write the zip and return its manifest; raises PackagingError before writing anything."""
    inputs = collect_inputs(ground_truth_dir, annotations_dir)
    payload: Dict[str, bytes] = {name: path.read_bytes() for name, path in inputs.items()}
    payload["DATASET_CARD.md"] = dataset_card(inputs).encode("utf-8")
    if zenodo_metadata is not None and Path(zenodo_metadata).is_file():
        payload[".zenodo.json"] = Path(zenodo_metadata).read_bytes()
    manifest = {
        "dataset": FOLDER,
        "version": VERSION,
        "doi": "to be assigned by Zenodo on deposit",
        "files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(payload.items())
        },
    }
    payload["MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(payload):
            info = zipfile.ZipInfo(f"{FOLDER}/{name}", date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload[name])
    temporary.replace(output)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bundle the annotated dataset into a zip for Zenodo.")
    parser.add_argument("--ground-truth-dir", type=Path, default=ROOT / "data" / "ground_truth")
    parser.add_argument("--annotations-dir", type=Path, default=ROOT / "data" / "annotations")
    parser.add_argument("--zenodo-metadata", type=Path, default=ROOT / ".zenodo.json")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / f"{FOLDER}.zip")
    args = parser.parse_args(argv)
    try:
        manifest = build_archive(args.output, args.ground_truth_dir, args.annotations_dir, args.zenodo_metadata)
    except PackagingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output} ({len(manifest['files']) + 1} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
