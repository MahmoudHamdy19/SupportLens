"""Command-line entry points of the data pipeline (thin wrappers used by ``scripts/*.py``)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import annotation as ann
from .adjudication import (
    AdjudicationError,
    adjudication_template,
    build_ground_truth,
    load_adjudication,
    validate_ground_truth_records,
)
from .agreement import build_agreement_report, find_disagreements
from .collector import CandidateListError, CollectionAborted, CollectorConfig, DatasetCollector
from .constants import ANNOTATORS, SEED, TAU_VALUES, TEMPORAL_VALUES
from .io_utils import (
    PROJECT_ROOT,
    DataError,
    atomic_write_many,
    configure_console,
    dumps_json,
    jsonl_text,
    read_jsonl,
    sha256_file,
    utc_now_iso,
    write_json,
)
from .sampling import stratified_sample
from .splits import add_input_provenance, stratified_split

DATA = PROJECT_ROOT / "data"


def _fail(message: str, code: int = 2) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def _load_sample(path: Path):
    records = read_jsonl(path)
    policies = {}
    for record in records:
        name = record.get("package_name")
        if not name or "policy_text" not in record:
            raise DataError(f"{path}: every sample record needs package_name and policy_text")
        if name in policies:
            raise DataError(f"{path}: duplicate package {name}")
        policies[name] = record["policy_text"]
    if not records:
        raise DataError(f"{path}: the sample is empty")
    return records, policies


# --------------------------------------------------------------------------- A. collection


def collect_main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="Collect the PyPI dataset (resumable).")
    parser.add_argument("--target", type=int, help="stop when the dataset holds this many records")
    parser.add_argument("--max-candidates", type=int, help="process at most this many new candidates in this run")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds to wait between candidates (default 1.0)")
    parser.add_argument("--output", type=Path, default=DATA / "processed" / "pypi_dataset.jsonl")
    parser.add_argument("--seed-file", type=Path, default=DATA / "seed" / "pypi_seed_packages.txt")
    parser.add_argument("--top-cache", type=Path, default=DATA / "cache" / "top-pypi-packages.min.json")
    parser.add_argument("--audit-dir", type=Path, default=DATA / "audit", help="exclusion and collection logs")
    parser.add_argument("--top-limit", type=int, help="use only the first N packages of the top-PyPI list")
    parser.add_argument("--refresh-top-list", action="store_true", help="download the top-PyPI list again")
    parser.add_argument("--seed-only", action="store_true", help="use the seed list only (no top-PyPI download)")
    parser.add_argument("--packages", nargs="+", help="process exactly these packages (debugging, smoke tests)")
    parser.add_argument(
        "--allow-partial-evidence",
        action="store_true",
        help="keep records whose OSV or GitHub Advisory source failed (the diagnostics stay in the record)",
    )
    parser.add_argument("--abort-after-transient", type=int, default=25)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = CollectorConfig(
        seed_file=args.seed_file,
        top_cache=args.top_cache,
        output=args.output,
        audit_dir=args.audit_dir,
        target=args.target,
        max_candidates=args.max_candidates,
        delay=args.delay,
        top_limit=args.top_limit,
        refresh_top_list=args.refresh_top_list,
        seed_only=args.seed_only,
        only_packages=args.packages,
        allow_partial_evidence=args.allow_partial_evidence,
        abort_after_transient=args.abort_after_transient,
    )
    try:
        summary = DatasetCollector(config).run()
    except CollectionAborted as exc:
        return _fail(str(exc), 3)
    except (CandidateListError, DataError) as exc:
        return _fail(str(exc))
    print(
        f"candidates={summary.total_candidates} skipped_done={summary.skipped_done} attempted={summary.attempted} "
        f"included={summary.included} excluded={summary.excluded} transient={summary.transient} "
        f"errors={summary.errors} dataset_size={summary.dataset_size}"
    )
    print(f"stopped: {summary.stopped_reason}")
    return 0


# --------------------------------------------------------------------------- B. sampling


def sample_main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="Draw the stratified random sample (thesis 3.3.1).")
    parser.add_argument("--input", type=Path, default=DATA / "processed" / "pypi_dataset.jsonl")
    parser.add_argument("--output", type=Path, default=DATA / "annotations" / "sample_300.jsonl")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--allow-smaller", action="store_true", help="accept fewer records than --size and record the real size")
    args = parser.parse_args(argv)
    try:
        records = read_jsonl(args.input)
        sampled, metadata = stratified_sample(records, args.size, seed=args.seed, allow_smaller=args.allow_smaller)
    except DataError as exc:
        return _fail(str(exc))
    metadata["input_file"] = str(args.input)
    metadata["input_sha256"] = sha256_file(args.input)
    metadata["created_at"] = utc_now_iso()
    meta_path = args.output.with_suffix(".meta.json")
    atomic_write_many({args.output: jsonl_text(sampled), meta_path: dumps_json(metadata) + "\n"})
    print(f"sample written: {len(sampled)} of {args.size} requested -> {args.output}")
    print(f"metadata: {meta_path}")
    return 0


# --------------------------------------------------------------------------- C. annotation


def prepare_annotation_main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="Write blind Label Studio task files for the two annotators.")
    parser.add_argument("--sample", type=Path, default=DATA / "annotations" / "sample_300.jsonl")
    parser.add_argument("--out-dir", type=Path, default=DATA / "annotations" / "label_studio")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    try:
        records, _ = _load_sample(args.sample)
        tasks = ann.build_tasks(sorted(records, key=lambda r: r["package_name"]))
        ann.assert_blind(tasks)
        files = {args.out_dir / "tasks_all.json": dumps_json(tasks, indent=1) + "\n"}
        for index, annotator in enumerate(ANNOTATORS, start=1):
            order = ann.shuffled_for_annotator(tasks, index, args.seed)
            ann.assert_blind(order)
            files[args.out_dir / f"tasks_{annotator}.json"] = dumps_json(order, indent=1) + "\n"
        atomic_write_many(files)
    except DataError as exc:
        return _fail(str(exc))
    print(f"{len(tasks)} tasks written to {args.out_dir} (tasks_all.json plus one blind, shuffled file per annotator)")
    return 0


def _load_annotators(args) -> tuple:
    records, policies = _load_sample(args.sample)
    if Path(args.annotator1).resolve() == Path(args.annotator2).resolve() or sha256_file(args.annotator1) == sha256_file(args.annotator2):
        raise DataError("The two annotator files are identical; the annotators must work independently.")
    first = ann.parse_annotator_file(args.annotator1, policies, who="annotator1")
    second = ann.parse_annotator_file(args.annotator2, policies, who="annotator2")
    ann.require_complete(first, policies, "annotator1")
    ann.require_complete(second, policies, "annotator2")
    return records, policies, first, second


def check_agreement_main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="Inter-annotator agreement (Cohen's kappa) before adjudication.")
    parser.add_argument("--sample", type=Path, default=DATA / "annotations" / "sample_300.jsonl")
    parser.add_argument("--annotator1", type=Path, required=True)
    parser.add_argument("--annotator2", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DATA / "annotations" / "agreement_report.json")
    parser.add_argument("--require-target", action="store_true", help="exit with status 1 when kappa is below 0.80")
    args = parser.parse_args(argv)
    try:
        _, _, first, second = _load_annotators(args)
        report = build_agreement_report(first, second, tau_values=TAU_VALUES, temporal_values=TEMPORAL_VALUES)
    except DataError as exc:
        return _fail(str(exc))
    report["inputs"] = {
        "sample": str(args.sample),
        "sample_sha256": sha256_file(args.sample),
        "annotator1_file": str(args.annotator1),
        "annotator1_sha256": sha256_file(args.annotator1),
        "annotator2_file": str(args.annotator2),
        "annotator2_sha256": sha256_file(args.annotator2),
    }
    report["generated_at"] = utc_now_iso()
    write_json(args.output, report)
    print(
        f"kappa={report['cohens_kappa']:.4f} (target {report['kappa_target']}) "
        f"agreement={report['percent_agreement']:.1f}% disagreements={report['n_disagreements']} -> {args.output}"
    )
    for warning in report["warnings"]:
        print(f"warning: {warning}")
    return 1 if (args.require_target and not report["meets_target"]) else 0


def adjudicate_main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="Build the ground truth from two annotators and the supervisor's decisions.")
    parser.add_argument("--sample", type=Path, default=DATA / "annotations" / "sample_300.jsonl")
    parser.add_argument("--annotator1", type=Path, required=True)
    parser.add_argument("--annotator2", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, help="supervisor decisions (JSON); required when annotators disagree")
    parser.add_argument("--output", type=Path, default=DATA / "ground_truth" / "ground_truth.jsonl")
    parser.add_argument("--template", type=Path, help="write a blank decision sheet for the supervisor and stop")
    args = parser.parse_args(argv)
    try:
        records, _, first, second = _load_annotators(args)
        if args.template:
            write_json(args.template, adjudication_template(first, second))
            print(f"{len(find_disagreements(first, second))} disagreement(s) listed in {args.template}; no ground truth written")
            return 0
        adjudicator, decisions = "", {}
        if args.adjudication:
            adjudicator, decisions = load_adjudication(args.adjudication)
        ground_truth = build_ground_truth(records, first, second, adjudicator, decisions)
        validate_ground_truth_records(ground_truth)
    except AdjudicationError as exc:
        return _fail(str(exc), 3)
    except DataError as exc:
        return _fail(str(exc))
    meta = {
        "n_records": len(ground_truth),
        "n_adjudicated": sum(1 for r in ground_truth if r["annotation"]["adjudicated_by"]),
        "adjudicator": adjudicator or None,
        "sample_sha256": sha256_file(args.sample),
        "annotator1_sha256": sha256_file(args.annotator1),
        "annotator2_sha256": sha256_file(args.annotator2),
        "adjudication_sha256": sha256_file(args.adjudication) if args.adjudication else None,
        "created_at": utc_now_iso(),
    }
    atomic_write_many(
        {
            args.output: jsonl_text(ground_truth),
            args.output.with_name("ground_truth_meta.json"): dumps_json(meta) + "\n",
        }
    )
    print(f"ground truth written: {len(ground_truth)} records ({meta['n_adjudicated']} adjudicated) -> {args.output}")
    return 0


def splits_main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="Stratified 70/15/15 split of the ground truth.")
    parser.add_argument("--input", type=Path, default=DATA / "ground_truth" / "ground_truth.jsonl")
    parser.add_argument("--out-dir", type=Path, default=DATA / "ground_truth")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    try:
        records = read_jsonl(args.input)
        validate_ground_truth_records(records)
        splits, report = stratified_split(records, seed=args.seed)
    except DataError as exc:
        return _fail(str(exc))
    report = add_input_provenance(report, args.input)
    files = {args.out_dir / f"{name}.jsonl": jsonl_text(items) for name, items in splits.items()}
    files[args.out_dir / "split_report.json"] = dumps_json(report) + "\n"
    atomic_write_many(files)
    sizes = report["split_sizes"]
    print(f"train={sizes['train']} val={sizes['val']} test={sizes['test']} -> {args.out_dir}")
    for warning in report["warnings"]:
        print(f"warning: {warning}")
    return 0
