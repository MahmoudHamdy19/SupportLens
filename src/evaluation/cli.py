"""Command-line entry points of the evaluation (thin wrappers used by ``scripts/*.py``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from src.data_pipeline.adjudication import validate_ground_truth_records
from src.data_pipeline.constants import SEED
from src.data_pipeline.io_utils import (
    PROJECT_ROOT,
    DataError,
    atomic_write_many,
    configure_console,
    dumps_json,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now_iso,
    write_json,
)

from . import expert as expert_module
from .extraction_eval import MetricsUnavailableError, resolve_scorer
from .runner import SYSTEM_IDS, EvaluationAborted, default_engine, default_explainer, default_factories, run_evaluation
from .verify import load_documents, render_files, verify

DATA = PROJECT_ROOT / "data"


def _fail(message: str, code: int = 2) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def _flatten(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None
    return [token for value in values for token in value.split(",") if token]


# --------------------------------------------------------------------------- D. evaluation


def run_evaluation_main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="Evaluate the four systems on the held-out test set (thesis 3.3.2-3.3.4).")
    parser.add_argument("--ground-truth-dir", type=Path, default=DATA / "ground_truth")
    parser.add_argument("--test-file", type=Path, help="default: <ground-truth-dir>/test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DATA / "evaluation")
    parser.add_argument("--systems", nargs="+", help=f"subset of {', '.join(SYSTEM_IDS)}")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="abort when a system is unavailable or degraded (default: on); --no-strict skips unavailable systems and names them in the manifest",
    )
    parser.add_argument("--limit", type=int, help="first N test records, for smoke tests only; writes to a separate folder")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    gt_dir = args.ground_truth_dir
    test_file = args.test_file or gt_dir / "test.jsonl"
    if not test_file.is_file():
        return _fail(
            f"ground-truth test file not found: {test_file}. It is produced from human annotation by "
            "scripts/adjudicate.py and scripts/make_splits.py; the evaluation never creates labels."
        )
    try:
        records = read_jsonl(test_file)
        validate_ground_truth_records(records)
        for other in ("train", "val"):
            path = gt_dir / f"{other}.jsonl"
            if path.is_file():
                overlap = {r["package_name"] for r in records} & {r["package_name"] for r in read_jsonl(path)}
                if overlap:
                    return _fail(f"data leakage: {len(overlap)} test package(s) also occur in {other}.jsonl")
    except DataError as exc:
        return _fail(str(exc))

    out_dir = args.output_dir
    if args.limit is not None:
        records = records[: args.limit]
        out_dir = out_dir / f"smoke_limit_{args.limit}"
        print(f"smoke test: {len(records)} record(s); results go to {out_dir} and are not reportable")
    systems = _flatten(args.systems)

    try:
        scorer = resolve_scorer()
        factories = default_factories()
        engine = default_engine()
        explainer = default_explainer()
        result = run_evaluation(
            records,
            factories,
            engine,
            scorer,
            out_dir,
            strict=args.strict,
            systems=systems,
            limit=args.limit,
            n_resamples=args.bootstrap_resamples,
            data_files={"test": test_file, **{n: gt_dir / f"{n}.jsonl" for n in ("train", "val", "ground_truth")}},
            models_dir=PROJECT_ROOT / "models",
            options={"argv": list(argv) if argv is not None else sys.argv[1:]},
            explainer=explainer,
            progress=lambda message: print(message, flush=True),
        )
    except (EvaluationAborted, MetricsUnavailableError, DataError, ImportError) as exc:
        return _fail(f"{exc}\nNo result file was written.", 3)

    for system, block in result["docs"]["verdict_metrics"]["systems"].items():
        ci = result["docs"]["statistical_tests"]["bootstrap_macro_f1"]["systems"][system]
        print(f"{system}: macro-F1 {block['macro_f1']:.4f} (95% CI {ci['ci_low']:.4f}-{ci['ci_high']:.4f}), accuracy {block['accuracy']:.4f}")
    for system, reason in result["skipped"].items():
        print(f"skipped {system}: {reason}")
    print(f"results written to {out_dir}")
    return 0


# --------------------------------------------------------------------------- E. expert rubric


def expert_main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="Expert rubric: prepare blind rating sheets, then ingest the completed ones.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="sample explanations and write blind rating sheets")
    mode.add_argument("--ingest", action="store_true", help="read the completed rating sheets of the raters")
    parser.add_argument("--predictions", type=Path, default=DATA / "evaluation" / "predictions.jsonl")
    parser.add_argument("--ground-truth", type=Path, default=DATA / "ground_truth" / "test.jsonl", help="source of the policy snippets")
    parser.add_argument("--system", default="qwen_finetuned_lora")
    parser.add_argument("--per-category", type=int, default=6)
    parser.add_argument("--raters", type=int, default=2, help="number of blind sheets to write (at least 2)")
    parser.add_argument("--out-dir", type=Path, default=DATA / "evaluation" / "expert")
    parser.add_argument("--ratings", nargs="+", type=Path, help="completed rating sheets (CSV), one per rater")
    parser.add_argument("--output", type=Path, default=DATA / "evaluation" / "expert_evaluation.json")
    args = parser.parse_args(argv)
    manifest_path = args.out_dir / "expert_sample.json"

    try:
        if args.prepare:
            if args.raters < 2:
                return _fail("at least two raters are required")
            rows = read_jsonl(args.predictions)
            policies = {r["package_name"]: r["policy_text"] for r in read_jsonl(args.ground_truth)}
            items, availability = expert_module.select_items(
                rows, policies, system=args.system, per_category=args.per_category, seed=SEED
            )
            sheets = expert_module.write_rating_sheets(items, args.out_dir, args.raters, SEED)
            manifest = {
                "created_at": utc_now_iso(),
                "seed": SEED,
                "system": args.system,
                "predictions_sha256": sha256_file(args.predictions),
                "availability": availability,
                "items": [
                    {k: item[k] for k in ("item_id", "package_name", "system", "predicted_verdict")} for item in sorted(items, key=lambda i: i["item_id"])
                ],
            }
            atomic_write_many({manifest_path: dumps_json(manifest) + "\n"})
            print(f"{len(items)} explanations sampled for {len(sheets)} raters -> {args.out_dir}")
            for label, info in availability.items():
                note = f"  ({info['shortfall']} fewer than requested: the category has only {info['available']})" if info["shortfall"] else ""
                print(f"  {label}: {info['sampled']} of {info['requested']}{note}")
            return 0

        if not args.ratings:
            return _fail("--ingest needs --ratings with the completed sheets")
        manifest = read_json(manifest_path)
        summary = expert_module.ingest_ratings(args.ratings, manifest, min_raters=2)
        summary["sample_manifest_sha256"] = sha256_file(manifest_path)
        summary["ingested_at"] = utc_now_iso()
        write_json(args.output, summary)
    except DataError as exc:
        return _fail(str(exc))
    for dim, block in summary["dimensions"].items():
        alpha = block["krippendorff_alpha_interval"]
        print(f"{dim}: mean {block['mean']:.3f}, std {block['std']:.3f}, alpha {alpha if alpha is None else round(alpha, 3)}")
    print(f"written to {args.output}")
    return 0


# --------------------------------------------------------------------------- F. verification


def _default_markdown() -> List[Path]:
    files = [PROJECT_ROOT / "README.md"]
    files.extend(sorted((PROJECT_ROOT / "docs").glob("*.md")))
    return [p for p in files if p.is_file()]


def verify_main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="Recompute every reported number and fail on any difference.")
    parser.add_argument("--eval-dir", type=Path, default=DATA / "evaluation")
    parser.add_argument("--ground-truth-dir", type=Path, default=DATA / "ground_truth")
    parser.add_argument("--annotations-dir", type=Path, default=DATA / "annotations")
    parser.add_argument("--markdown", nargs="*", type=Path, help="files with RESULT markers (default: README.md and docs/*.md)")
    args = parser.parse_args(argv)
    try:
        scorer = resolve_scorer()
        report = verify(
            eval_dir=args.eval_dir,
            ground_truth_dir=args.ground_truth_dir,
            annotations_dir=args.annotations_dir,
            scorer=scorer,
            markdown_files=args.markdown if args.markdown is not None else _default_markdown(),
        )
    except (MetricsUnavailableError, DataError) as exc:
        return _fail(str(exc))
    for item in report.checked:
        print(f"ok      {item}")
    for item in report.skipped:
        print(f"skipped {item}")
    for item in report.problems:
        print(f"FAIL    {item}")
    print("VERIFICATION PASSED" if report.ok else f"VERIFICATION FAILED ({len(report.problems)} problem(s))")
    return 0 if report.ok else 1


def render_main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(description="Fill the RESULT markers of Markdown files from the result JSON files.")
    parser.add_argument("--eval-dir", type=Path, default=DATA / "evaluation")
    parser.add_argument("--ground-truth-dir", type=Path, default=DATA / "ground_truth")
    parser.add_argument("--annotations-dir", type=Path, default=DATA / "annotations")
    parser.add_argument("--markdown", nargs="*", type=Path, help="default: README.md and docs/*.md")
    args = parser.parse_args(argv)
    documents = load_documents(args.eval_dir, args.annotations_dir, args.ground_truth_dir)
    if not documents:
        return _fail("no result file found; run the evaluation first")
    files = args.markdown if args.markdown is not None else _default_markdown()
    try:
        counts = render_files(files, documents)
    except DataError as exc:
        return _fail(str(exc))
    for path, count in counts.items():
        print(f"{path}: {count} marker(s)")
    print("Run scripts/verify_results.py to confirm that the rendered values match a recomputation.")
    return 0
