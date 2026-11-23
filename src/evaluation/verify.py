"""Independent verification of every reported number, and rendering of README tables.

``verify`` recomputes the headline numbers from the raw material and fails when any value in
the result files or in the Markdown tables differs:

* extraction_metrics.json, verdict_metrics.json, statistical_tests.json and
  failure_analysis.json are recomputed from ``predictions.jsonl`` and the test ground truth.
  This covers the verdict macro-F1 and extraction bootstrap CIs (t F1, v F1, tuple EM/PM),
  both McNemar tests (verdict correctness and tuple exact match), the median-split and the
  active-versus-abandoned Mann-Whitney tests, and the RQ3 failure analysis;
* expert_evaluation.json is recomputed from the rating matrix stored in it;
* agreement_report.json is recomputed from the annotation provenance of ``ground_truth.jsonl``
  and split_report.json from the split files;
* the run manifest must describe a full strict run (no ``--limit``, no skipped system, no
  degraded prediction), and the stored data hashes must match the files on disk.

Markdown markers have the form ``<!-- RESULT:key -->value<!-- /RESULT -->`` where ``key`` is a
dotted path into the merged result documents, for example
``verdict_metrics.systems.qwen_finetuned_lora.macro_f1``. An optional format follows a bar,
``key|.2f``; the default is four decimals for floats. The top-level names are the file stems:
extraction_metrics, verdict_metrics, statistical_tests, failure_analysis, expert_evaluation,
agreement_report, split_report and run_manifest.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.data_pipeline.adjudication import validate_ground_truth_records
from src.data_pipeline.agreement import build_agreement_report
from src.data_pipeline.annotation import Annotation
from src.data_pipeline.constants import SEED, TAU_VALUES, TEMPORAL_VALUES
from src.data_pipeline.io_utils import DataError, atomic_write_many, read_json, read_jsonl, sha256_file
from src.data_pipeline.splits import stratified_split

from .expert import summarise_ratings
from .extraction_eval import Scorer
from .runner import compute_results

MARKER = re.compile(
    r"<!--\s*RESULT:(?P<key>[^\s|>]+)(?:\|(?P<fmt>[^\s>]+))?\s*-->(?P<value>.*?)<!--\s*/RESULT\s*-->",
    re.DOTALL,
)

RESAMPLES = 1000
TOLERANCE = 1e-9


@dataclass
class VerifyReport:
    problems: List[str] = field(default_factory=list)
    checked: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


# --------------------------------------------------------------------------- helpers


def compare(expected: Any, actual: Any, path: str = "") -> List[str]:
    """Differences between two JSON-like values; floats match within a tight tolerance."""
    here = path or "<root>"
    if isinstance(expected, dict) and isinstance(actual, dict):
        problems: List[str] = []
        for key in sorted(set(expected) | set(actual)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in expected:
                problems.append(f"{sub}: not in the recomputed result")
            elif key not in actual:
                problems.append(f"{sub}: missing from the file")
            else:
                problems.extend(compare(expected[key], actual[key], sub))
        return problems
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{here}: list length {len(actual)} in the file, {len(expected)} recomputed"]
        problems = []
        for index, (e, a) in enumerate(zip(expected, actual)):
            problems.extend(compare(e, a, f"{path}[{index}]"))
        return problems
    if isinstance(expected, bool) or isinstance(actual, bool):
        return [] if expected is actual else [f"{here}: file {actual!r}, recomputed {expected!r}"]
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isclose(expected, actual, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            return []
        return [f"{here}: file {actual!r}, recomputed {expected!r}"]
    return [] if expected == actual else [f"{here}: file {actual!r}, recomputed {expected!r}"]


def lookup(documents: Mapping[str, Any], key: str) -> Any:
    node: Any = documents
    for part in key.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise KeyError(key)
    return node


def format_value(value: Any, fmt: Optional[str] = None) -> str:
    if fmt:
        return format(value, fmt)
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_documents(eval_dir: Path, annotations_dir: Path, ground_truth_dir: Path) -> Dict[str, Any]:
    locations = {
        "extraction_metrics": Path(eval_dir) / "extraction_metrics.json",
        "verdict_metrics": Path(eval_dir) / "verdict_metrics.json",
        "statistical_tests": Path(eval_dir) / "statistical_tests.json",
        "failure_analysis": Path(eval_dir) / "failure_analysis.json",
        "expert_evaluation": Path(eval_dir) / "expert_evaluation.json",
        "run_manifest": Path(eval_dir) / "run_manifest.json",
        "agreement_report": Path(annotations_dir) / "agreement_report.json",
        "split_report": Path(ground_truth_dir) / "split_report.json",
    }
    return {name: read_json(path) for name, path in locations.items() if path.is_file()}


# --------------------------------------------------------------------------- recomputation


def recompute_evaluation(eval_dir: Path, test_file: Path, scorer: Scorer) -> Dict[str, Any]:
    records = read_jsonl(test_file)
    rows = read_jsonl(Path(eval_dir) / "predictions.jsonl")
    return compute_results(records, rows, scorer, n_resamples=RESAMPLES, seed=SEED)


def _annotations_from_ground_truth(records: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Annotation], Dict[str, Annotation]]:
    first, second = {}, {}
    for record in records:
        name = record["package_name"]
        a1, a2 = record["annotation"]["annotator1"], record["annotation"]["annotator2"]
        first[name] = Annotation(name, a1["commitment"], a1["verdict"])
        second[name] = Annotation(name, a2["commitment"], a2["verdict"])
    return first, second


def _check_manifest(manifest: Mapping[str, Any], paths: Mapping[str, Path], report: VerifyReport) -> None:
    run = manifest.get("run", {})
    if run.get("limit") is not None:
        report.problems.append("run_manifest: the run used --limit (a smoke test); its results cannot be reported")
    if run.get("strict") is not True:
        report.problems.append("run_manifest: the run was not strict")
    if run.get("skipped_systems"):
        report.problems.append(f"run_manifest: systems were skipped: {sorted(run['skipped_systems'])}")
    if run.get("degraded_predictions"):
        report.problems.append(f"run_manifest: {run['degraded_predictions']} degraded prediction(s)")
    for name, info in (manifest.get("data_files") or {}).items():
        path = paths.get(name)
        if path is None or not path.is_file():
            report.skipped.append(f"hash of {name} (file not found)")
            continue
        if sha256_file(path) != info.get("sha256"):
            report.problems.append(f"run_manifest: {name} was changed after the run (hash mismatch)")
        else:
            report.checked.append(f"hash of {name}")


# --------------------------------------------------------------------------- verification


def verify(
    *,
    eval_dir: Path,
    ground_truth_dir: Path,
    annotations_dir: Path,
    scorer: Scorer,
    markdown_files: Sequence[Path] = (),
    test_file: Optional[Path] = None,
) -> VerifyReport:
    report = VerifyReport()
    eval_dir, ground_truth_dir, annotations_dir = Path(eval_dir), Path(ground_truth_dir), Path(annotations_dir)
    test_file = Path(test_file) if test_file else ground_truth_dir / "test.jsonl"
    documents = load_documents(eval_dir, annotations_dir, ground_truth_dir)
    authoritative = dict(documents)

    # 1. evaluation documents
    predictions = eval_dir / "predictions.jsonl"
    if predictions.is_file() and test_file.is_file():
        if "run_manifest" in documents:
            paths = {
                "predictions": predictions,
                "test": test_file,
                **{n: ground_truth_dir / f"{n}.jsonl" for n in ("train", "val", "ground_truth")},
            }
            _check_manifest(documents["run_manifest"], paths, report)
        else:
            report.problems.append("run_manifest.json is missing")
        try:
            recomputed = recompute_evaluation(eval_dir, test_file, scorer)
        except DataError as exc:
            report.problems.append(f"recomputation failed: {exc}")
            recomputed = {}
        for name, expected in recomputed.items():
            if name not in documents:
                report.problems.append(f"{name}.json is missing")
                continue
            differences = compare(expected, documents[name], name)
            report.problems.extend(differences)
            if not differences:
                report.checked.append(f"{name}.json recomputed from predictions.jsonl")
            authoritative[name] = expected
    else:
        report.problems.append("predictions.jsonl or the test ground truth is missing; evaluation numbers cannot be verified")

    # 2. expert evaluation
    expert = documents.get("expert_evaluation")
    if expert is not None:
        try:
            expected = summarise_ratings(expert["ratings"], expert["items"])
            differences = compare(expected, {k: expert.get(k) for k in expected}, "expert_evaluation")
            report.problems.extend(differences)
            if not differences:
                report.checked.append("expert_evaluation.json recomputed from its rating matrix")
            authoritative["expert_evaluation"] = {**expert, **expected}
        except (KeyError, ValueError) as exc:
            report.problems.append(f"expert_evaluation.json cannot be recomputed: {exc}")
    else:
        report.skipped.append("expert_evaluation.json (not present)")

    # 3. agreement and splits
    ground_truth_file = ground_truth_dir / "ground_truth.jsonl"
    if ground_truth_file.is_file():
        records = read_jsonl(ground_truth_file)
        validate_ground_truth_records(records)
        if "agreement_report" in documents:
            first, second = _annotations_from_ground_truth(records)
            expected = build_agreement_report(first, second, tau_values=TAU_VALUES, temporal_values=TEMPORAL_VALUES)
            actual = {k: v for k, v in documents["agreement_report"].items() if k in expected}
            differences = compare(expected, actual, "agreement_report")
            report.problems.extend(differences)
            if not differences:
                report.checked.append("agreement_report.json recomputed from ground_truth.jsonl")
            authoritative["agreement_report"] = {**documents["agreement_report"], **expected}
        if "split_report" in documents:
            parts = {n: ground_truth_dir / f"{n}.jsonl" for n in ("train", "val", "test")}
            if all(p.is_file() for p in parts.values()):
                loaded = {n: read_jsonl(p) for n, p in parts.items()}
                names = [r["package_name"] for rows in loaded.values() for r in rows]
                if len(names) != len(set(names)):
                    report.problems.append("split files: a package occurs in more than one split")
                if set(names) != {r["package_name"] for r in records}:
                    report.problems.append("split files do not cover the ground truth exactly")
                expected_splits, expected_split = stratified_split(records, seed=SEED)
                keys = ("seed", "total_records", "split_sizes", "class_counts", "per_class")
                differences = compare(
                    {k: expected_split[k] for k in keys},
                    {k: documents["split_report"].get(k) for k in keys},
                    "split_report",
                )
                report.problems.extend(differences)
                if not differences:
                    report.checked.append("split_report.json recomputed from ground_truth.jsonl")
                for name, rows in loaded.items():
                    if [r["package_name"] for r in rows] != [r["package_name"] for r in expected_splits[name]]:
                        report.problems.append(f"{name}.jsonl does not equal the seed-42 stratified split")
            else:
                report.skipped.append("split files (not all present)")
    else:
        report.skipped.append("agreement and split verification (ground_truth.jsonl not present)")

    # 4. markdown markers
    for path in markdown_files:
        path = Path(path)
        if not path.is_file():
            report.problems.append(f"{path}: markdown file not found")
            continue
        text = path.read_text(encoding="utf-8")
        for match in MARKER.finditer(text):
            key, fmt, shown = match.group("key"), match.group("fmt"), match.group("value").strip()
            try:
                expected_text = format_value(lookup(authoritative, key), fmt)
            except KeyError:
                report.problems.append(f"{path.name}: marker {key!r} refers to a value that does not exist")
                continue
            except (ValueError, TypeError) as exc:
                report.problems.append(f"{path.name}: marker {key!r} has an unusable format ({exc})")
                continue
            if shown != expected_text:
                report.problems.append(f"{path.name}: marker {key} shows {shown!r}, the verified value is {expected_text!r}")
            else:
                report.checked.append(f"{path.name}: {key}")
    return report


# --------------------------------------------------------------------------- rendering


def render_markers(text: str, documents: Mapping[str, Any]) -> Tuple[str, List[str]]:
    """Fill every marker from ``documents``; return the new text and the list of problems."""
    problems: List[str] = []

    def replace(match: "re.Match[str]") -> str:
        key, fmt = match.group("key"), match.group("fmt")
        try:
            value = format_value(lookup(documents, key), fmt)
        except KeyError:
            problems.append(f"marker {key!r}: no such value in the result files")
            return match.group(0)
        except (ValueError, TypeError) as exc:
            problems.append(f"marker {key!r}: unusable format ({exc})")
            return match.group(0)
        fmt_part = f"|{fmt}" if fmt else ""
        return f"<!-- RESULT:{key}{fmt_part} -->{value}<!-- /RESULT -->"

    return MARKER.sub(replace, text), problems


def render_files(markdown_files: Sequence[Path], documents: Mapping[str, Any]) -> Dict[str, int]:
    """Update the markers of each file in place; nothing is written when any marker cannot be filled."""
    updated: Dict[Path, str] = {}
    counts: Dict[str, int] = {}
    all_problems: List[str] = []
    for path in map(Path, markdown_files):
        text = path.read_text(encoding="utf-8")
        new_text, problems = render_markers(text, documents)
        all_problems.extend(f"{path.name}: {p}" for p in problems)
        counts[str(path)] = len(MARKER.findall(text))
        if new_text != text:
            updated[path] = new_text
    if all_problems:
        raise DataError("Rendering refused: " + "; ".join(all_problems))
    atomic_write_many(updated)
    return counts
