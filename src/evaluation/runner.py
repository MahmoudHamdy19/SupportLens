"""End-to-end evaluation on the held-out test set (thesis Sections 3.3.2 to 3.3.4).

For each of the four systems (regex baseline, fine-tuned RoBERTa, zero-shot Qwen, fine-tuned
Qwen with LoRA) the runner

1. extracts the commitment tuple from the policy text of every test record;
2. scores the extraction against the annotated commitment (Q1);
3. feeds the PREDICTED commitment, never the gold one, together with the record's releases
   and vulnerabilities to the rule engine and compares the verdict with the annotated verdict
   (Q2);
4. computes the statistics of Section 3.3.4: bootstrap 95% confidence intervals (1,000
   resamples, seed 42, one set of resample indices shared by every metric and system) for the
   verdict macro-F1 and for the extraction scores t F1, v F1, tuple EM and tuple PM; McNemar's
   tests of the fine-tuned Qwen against each baseline on per-record verdict correctness and on
   per-record tuple exact match; Mann-Whitney U tests over project characteristics (median
   splits) and between active and abandoned projects;
5. writes the RQ3 failure analysis (failure_analysis.json, see ``failure_analysis.py``).

The record field ``mapping_uncertain`` (written by the collector; False when absent) is handed
to the rule engine, so that an uncertain package-to-repository mapping triggers rule R4.

Rules enforced here:

* Nothing is written unless the whole run succeeds; all result files are written together.
* The learned extractors are always built with ``strict=True`` and never fall back. In strict
  mode (the default) a system that is unavailable, produces a degraded result or reports
  provenance that contradicts its role aborts the run; with ``--no-strict`` an unavailable
  system is skipped and named in the manifest (smoke tests on machines without a GPU).
* The reference time handed to the rule engine is the record's evidence collection time, so
  that a verdict does not depend on the day the evaluation is run.
"""

from __future__ import annotations

import gc
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.data_pipeline.constants import COMMITMENT_KEYS, SEED, VERDICTS
from src.data_pipeline.io_utils import DataError, atomic_write_many, dumps_json, jsonl_text

from .extraction_eval import METRICS_MODULE, SCORER_NAME, Scorer, aggregate_extraction, score_columns
from .failure_analysis import ABANDONED, ACTIVE, ACTIVITY_DEFINITION, failure_analysis, project_activity
from .repro import build_manifest
from .stats import (
    bootstrap_indices,
    bootstrap_macro_f1,
    bootstrap_mean_ci,
    mann_whitney_groups,
    mann_whitney_median_split,
    mcnemar_test,
)
from .verdict_metrics import classification_report

# Extraction scores with a bootstrap CI (thesis 3.3.4): reported name -> per-record score key.
EXTRACTION_BOOTSTRAP_METRICS = {
    "t_f1": "t_f1",
    "v_f1": "v_f1",
    "tuple_exact_match": "tuple_exact_match",
    "tuple_partial_match": "tuple_partial_match",
}

SYSTEM_IDS = ("regex_baseline", "roberta_finetuned", "qwen_zero_shot", "qwen_finetuned_lora")
PROPOSED_SYSTEM = "qwen_finetuned_lora"
EXPECTED_LORA = {"qwen_zero_shot": False, "qwen_finetuned_lora": True}
PROVENANCE_KEYS = ("extractor_engine", "actual_model_used", "lora_applied", "execution_device", "degraded")

Factory = Callable[[], Any]


class EvaluationAborted(RuntimeError):
    """The evaluation cannot continue; no result file has been written."""


# --------------------------------------------------------------------------- systems


def default_factories() -> Dict[str, Factory]:
    """Builders for the four systems; each builder loads its model only when called.

    The learned extractors are always built with ``strict=True``: a missing model, adapter or
    CUDA device raises ``ModelUnavailableError`` instead of degrading to another extractor.
    """

    def regex() -> Any:
        from src.layer1_ai_extraction import RuleBasedRegexBaseline

        return RuleBasedRegexBaseline()

    def roberta() -> Any:
        from src.layer1_ai_extraction import RoBERTaCommitmentExtractor

        return RoBERTaCommitmentExtractor(strict=True)

    def qwen_zero_shot() -> Any:
        from src.layer1_ai_extraction import QwenCommitmentExtractor

        return QwenCommitmentExtractor(use_adapter=False, strict=True)

    def qwen_finetuned() -> Any:
        from src.layer1_ai_extraction import QwenCommitmentExtractor

        return QwenCommitmentExtractor(use_adapter=True, strict=True)

    return {
        "regex_baseline": regex,
        "roberta_finetuned": roberta,
        "qwen_zero_shot": qwen_zero_shot,
        "qwen_finetuned_lora": qwen_finetuned,
    }


def default_engine() -> Any:
    from src.layer3_rule_assessment import AssessmentRuleEngine

    return AssessmentRuleEngine()


def default_explainer() -> Any:
    """Layer 4 template-based explanation generator (thesis Section 3.2.5)."""
    from src.layer4_explanation_output.explanation_generation import ExplanationGenerator

    return ExplanationGenerator()


def _is_model_unavailable(exc: BaseException) -> bool:
    return any(cls.__name__ == "ModelUnavailableError" for cls in type(exc).__mro__)


def _release(extractor: Any) -> None:
    del extractor
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _as_dict(output: Any) -> Dict[str, Any]:
    if hasattr(output, "to_dict"):
        return dict(output.to_dict())
    if isinstance(output, Mapping):
        return dict(output)
    raise EvaluationAborted(f"An extractor returned {type(output).__name__}; expected an object with to_dict()")


def parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- running


def _source_references(record: Mapping[str, Any]) -> List[str]:
    return [
        f"PyPI JSON API: {len(record.get('releases') or [])} releases",
        f"OSV / GitHub Advisory / NVD: {len(record.get('vulnerabilities') or [])} de-duplicated vulnerability records",
        f"Security policy: {record.get('policy_url') or record.get('policy_path') or 'SECURITY.md'}",
    ]


def mapping_uncertain(record: Mapping[str, Any]) -> bool:
    """The collector's ``mapping_uncertain`` flag of a ground-truth record (False when absent)."""
    return record.get("mapping_uncertain") is True


def _reason_codes(result: Mapping[str, Any]) -> List[str]:
    """Distinct reason codes of the Layer 3 evidence log, in order of first occurrence."""
    codes: List[str] = []
    for entry in result.get("evidence_log") or []:
        code = entry.get("reason") if isinstance(entry, Mapping) else getattr(entry, "reason", None)
        if code and code not in codes:
            codes.append(str(code))
    return codes


def _predict_row(
    system: str, record: Mapping[str, Any], extraction: Dict[str, Any], engine: Any, strict: bool, explainer: Any = None
) -> Dict[str, Any]:
    missing = [k for k in COMMITMENT_KEYS if k not in extraction]
    if missing:
        raise EvaluationAborted(f"{system}: extractor output lacks {missing} for {record['package_name']}")
    provenance = {key: extraction.get(key) for key in PROVENANCE_KEYS}
    if strict:
        absent = [k for k, v in provenance.items() if k not in extraction]
        if absent:
            raise EvaluationAborted(f"{system}: extractor output lacks provenance keys {absent}; results could be mislabelled")
        if provenance["degraded"]:
            raise EvaluationAborted(
                f"{system}: a fallback produced the result for {record['package_name']} "
                f"(engine {provenance['extractor_engine']!r}); strict evaluation does not accept degraded output"
            )
    expected = EXPECTED_LORA.get(system)
    if expected is not None and provenance["lora_applied"] is not None and bool(provenance["lora_applied"]) != expected:
        raise EvaluationAborted(
            f"{system}: lora_applied={provenance['lora_applied']} contradicts the system definition (expected {expected})"
        )

    reference = parse_time(record.get("collected_at"))
    assessment = engine.evaluate(
        commitment=extraction,
        vulnerabilities=record.get("vulnerabilities"),
        releases=record.get("releases"),
        reference_time=reference,
        package_name=record["package_name"],
        source_diagnostics=record.get("source_diagnostics"),
        mapping_uncertain=mapping_uncertain(record),
    )
    result = assessment.to_dict() if hasattr(assessment, "to_dict") else dict(assessment)
    verdict = result.get("verdict")
    if verdict not in VERDICTS:
        raise EvaluationAborted(f"The rule engine returned {verdict!r}, which is not one of the five categories")
    explanation = result.get("explanation")
    if not explanation and explainer is not None:
        explanation = explainer.generate_explanation(
            assessment,
            commitment=extraction,
            policy_citation=record.get("policy_url") or record.get("policy_path"),
            source_references=_source_references(record),
        )
    return {
        "package_name": record["package_name"],
        "system": system,
        "commitment": {key: extraction[key] for key in COMMITMENT_KEYS},
        "is_measurable": extraction.get("is_measurable"),
        "provenance": provenance,
        "assessment_engine": type(engine).__name__,
        "reference_time": reference.isoformat() if reference else None,
        "verdict": verdict,
        "rules_triggered": result.get("rules_triggered"),
        "reason_codes": _reason_codes(result),
        "mapping_uncertain": mapping_uncertain(record),
        "explanation": explanation,
        "evidence_trail": result.get("evidence_trail"),
    }


def preflight(factories: Mapping[str, Factory], systems: Sequence[str]) -> Dict[str, str]:
    """Ask every selected extractor whether it can run, before any record is processed.

    Extractors that offer ``availability_problems()`` report missing weights, adapters, libraries or
    devices without loading a model. Returns ``{system: reason}`` for the unavailable systems.
    """
    unavailable: Dict[str, str] = {}
    for system in systems:
        try:
            probe = factories[system]()
        except Exception as exc:
            if _is_model_unavailable(exc):
                unavailable[system] = f"{type(exc).__name__}: {exc}"
                continue
            raise EvaluationAborted(f"System {system} cannot be constructed ({type(exc).__name__}: {exc})") from exc
        check = getattr(probe, "availability_problems", None)
        problems = list(check()) if callable(check) else []
        _release(probe)
        if problems:
            unavailable[system] = "; ".join(problems)
    return unavailable


def run_systems(
    records: Sequence[Mapping[str, Any]],
    factories: Mapping[str, Factory],
    engine: Any,
    *,
    strict: bool = True,
    systems: Optional[Sequence[str]] = None,
    explainer: Any = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Run the selected systems system by system; return prediction rows and skipped systems."""
    selected = list(systems) if systems else [s for s in SYSTEM_IDS if s in factories]
    unknown = [s for s in selected if s not in factories]
    if unknown:
        raise EvaluationAborted(f"Unknown system(s) {unknown}; available: {sorted(factories)}")

    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, str] = {}

    unavailable = preflight(factories, selected)
    if unavailable:
        if strict:
            detail = " | ".join(f"{name}: {reason}" for name, reason in unavailable.items())
            raise EvaluationAborted(
                f"Required models are unavailable: {detail}. Strict evaluation never falls back; provide the "
                "weights or select other systems with --systems."
            )
        skipped.update(unavailable)
        selected = [name for name in selected if name not in unavailable]

    for system in selected:
        if progress:
            progress(f"system {system}: start ({len(records)} records)")
        system_rows: List[Dict[str, Any]] = []
        extractor = None
        try:
            extractor = factories[system]()
            for record in records:
                output = extractor.extract(record["policy_text"])
                system_rows.append(_predict_row(system, record, _as_dict(output), engine, strict, explainer))
        except EvaluationAborted:
            raise
        except Exception as exc:
            if _is_model_unavailable(exc) and not strict:
                skipped[system] = f"{type(exc).__name__}: {exc}"
                if progress:
                    progress(f"system {system}: skipped ({exc})")
                continue
            if _is_model_unavailable(exc):
                raise EvaluationAborted(
                    f"System {system} is unavailable ({exc}). Strict evaluation never falls back; "
                    "provide the required model weights or select other systems with --systems."
                ) from exc
            raise EvaluationAborted(f"System {system} failed ({type(exc).__name__}: {exc})") from exc
        finally:
            if extractor is not None:
                _release(extractor)
        rows.extend(system_rows)
    return rows, skipped


# --------------------------------------------------------------------------- results


def project_characteristics(record: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    created, reference = parse_time(record.get("repo_created_at")), parse_time(record.get("collected_at"))
    age = (reference - created).total_seconds() / (365.25 * 86400) if created and reference else None
    stars = record.get("stars")
    words = record.get("policy_word_count")
    return {
        "project_age_years": age,
        "stars": float(stars) if isinstance(stars, (int, float)) and not isinstance(stars, bool) else None,
        "policy_length_words": float(words) if isinstance(words, (int, float)) and not isinstance(words, bool) else None,
    }


def _rows_by_system(records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    names = [r["package_name"] for r in records]
    grouped: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        per_system = grouped.setdefault(row["system"], {})
        if row["package_name"] in per_system:
            raise DataError(f"Duplicate prediction for {row['system']} / {row['package_name']}")
        per_system[row["package_name"]] = row
    ordered: Dict[str, List[Mapping[str, Any]]] = {}
    for system in [s for s in SYSTEM_IDS if s in grouped] + sorted(set(grouped) - set(SYSTEM_IDS)):
        missing = [n for n in names if n not in grouped[system]]
        extra = sorted(set(grouped[system]) - set(names))
        if missing or extra:
            raise DataError(
                f"Predictions of {system} do not match the test set (missing {missing[:3]}, unexpected {extra[:3]})"
            )
        ordered[system] = [grouped[system][n] for n in names]
    if not ordered:
        raise DataError("No predictions to evaluate")
    return ordered


def compute_results(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    scorer: Scorer,
    *,
    n_resamples: int = 1000,
    seed: int = SEED,
) -> Dict[str, Dict[str, Any]]:
    """Compute extraction_metrics, verdict_metrics, statistical_tests and failure_analysis from predictions."""
    by_system = _rows_by_system(records, rows)
    gold_commitments = [r["commitment"] for r in records]
    gold_verdicts = [r["verdict"] for r in records]

    extraction: Dict[str, Any] = {}
    verdicts: Dict[str, Any] = {}
    correctness: Dict[str, List[bool]] = {}
    exact_match: Dict[str, List[bool]] = {}
    columns: Dict[str, Dict[str, List[float]]] = {}
    for system, system_rows in by_system.items():
        predicted_commitments = [r["commitment"] for r in system_rows]
        columns[system] = score_columns(predicted_commitments, gold_commitments, scorer)
        extraction[system] = aggregate_extraction(predicted_commitments, gold_commitments, scorer, columns[system])
        predicted = [r["verdict"] for r in system_rows]
        verdicts[system] = classification_report(gold_verdicts, predicted, VERDICTS)
        correctness[system] = [g == p for g, p in zip(gold_verdicts, predicted)]
        exact_match[system] = [score >= 1.0 for score in columns[system]["tuple_exact_match"]]

    indices = bootstrap_indices(len(records), n_resamples, seed)
    bootstrap = {
        system: bootstrap_macro_f1(gold_verdicts, [r["verdict"] for r in system_rows], indices, VERDICTS, seed=seed)
        for system, system_rows in by_system.items()
    }
    bootstrap_extraction = {
        system: {
            name: bootstrap_mean_ci(columns[system][key], indices, seed=seed)
            for name, key in EXTRACTION_BOOTSTRAP_METRICS.items()
        }
        for system in by_system
    }

    mcnemar = _mcnemar_block(correctness, "per-record verdict correctness (predicted verdict equals the gold verdict)")
    mcnemar_tuple = _mcnemar_block(exact_match, "per-record tuple exact match of the extracted commitment c (EM = 1)")

    characteristics = [project_characteristics(r) for r in records]
    mann_whitney = {
        "grouping": "median split; low: value <= median, high: value > median",
        "outcome": "per-record verdict correctness (1 correct, 0 wrong)",
        "systems": {
            system: {
                name: mann_whitney_median_split([1.0 if c else 0.0 for c in flags], [ch[name] for ch in characteristics], name)
                for name in ("project_age_years", "stars", "policy_length_words")
            }
            for system, flags in correctness.items()
        },
    }
    activity = [project_activity(r) for r in records]
    mann_whitney_activity = {
        "grouping": ACTIVITY_DEFINITION,
        "outcome": "per-record verdict correctness (1 correct, 0 wrong)",
        "systems": {
            system: mann_whitney_groups([1.0 if c else 0.0 for c in flags], activity, ACTIVE, ABANDONED)
            for system, flags in correctness.items()
        },
    }

    return {
        "extraction_metrics": {
            "n_records": len(records),
            "metric_provider": f"{METRICS_MODULE}.{SCORER_NAME}",
            "systems": extraction,
        },
        "verdict_metrics": {"n_records": len(records), "labels": list(VERDICTS), "systems": verdicts},
        "statistical_tests": {
            "bootstrap_macro_f1": {
                "n_resamples": n_resamples,
                "seed": seed,
                "level": 0.95,
                "percentiles": [2.5, 97.5],
                "systems": bootstrap,
            },
            "bootstrap_extraction": {
                "n_resamples": n_resamples,
                "seed": seed,
                "level": 0.95,
                "percentiles": [2.5, 97.5],
                "resample_indices": "the same indices as bootstrap_macro_f1",
                "metrics": list(EXTRACTION_BOOTSTRAP_METRICS),
                "systems": bootstrap_extraction,
            },
            "mcnemar": mcnemar,
            "mcnemar_tuple_exact_match": mcnemar_tuple,
            "mann_whitney": mann_whitney,
            "mann_whitney_activity": mann_whitney_activity,
        },
        "failure_analysis": failure_analysis(records, by_system, columns),
    }


def _mcnemar_block(flags_by_system: Mapping[str, Sequence[bool]], outcome: str) -> Dict[str, Any]:
    if PROPOSED_SYSTEM not in flags_by_system:
        return {"status": "skipped", "reason": f"{PROPOSED_SYSTEM} was not evaluated"}
    return {
        "proposed_system": PROPOSED_SYSTEM,
        "outcome": outcome,
        "definition": "b: baseline correct and proposed wrong; c: baseline wrong and proposed correct",
        "baselines": {
            system: mcnemar_test(flags, flags_by_system[PROPOSED_SYSTEM])
            for system, flags in flags_by_system.items()
            if system != PROPOSED_SYSTEM
        },
    }


# --------------------------------------------------------------------------- orchestration


def run_evaluation(
    records: Sequence[Mapping[str, Any]],
    factories: Mapping[str, Factory],
    engine: Any,
    scorer: Scorer,
    out_dir: Path,
    *,
    strict: bool = True,
    systems: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    n_resamples: int = 1000,
    seed: int = SEED,
    data_files: Optional[Mapping[str, Path]] = None,
    models_dir: Optional[Path] = None,
    probe_gpu: bool = True,
    options: Optional[Mapping[str, Any]] = None,
    explainer: Any = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run, compute and write every result file in one step; raise before writing on any failure."""
    if not records:
        raise EvaluationAborted("The test set is empty")
    rows, skipped = run_systems(
        records, factories, engine, strict=strict, systems=systems, explainer=explainer, progress=progress
    )
    if not rows:
        raise EvaluationAborted("No system produced predictions")
    docs = compute_results(records, rows, scorer, n_resamples=n_resamples, seed=seed)

    predictions_text = jsonl_text(rows)
    manifest = build_manifest(
        data_files=data_files or {},
        models_dir=models_dir,
        seed=seed,
        probe_gpu=probe_gpu,
        run={
            "strict": strict,
            "limit": limit,
            "smoke_test": limit is not None,
            "systems_requested": list(systems) if systems else list(SYSTEM_IDS),
            "systems_evaluated": sorted({r["system"] for r in rows}),
            "skipped_systems": skipped,
            "n_test_records": len(records),
            "degraded_predictions": sum(1 for r in rows if r["provenance"].get("degraded")),
            "assessment_engine": type(engine).__name__,
            "explanation_generator": type(explainer).__name__ if explainer is not None else None,
            "metric_provider": docs["extraction_metrics"]["metric_provider"],
            "bootstrap_resamples": n_resamples,
            "extractor_provenance": _provenance_summary(rows),
            **dict(options or {}),
        },
    )
    manifest["data_files"]["predictions"] = {
        "path": "predictions.jsonl",
        "relative_to": "output_dir",
        "sha256": hashlib.sha256(predictions_text.encode("utf-8")).hexdigest(),
    }
    out_dir = Path(out_dir)
    files = {
        out_dir / "predictions.jsonl": predictions_text,
        out_dir / "extraction_metrics.json": dumps_json(docs["extraction_metrics"]) + "\n",
        out_dir / "verdict_metrics.json": dumps_json(docs["verdict_metrics"]) + "\n",
        out_dir / "statistical_tests.json": dumps_json(docs["statistical_tests"]) + "\n",
        out_dir / "failure_analysis.json": dumps_json(docs["failure_analysis"]) + "\n",
        out_dir / "run_manifest.json": dumps_json(manifest) + "\n",
    }
    atomic_write_many(files)
    return {"docs": docs, "manifest": manifest, "n_predictions": len(rows), "skipped": skipped}


def _provenance_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        p = row["provenance"]
        key = f"{p.get('extractor_engine')}|{p.get('actual_model_used')}|lora={p.get('lora_applied')}|{p.get('execution_device')}"
        entry = summary.setdefault(row["system"], {})
        entry[key] = entry.get(key, 0) + 1
    return summary
