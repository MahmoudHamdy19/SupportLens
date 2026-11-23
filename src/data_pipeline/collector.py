"""Collection of the PyPI dataset (thesis Sections 3.2.1 and 3.3.1).

Candidates are the seed list followed by the public top-PyPI list. Each candidate goes
through the Layer 0 pipeline with the version-support gate; an included package is then
enriched by the Layer 2 evidence collector and appended to ``pypi_dataset.jsonl``.

Design rules
------------
* Resumable: package names already stored in the dataset, or excluded for a definitive
  reason (Layer 0 exclusion log or this module's collection log), are skipped.
* Transient failures (network errors, rate limits, a failed evidence source) are logged
  and retried by the next run; they never become an exclusion and never produce a record
  with silently missing evidence. ``--allow-partial-evidence`` overrides this and keeps
  the source diagnostics in the record.
* ``GITHUB_TOKEN`` is read from the environment only. It is never printed or written.
* ``policy_text`` is the Layer 0 preprocessed text (``PolicyPreprocessor.clean_text``: HTML
  comments, badges and extra blank lines removed, line endings LF), which is exactly the Layer 1
  input of the pipeline, so annotators, training and inference see identical characters. The
  unmodified SECURITY.md is kept in ``raw_policy_text``.
* ``repo_fork`` and ``repo_archived`` come from the GitHub repository metadata;
  ``mapping_uncertain`` is True when the repository is a fork or archived (Algorithm 1, R4).

The repository metadata (stars, creation and push dates) comes from ``result["repo_metadata"]``
of Layer 0; when it is absent the collector asks the GitHub API once more through the Layer 0
repository mapper. Both the Layer 0 key names
(``stars``) and the raw GitHub names (``stargazers_count``) are accepted.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import requests

from .io_utils import (
    PROJECT_ROOT,
    DataError,
    append_jsonl,
    atomic_write_many,
    dumps_json,
    normalise_package_name,
    read_json,
    read_jsonl,
    utc_now_iso,
    word_count,
)

TOP_PYPI_URL = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json"

log = logging.getLogger("supportlens.collector")

_TRANSIENT_MARKERS = (
    "connection error",
    "access_failure",
    "could not be confirmed",
    "rate limit",
    "unexpected http",
    "timeout",
    "timed out",
)


class CandidateListError(DataError):
    """The candidate list could not be built."""


class CollectionAborted(RuntimeError):
    """Too many consecutive transient failures; the run stops instead of burning the quota."""


@dataclass
class CollectorConfig:
    seed_file: Path = PROJECT_ROOT / "data" / "seed" / "pypi_seed_packages.txt"
    top_cache: Path = PROJECT_ROOT / "data" / "cache" / "top-pypi-packages.min.json"
    output: Path = PROJECT_ROOT / "data" / "processed" / "pypi_dataset.jsonl"
    audit_dir: Path = PROJECT_ROOT / "data" / "audit"
    target: Optional[int] = None
    max_candidates: Optional[int] = None
    delay: float = 1.0
    top_limit: Optional[int] = None
    refresh_top_list: bool = False
    seed_only: bool = False
    only_packages: Optional[List[str]] = None
    allow_partial_evidence: bool = False
    abort_after_transient: int = 25

    @property
    def exclusion_log(self) -> Path:
        return Path(self.audit_dir) / "exclusion_log.jsonl"

    @property
    def collection_log(self) -> Path:
        return Path(self.audit_dir) / "collection_log.jsonl"


@dataclass
class CollectionSummary:
    total_candidates: int = 0
    skipped_done: int = 0
    attempted: int = 0
    included: int = 0
    excluded: int = 0
    transient: int = 0
    errors: int = 0
    dataset_size: int = 0
    stopped_reason: str = ""
    transient_details: List[str] = field(default_factory=list)


@dataclass
class ProcessedState:
    included: set = field(default_factory=set)
    excluded: set = field(default_factory=set)


# --------------------------------------------------------------------------- candidates


def load_seed_packages(path: Path) -> List[str]:
    path = Path(path)
    if not path.is_file():
        raise CandidateListError(f"Seed file not found: {path}")
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def _parse_top_payload(data: Any, source: str) -> List[str]:
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise CandidateListError(f"{source}: unexpected top-PyPI payload (no 'rows' list)")
    names = []
    for row in rows:
        project = row.get("project") if isinstance(row, dict) else None
        if isinstance(project, str) and project:
            names.append(project)
    if not names:
        raise CandidateListError(f"{source}: the payload contains no project names")
    return names


def fetch_top_packages(
    cache_path: Path,
    *,
    url: str = TOP_PYPI_URL,
    refresh: bool = False,
    http_get: Optional[Callable[..., Any]] = None,
    timeout: int = 60,
) -> List[str]:
    """Return the top-PyPI project names, using the cache unless ``refresh`` is set.

    A failed download falls back to an existing cache; without a cache it raises.
    """
    cache_path = Path(cache_path)
    if cache_path.is_file() and not refresh:
        return _parse_top_payload(read_json(cache_path), str(cache_path))
    http_get = http_get or requests.get
    try:
        response = http_get(url, headers={"User-Agent": "SupportLens-Collector/1.0"}, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        names = _parse_top_payload(data, url)
    except (requests.RequestException, ValueError, CandidateListError) as exc:
        if cache_path.is_file():
            log.warning("Could not refresh the top-PyPI list (%s); using the cached copy.", exc)
            return _parse_top_payload(read_json(cache_path), str(cache_path))
        raise CandidateListError(
            f"Could not download the top-PyPI list from {url} and no cache exists at {cache_path}: {exc}. "
            "Retry later, or run with --seed-only to use the seed list alone."
        ) from exc
    atomic_write_many({cache_path: dumps_json(data, indent=None) + "\n"})
    return names


def build_candidates(
    seed: Sequence[str],
    top: Sequence[str],
    *,
    only: Optional[Sequence[str]] = None,
    top_limit: Optional[int] = None,
) -> List[str]:
    """Seed packages first, then the top list; duplicates (PEP 503 normalised) are dropped."""
    if only:
        ordered = list(only)
    else:
        ordered = list(seed) + (list(top[:top_limit]) if top_limit else list(top))
    seen: set = set()
    candidates: List[str] = []
    for name in ordered:
        key = normalise_package_name(name)
        if key and key not in seen:
            seen.add(key)
            candidates.append(name.strip())
    return candidates


# --------------------------------------------------------------------------- state


def classify_exclusion(reason: Optional[str], retrieval_status: Optional[str] = None) -> str:
    """Return ``"transient"`` or ``"definitive"`` for a Layer 0 exclusion.

    The classification keys off the phrases that Layer 0 writes into its exclusion log
    (connection errors, rate limits, unconfirmed GitHub verification, access failures,
    PyPI 429/5xx). Everything else, for example a missing repository or SECURITY.md or a
    missing version-support statement, is a definitive exclusion.
    """
    if retrieval_status == "ACCESS_FAILURE":
        return "transient"
    text = (reason or "").lower()
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return "transient"
    match = re.search(r"pypi api returned http (\d+)", text)
    if match and (int(match.group(1)) >= 500 or int(match.group(1)) == 429):
        return "transient"
    return "definitive"


def load_processed_state(config: CollectorConfig) -> ProcessedState:
    state = ProcessedState()
    if Path(config.output).is_file():
        for record in read_jsonl(config.output):
            state.included.add(normalise_package_name(record["package_name"]))
    if config.exclusion_log.is_file():
        for entry in read_jsonl(config.exclusion_log):
            name = entry.get("package_name")
            if name and classify_exclusion(entry.get("exclusion_reason"), entry.get("retrieval_status")) == "definitive":
                state.excluded.add(normalise_package_name(name))
    if config.collection_log.is_file():
        for entry in read_jsonl(config.collection_log):
            if entry.get("outcome") == "EXCLUDED" and entry.get("package_name"):
                state.excluded.add(normalise_package_name(entry["package_name"]))
    return state


# --------------------------------------------------------------------------- records


def incomplete_sources(diagnostics: Mapping[str, Any]) -> List[str]:
    """Evidence sources that did not succeed. OSV and the GitHub Advisory database are required."""
    problems = []
    for source in ("osv", "github_advisory"):
        status = (diagnostics.get(source) or {}).get("status")
        if status != "SUCCESS":
            problems.append(f"{source}={status}")
    return problems


def normalise_repo_metadata(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Accept both Layer 0 keys (``stars``) and raw GitHub keys (``stargazers_count``)."""
    stars = meta.get("stars", meta.get("stargazers_count"))
    return {
        "stars": stars,
        "created_at": meta.get("created_at"),
        "pushed_at": meta.get("pushed_at"),
        "archived": meta.get("archived"),
        "fork": meta.get("fork"),
    }


def normalise_policy_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def build_record(
    result: Mapping[str, Any],
    evidence: Mapping[str, Any],
    repo_metadata: Mapping[str, Any],
    collected_at: str,
) -> Dict[str, Any]:
    raw_text = normalise_policy_text(result.get("raw_policy_text") or "")
    preprocessed = result.get("preprocessed_text")
    if preprocessed is None:
        from src.layer0_data_acquisition.policy_preprocessing import PolicyPreprocessor

        preprocessed = PolicyPreprocessor.clean_text(raw_text)
    policy_text = normalise_policy_text(preprocessed)
    if not policy_text.strip():
        raise ValueError("Layer 0 returned an empty policy text for an included package")
    fork = repo_metadata.get("fork")
    archived = repo_metadata.get("archived")
    return {
        "package_name": result["package_name"],
        "github_repo": result["github_repo"],
        "policy_text": policy_text,
        "raw_policy_text": raw_text,
        "policy_path": result.get("security_policy_path"),
        "policy_url": result.get("security_policy_url"),
        "releases": list(result.get("releases") or []),
        "vulnerabilities": list(evidence.get("vulnerabilities") or []),
        "source_diagnostics": dict(evidence.get("source_diagnostics") or {}),
        "stars": repo_metadata["stars"],
        "repo_created_at": repo_metadata["created_at"],
        "pushed_at": repo_metadata["pushed_at"],
        "repo_fork": fork,
        "repo_archived": archived,
        "mapping_uncertain": bool(fork) or bool(archived),
        "policy_word_count": word_count(policy_text),
        "collected_at": collected_at,
    }


# --------------------------------------------------------------------------- collector


class DatasetCollector:
    """Runs the collection loop. Collaborators are injectable so the loop is testable offline."""

    def __init__(
        self,
        config: CollectorConfig,
        *,
        pipeline: Any = None,
        evidence_collector: Any = None,
        repo_metadata_fn: Optional[Callable[[str], Mapping[str, Any]]] = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], str] = utc_now_iso,
    ):
        self.config = config
        self.token = os.environ.get("GITHUB_TOKEN") or None
        self._pipeline = pipeline
        self._evidence = evidence_collector
        self._repo_metadata_fn = repo_metadata_fn
        self._sleep = sleep
        self._now = now

    # ---- wiring

    def _wire(self) -> None:
        if self._pipeline is None:
            from src.layer0_data_acquisition import data_acquisition_pipeline as layer0

            layer0.EXCLUSION_LOG_PATH = self.config.exclusion_log
            self._pipeline = layer0.DataAcquisitionPipeline(github_token=self.token)
        if self._evidence is None:
            from src.layer2_evidence_layer import EvidenceCollector

            self._evidence = EvidenceCollector()

    def _scrub(self, text: str) -> str:
        return text.replace(self.token, "***") if self.token else text

    def _repo_metadata(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        """Stars and dates of the repository: from Layer 0 when present, else one extra API call."""
        provided = normalise_repo_metadata(result.get("repo_metadata") or {})
        if provided["stars"] is not None and provided["created_at"] is not None:
            return provided
        repo = result["github_repo"]
        if self._repo_metadata_fn is not None:
            return normalise_repo_metadata(self._repo_metadata_fn(repo))
        from src.layer0_data_acquisition.repository_mapping import RepositoryMapper

        owner, name = repo.split("/", 1)
        valid, _, message, meta, _ = RepositoryMapper.verify_via_github_api(owner, name, github_token=self.token)
        if not valid or not meta:
            raise RuntimeError(f"repository metadata unavailable: {message}")
        return normalise_repo_metadata(meta)

    def _log(self, name: str, outcome: str, reason: Optional[str] = None, repo: Optional[str] = None) -> None:
        append_jsonl(
            self.config.collection_log,
            {
                "timestamp": self._now(),
                "package_name": name,
                "outcome": outcome,
                "reason": self._scrub(reason) if reason else None,
                "github_repo": repo,
            },
        )

    # ---- candidates

    def load_candidates(self) -> List[str]:
        cfg = self.config
        if cfg.only_packages:
            return build_candidates([], [], only=cfg.only_packages)
        seed = load_seed_packages(cfg.seed_file)
        top: List[str] = []
        if not cfg.seed_only:
            top = fetch_top_packages(cfg.top_cache, refresh=cfg.refresh_top_list)
        return build_candidates(seed, top, top_limit=cfg.top_limit)

    # ---- one package

    def _process(self, name: str) -> tuple:
        """Return ``(outcome, detail)`` with outcome in INCLUDED, EXCLUDED, TRANSIENT, ERROR."""
        try:
            result = self._pipeline.process_package(name, require_version_support=True)
        except Exception as exc:  # network or unexpected failure: log and continue
            return "ERROR", self._scrub(f"{type(exc).__name__}: {exc}")

        if result.get("status") != "INCLUDED":
            reason = result.get("exclusion_reason") or "excluded without reason"
            kind = classify_exclusion(reason, result.get("retrieval_status"))
            return ("TRANSIENT" if kind == "transient" else "EXCLUDED"), reason

        try:
            meta = self._repo_metadata(result)
            if meta["stars"] is None or meta["created_at"] is None:
                return "TRANSIENT", "repository metadata lacks stars or creation date"
            evidence = self._evidence.collect_unified_evidence(
                name, pypi_releases=result.get("releases"), github_token=self.token
            )
            problems = incomplete_sources(evidence.get("source_diagnostics") or {})
            if problems and not self.config.allow_partial_evidence:
                return "TRANSIENT", "evidence sources incomplete: " + ", ".join(problems)
            record = build_record(result, evidence, meta, self._now())
        except Exception as exc:
            return "ERROR", self._scrub(f"{type(exc).__name__}: {exc}")

        append_jsonl(self.config.output, record)
        detail = f"stars={record['stars']} vulns={len(record['vulnerabilities'])} words={record['policy_word_count']}"
        return "INCLUDED", detail

    # ---- main loop

    def run(self) -> CollectionSummary:
        cfg = self.config
        self._wire()
        state = load_processed_state(cfg)
        candidates = self.load_candidates()
        summary = CollectionSummary(total_candidates=len(candidates), dataset_size=len(state.included))
        if not self.token:
            log.warning(
                "GITHUB_TOKEN is not set: the GitHub API allows only 60 requests per hour. "
                "A full collection needs a token."
            )

        consecutive_trouble = 0
        for name in candidates:
            if cfg.target is not None and summary.dataset_size >= cfg.target:
                summary.stopped_reason = f"target of {cfg.target} records reached"
                break
            if cfg.max_candidates is not None and summary.attempted >= cfg.max_candidates:
                summary.stopped_reason = f"--max-candidates {cfg.max_candidates} reached"
                break
            key = normalise_package_name(name)
            if key in state.included or key in state.excluded:
                summary.skipped_done += 1
                continue

            outcome, detail = self._process(name)
            summary.attempted += 1
            self._log(name, outcome, detail if outcome != "INCLUDED" else None)
            log.info("[%d] %s -> %s %s", summary.attempted, name, outcome, detail or "")

            if outcome == "INCLUDED":
                summary.included += 1
                summary.dataset_size += 1
                state.included.add(key)
                consecutive_trouble = 0
            elif outcome == "EXCLUDED":
                summary.excluded += 1
                state.excluded.add(key)
                consecutive_trouble = 0
            else:
                if outcome == "TRANSIENT":
                    summary.transient += 1
                else:
                    summary.errors += 1
                summary.transient_details.append(f"{name}: {detail}")
                consecutive_trouble += 1
                if consecutive_trouble >= cfg.abort_after_transient:
                    raise CollectionAborted(
                        f"{consecutive_trouble} consecutive transient failures (last: {detail}). "
                        "Typical causes are an exhausted GitHub rate limit (set GITHUB_TOKEN) or a lost "
                        "connection. Finished records are saved; run again to resume."
                    )
            if cfg.delay > 0:
                self._sleep(cfg.delay)
        if not summary.stopped_reason:
            summary.stopped_reason = "candidate list exhausted"
        return summary
