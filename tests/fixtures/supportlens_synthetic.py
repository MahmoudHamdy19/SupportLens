"""SYNTHETIC TEST DATA. Invented only to exercise the code; never copy anything here into data/.

Nothing in this module is a real package, policy, annotation or result. Packages are called
``synthpkg-NN``; verdict labels are assigned by the test that needs them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SYNTHETIC_NOTICE = "synthetic test data, not a real dataset"

POLICY_TEMPLATE = """# Security Policy {n}

## Supported Versions

Only version {major}.x of synthpkg-{n:02d} receives security fixes. Older releases are not supported.

## Reporting a Vulnerability

Send a private report to security@synthetic.invalid and wait for our reply.
"""

COMMITMENT_SENTENCE = "Only version {major}.x of synthpkg-{n:02d} receives security fixes."


def policy_text(n: int, major: int = 2) -> str:
    return POLICY_TEMPLATE.format(n=n, major=major)


def commitment_sentence(n: int, major: int = 2) -> str:
    return COMMITMENT_SENTENCE.format(n=n, major=major)


def release(version: str, published_at: Optional[str]) -> Dict[str, Any]:
    return {"version": version, "tag": version, "published_at": published_at, "source": "synthetic"}


def vulnerability(identifier: str, affected: List[str], fixed: List[str], published_at: str = "2024-03-01T00:00:00Z") -> Dict[str, Any]:
    return {
        "id": identifier,
        "osv_id": identifier,
        "cve_ids": [],
        "ghsa_ids": [identifier],
        "aliases": [],
        "summary": "synthetic vulnerability",
        "details": "",
        "published_at": published_at,
        "modified_at": published_at,
        "affected_versions": affected,
        "affected_ranges": [],
        "fixed_versions": fixed,
        "is_patched": bool(fixed),
        "sources": ["synthetic"],
        "database_source": "synthetic",
        "advisory_url": None,
    }


def dataset_record(n: int, *, stars: int = 10, with_vulnerability: bool = False, major: int = 2) -> Dict[str, Any]:
    """One record of the shape written by the collector."""
    text = policy_text(n, major)
    vulnerabilities = [vulnerability(f"SYN-{n:03d}", [f"{major}.0.0"], [f"{major}.0.1"])] if with_vulnerability else []
    return {
        "package_name": f"synthpkg-{n:02d}",
        "github_repo": f"synthetic/synthpkg-{n:02d}",
        "policy_text": text,
        "policy_path": "SECURITY.md",
        "policy_url": f"https://example.invalid/synthpkg-{n:02d}/SECURITY.md",
        "releases": [release(f"{major}.0.0", "2023-01-01T00:00:00Z"), release(f"{major}.0.1", "2024-04-01T00:00:00Z")],
        "vulnerabilities": vulnerabilities,
        "source_diagnostics": {"osv": {"status": "SUCCESS", "count": len(vulnerabilities)}, "github_advisory": {"status": "SUCCESS", "count": 0}},
        "stars": stars,
        "repo_created_at": "2018-01-01T00:00:00Z",
        "pushed_at": "2026-01-01T00:00:00Z",
        "policy_word_count": len(text.split()),
        "collected_at": "2026-06-01T00:00:00+00:00",
    }


def dataset(count: int) -> List[Dict[str, Any]]:
    """Records with varied stars and vulnerabilities so that every stratum occurs."""
    return [
        dataset_record(n, stars=(n * 37) % 500, with_vulnerability=(n % 3 == 0)) for n in range(1, count + 1)
    ]


def gold_record(n: int, verdict: str, *, stars: int = 10, tau: str = "explicit_range") -> Dict[str, Any]:
    """A ground-truth style record (INTERFACES.md section 5) with synthetic provenance."""
    base = dataset_record(n, stars=stars, with_vulnerability=True)
    commitment = {
        "commitment_text": commitment_sentence(n),
        "version_expression": ["2.x"],
        "support_status": tau,
        "temporal_reference": "vulnerability_publication_time",
    }
    annotator = {"commitment": commitment, "verdict": verdict}
    return {
        "package_name": base["package_name"],
        "github_repo": base["github_repo"],
        "policy_text": base["policy_text"],
        "commitment": commitment,
        "verdict": verdict,
        "releases": base["releases"],
        "vulnerabilities": base["vulnerabilities"],
        "stars": stars,
        "repo_created_at": base["repo_created_at"],
        "policy_word_count": base["policy_word_count"],
        "collected_at": base["collected_at"],
        "source_diagnostics": base["source_diagnostics"],
        "annotation": {"annotator1": annotator, "annotator2": annotator, "adjudicated_by": None},
    }


# --------------------------------------------------------------------------- stubs


class ModelUnavailableError(RuntimeError):
    """Stand-in with the same name as the Layer 1 exception (matched by class name)."""


class StubExtractor:
    """Returns a fixed commitment dict with provenance; ``fail`` makes it raise instead."""

    def __init__(self, *, lora: bool = False, degraded: bool = False, commitment: Optional[Dict[str, Any]] = None,
                 fail: Optional[Exception] = None, engine: str = "stub-extractor"):
        self.lora = lora
        self.degraded = degraded
        self.fail = fail
        self.engine = engine
        self.commitment = commitment or {
            "commitment_text": "predicted sentence",
            "version_expression": ["3.x"],
            "support_status": "explicit_range",
            "temporal_reference": "release_time",
        }
        self.calls = 0

    def extract(self, text: str) -> Dict[str, Any]:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return {
            **self.commitment,
            "is_measurable": True,
            "extractor_engine": self.engine,
            "actual_model_used": "stub-model",
            "lora_applied": self.lora,
            "execution_device": "cpu",
            "degraded": self.degraded,
        }


class StubEngine:
    """Records every commitment it receives and answers with a verdict chosen by the test."""

    def __init__(self, verdict_for=None, with_explanation: bool = True):
        self.received: List[Dict[str, Any]] = []
        self.kwargs: List[Dict[str, Any]] = []
        self.verdict_for = verdict_for or (lambda commitment, package: "Consistent")
        self.with_explanation = with_explanation

    def evaluate(self, commitment, vulnerabilities, releases, reference_time=None, package_name=None,
                 source_diagnostics=None, mapping_uncertain=False):
        self.received.append(dict(commitment))
        self.kwargs.append(
            {"vulnerabilities": vulnerabilities, "releases": releases, "reference_time": reference_time,
             "package_name": package_name, "source_diagnostics": source_diagnostics}
        )
        verdict = self.verdict_for(commitment, package_name)
        return _StubResult(
            {
                "verdict": verdict,
                "rules_triggered": ["R1"],
                "explanation": f"Synthetic explanation for {package_name}: {verdict}." if self.with_explanation else None,
                "evidence_trail": [f"synthetic evidence line for {package_name}"],
            }
        )


class _StubResult:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


def stub_scorer(pred: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    """Deterministic per-record scores for aggregation tests (a stand-in for the Layer 1 helper)."""
    tau = 1.0 if pred.get("support_status") == gold.get("support_status") else 0.0
    temporal = 1.0 if pred.get("temporal_reference") == gold.get("temporal_reference") else 0.0
    text = 1.0 if pred.get("commitment_text") == gold.get("commitment_text") else 0.0
    version = 1.0 if pred.get("version_expression") == gold.get("version_expression") else 0.0
    return {
        "t_precision": text, "t_recall": text, "t_f1": text,
        "v_precision": version, "v_recall": version, "v_f1": version,
        "tau_match": tau, "T_match": temporal,
        "tuple_exact_match": 1.0 if (tau and temporal and text and version) else 0.0,
        "tuple_partial_match": (tau + temporal + text + version) / 4.0,
    }


class StubExplainer:
    """Stands in for the Layer 4 generator; records what it was given."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def generate_explanation(self, assessment, commitment=None, policy_citation=None, source_references=None):
        data = assessment.to_dict()
        self.calls.append({"commitment": commitment, "policy_citation": policy_citation, "source_references": source_references})
        return f"Layer 4 stub explanation: {data['verdict']}"
