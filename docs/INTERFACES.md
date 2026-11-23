# SupportLens: interface contract between layers

This file fixes the data shapes and function signatures that every layer, script and test relies on.
The authority for behaviour is the thesis text in `docs/thesis_text_extract.txt` (Chapter 3).
Run everything from the project root. Imports use the `src.` prefix (`pytest.ini` sets `pythonpath = .`).

## 1. Commitment tuple `c = <t, tau, v, T>` (Layer 1 output)

A plain dict (also returned by `CommitmentExtractionOutput.to_dict()`), with exactly these thesis keys:

| key | type | meaning |
|---|---|---|
| `commitment_text` | str | t: verbatim sentence(s) from SECURITY.md |
| `support_status` | `latest_only` \| `explicit_range` \| `all_versions` \| `ambiguous` | tau: commitment type |
| `version_expression` | list[str] | v: raw version tokens such as `["5.2.x", ">=3.0"]`, `["latest"]` |
| `temporal_reference` | `release_time` \| `vulnerability_publication_time` | T: anchor of the evaluation |

Derived and provenance keys (always present in `to_dict()`):
`is_measurable` (bool, derived only from the four fields: false when `support_status == "ambiguous"`, or when
`version_expression` is empty and `support_status == "explicit_range"`), `extractor_engine`, `actual_model_used`,
`lora_applied`, `execution_device`, `degraded` (true when a fallback produced the result).
There are NO other semantic keys (no SLA, no EOL, no short aliases `t`/`tau`/`v`/`T`).

## 2. Releases and vulnerabilities (Layer 0 and Layer 2 output)

Release (PyPI JSON API): `{"version": str, "tag": str, "published_at": ISO-8601 str | None, "source": str}`.

Vulnerability (one canonical record per CVE/GHSA, Layer 2 `EvidenceCollector`):
`id, osv_id, cve_ids, ghsa_ids, aliases, summary, details, published_at (None if missing), modified_at,
affected_versions, affected_ranges, fixed_versions, is_patched, sources, database_source, advisory_url`.

`EvidenceCollector().collect_unified_evidence(package_name, pypi_releases=None, github_token=None)` returns
`{"package_name", "vulnerabilities", "total_vulnerabilities", "unpatched_count", "source_diagnostics"}` where
`source_diagnostics` has the keys `osv`, `github_advisory`, `nvd` (and `pypi`), each `{"status": "SUCCESS"|"FAILURE"|...,
"count": int, "error": str | None}`.

## 3. Layer 1 API (`src/layer1_ai_extraction`)

```python
QwenCommitmentExtractor(lora_dir=None, use_adapter=True, strict=False, device=None).extract(text) -> CommitmentExtractionOutput
RuleBasedRegexBaseline().extract(text) -> CommitmentExtractionOutput
RoBERTaCommitmentExtractor(model_dir=None).extract(text) -> CommitmentExtractionOutput
```
* `strict=True` (used by every evaluation script): never falls back. If the model, the adapter or CUDA is missing it
  raises `ModelUnavailableError` (defined in `src/layer1_ai_extraction/errors.py`).
* `strict=False` (live app and CLI): may fall back to the regex baseline, and then `degraded=True` and
  `extractor_engine` says so.
* `use_adapter=False` is the zero-shot baseline (same base model, same prompt and schema, no LoRA adapter).

## 4. Layer 3 API (`src/layer3_rule_assessment`)

```python
AssessmentRuleEngine().evaluate(
    commitment: dict,                      # section 1
    vulnerabilities: list[dict] | None,    # section 2; None means "not collected"
    releases: list[dict] | None,           # section 2; None means "not collected"
    reference_time: datetime | None = None,
    package_name: str | None = None,
    source_diagnostics: dict | None = None,
    mapping_uncertain: bool = False,
) -> RuleAssessmentResult
```
`RuleAssessmentResult.to_dict()["verdict"]` is one of the five strings in `VERDICT_CATEGORIES`
(`Consistent`, `Partially Consistent`, `Potential Inconsistency`, `Insufficient Evidence`, `Unmeasurable Commitment`).
Rule ids are exactly `R1`..`R5` (Algorithm 1 of the thesis).

## 5. Ground-truth and split records (`data/ground_truth/*.jsonl`, one JSON object per line)

```json
{
  "package_name": "urllib3",
  "github_repo": "urllib3/urllib3",
  "policy_text": "<Layer 0 preprocessed SECURITY.md text: PolicyPreprocessor.clean_text(raw)>",
  "raw_policy_text": "<unmodified SECURITY.md, kept for reference>",
  "commitment": {"commitment_text": "...", "version_expression": ["..."], "support_status": "...", "temporal_reference": "...",
                 "temporal_expression": "<verbatim phrase in policy_text that anchors T, or null>"},
  "verdict": "Consistent",
  "releases": [...], "vulnerabilities": [...],
  "stars": 0, "repo_created_at": "ISO-8601", "policy_word_count": 0,
  "repo_fork": false, "repo_archived": false, "mapping_uncertain": false,
  "annotation": {"annotator1": {...}, "annotator2": {...}, "adjudicated_by": "supervisor" | null}
}
```
`policy_text` is the Layer 0 preprocessed text (`PolicyPreprocessor.clean_text`: HTML comments,
badges and extra blank lines removed, line endings LF). It is exactly the Layer 1 input of
`SupportLensPipeline` and of `supportlens_cli.py --file`, so annotation, training and inference see
identical text (thesis 3.2.1/3.2.2). `mapping_uncertain` is true when the repository is a fork or
archived; it is passed to `AssessmentRuleEngine.evaluate(mapping_uncertain=...)` (Algorithm 1, R4).
Files: `ground_truth.jsonl` (all), `train.jsonl`, `val.jsonl`, `test.jsonl` (stratified 70/15/15, seed 42).
No script may create or edit these files from heuristics. They come only from human annotation and adjudication.

## 6. Non-negotiable project rules

1. Every number in `data/evaluation/*.json`, in the README and in any report is written by a script from a real run.
2. No simulated annotators, no simulated baselines, no gold commitments fed to Layer 3 in an evaluation.
3. Seeds: `42` everywhere (random, numpy, torch, transformers). Never use Python `hash()` for anything that affects results.
4. Network code lives only in Layer 0, Layer 2 and the collection script. Everything else works offline.
5. Tests are offline by default; tests that need the network or a GPU carry `@pytest.mark.network` or `@pytest.mark.gpu`.
6. No secrets in the repository. `.env` is ignored. `.env.example` lists only `GITHUB_TOKEN` and `NVD_API_KEY`.
