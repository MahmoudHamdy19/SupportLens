# Thesis-to-code traceability

Every implemented claim of thesis Chapters 1–3 (`docs/thesis_text_extract.txt`) mapped to the code.
Status: **Implemented**; **Tooling ready** (the code is complete, but the step needs people, a GPU or
trained weights); **Interpretation** (the thesis leaves the point open; the resolution is documented in
the named module).

| Thesis | Claim | Status | Where |
|---|---|---|---|
| 1.2, 3.1.2 | Scope: PyPI packages with a GitHub repository and a SECURITY.md that states version support | Implemented | `src/layer0_data_acquisition/data_acquisition_pipeline.py` (version-support gate), `src/data_pipeline/collector.py` |
| 3.1.3.1 | Inputs S, V (OSV, NVD, GitHub Advisory), R (PyPI) | Implemented | `layer0_data_acquisition`, `layer2_evidence_layer/evidence_sources.py` |
| 3.1.3.2 | Tuple c = <t, τ, v, T>, τ in four classes, T in two | Implemented | `layer1_ai_extraction/schema.py`, `docs/INTERFACES.md` §1 |
| 3.1.3.3–4 | f(S, V, R) → (y, ε), five exclusive verdicts | Implemented | `src/supportlens_pipeline.py`, `layer3_rule_assessment/verdict_categories.py` |
| 3.2.1 | PyPI project_urls, GitHub API verification (PyGithub), SECURITY.md retrieval, Markdown preprocessing | Implemented | `layer0_data_acquisition/*` |
| 3.2.1 | Exclusion reasons written to an audit log | Implemented | `data/audit/exclusion_log.jsonl` |
| 3.2.2 | Qwen2.5-7B-Instruct + LoRA (peft); input is the preprocessed policy text | Implemented | `layer1_ai_extraction/qwen_extractor.py`, `train_qwen_lora.py` |
| 3.2.2 | Generation strictly constrained by the JSON schema | Implemented | `layer1_ai_extraction/constrained_decoding.py` |
| 3.2.2, 3.3.2 | LoRA on the attention layers of every Transformer layer, about 1–2% trainable | Implemented | `train_qwen_lora.py` (q/k/v/o, r=128 ≈ 1.06%; 4-bit base to fit one T4 GPU) |
| 3.2.3 | Merge and de-duplicate by CVE ID; pandas, packaging, OSV API | Implemented | `evidence_sources.py`, `version_normalization.py` |
| 3.2.3 | Resolve "latest" against the release history at time T | Implemented | `version_normalization.py`, `normalized_scope.py` |
| 3.2.4 | Algorithm 1: R5 → R4 → per-vulnerability R1/R2/R3 → aggregation | Implemented | `layer3_rule_assessment/rule_engine.py` |
| 3.2.4 | "Reasonable timeframe" for R3 | Interpretation | 90 days, `rule_engine.py` (`REASONABLE_TIMEFRAME_DAYS`) |
| 3.2.4 | Empty V, or no vulnerability inside the commitment scope | Interpretation | Insufficient Evidence (nothing to test the commitment against), `rule_engine.py` |
| 3.2.4 | "Partial" fix | Interpretation | Fixed on some affected release lines only; a release line is the major version, or the major.minor line when the commitment names that minor line (e.g. "2.1.x" and "2.2.x", or ">=2.1,<2.3"), `rule_engine.py` (`_release_line_key`, `_remediated`) |
| 3.2.4 | Aggregation by highest severity, consistent with Table 3.1 | Implemented | R3 with R1/R2 → Partially Consistent; R3 alone (even with R4) → Potential Inconsistency; R2 → Partially Consistent; R4 → Insufficient Evidence; else Consistent |
| 3.2.4 | Mapping uncertain → R4 | Implemented | Fork or archived repository (`repository_mapping.py`, `supportlens_pipeline.py`) |
| 3.2.5 | Jinja2 template explanation with CVEs, dates, URLs and policy line numbers; JSON record | Implemented | `layer4_explanation_output/*`, `data/reports/` |
| 3.3.1 | Data collection from the PyPI JSON API | Implemented | `scripts/collect_dataset.py` |
| 3.3.1 | Stratified random sample of 300 | Implemented / Interpretation | `scripts/sample_dataset.py`; strata = stars tercile × has a vulnerability record |
| 3.3.1 | Two independent annotators, codebook, Label Studio | Tooling ready | `scripts/prepare_annotation.py`, `data/annotations/annotation_guidelines.md`, `label_studio_config.xml` |
| 3.3.1 | Cohen's κ before adjudication (target ≥ 0.80) | Implemented | `scripts/check_agreement.py` |
| 3.3.1 | Supervisor adjudication | Tooling ready | `scripts/adjudicate.py` |
| 3.3.1 | 70/15/15 split | Implemented | `scripts/make_splits.py` (stratified by verdict, seed 42) |
| 3.3.2 | Regex/keyword baseline | Implemented | `layer1_ai_extraction/baselines.py` |
| 3.3.2 | Zero-shot Qwen2.5-7B with the same schema | Implemented | `QwenCommitmentExtractor(use_adapter=False)` |
| 3.3.2 | RoBERTa-base token classification of t, v and the temporal reference; AdamW, CE, lr 2e-5, batch 16, 5 epochs | Implemented / Tooling ready; CE is class-weighted (sqrt inverse frequency) so the rare VERSION labels are learned, `--class-weighting none` restores the uniform loss | `roberta_baseline.py`, `train_roberta_baseline.py` |
| 3.3.3 | Token-level P/R/F1; tuple EM and PM (Q1) | Implemented | `layer1_ai_extraction/metrics.py`, `src/evaluation/extraction_eval.py` |
| 3.3.3 | Macro-F1 and 5×5 confusion matrix (Q2), end to end with predicted commitments | Implemented | `src/evaluation/runner.py`, `verdict_metrics.py` |
| 3.3.3 | Expert rubric: correctness, completeness, traceability, 1–5 | Tooling ready | `scripts/run_expert_evaluation.py` |
| 3.3.3, 1.2 Q3 | Robustness across project age, stars, policy length; failure classes; active vs abandoned | Implemented | `src/evaluation/runner.py`, `failure_analysis.py` |
| 3.3.4 | McNemar (fine-tuned vs each baseline), Mann-Whitney U, bootstrap 95% CI (1,000) | Implemented | `src/evaluation/stats.py` |
| 3.4.1 | Open-source code; dataset on Zenodo with a DOI | Implemented / Tooling ready | `LICENSE`, `CITATION.cff`, `.zenodo.json`, `scripts/package_dataset.py` |
| 3.4.1 | Seed 42 for splits and training; hardware, CUDA and hyper-parameters recorded | Implemented | `layer1_ai_extraction/seeding.py`, `run_metadata.py`, `src/evaluation/repro.py` |
| 3.4.2 | Deterministic verdicts and template explanations | Implemented | Layers 3 and 4; evaluation refuses fallbacks (`--strict`) |
| 3.4.2 | Every reported number comes from a run | Implemented | `scripts/verify_results.py`, `scripts/render_results.py` |
