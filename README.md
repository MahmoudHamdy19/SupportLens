# SupportLens

**An explainable AI framework for assessing version-support commitments against vulnerability and release evidence.**

This repository implements the system described in Chapter 3 of the thesis (`docs/thesis_text_extract.txt`).
It maps a PyPI package's security policy `S`, vulnerability evidence `V` and release history `R`
to a verdict `y` and an evidence-grounded explanation `ε`: `f(S, V, R) → (y, ε)`.
Layer contracts are fixed in [docs/INTERFACES.md](docs/INTERFACES.md). The thesis-to-code mapping is in
[docs/THESIS_TRACEABILITY.md](docs/THESIS_TRACEABILITY.md).

## Architecture (thesis 3.2)

| Layer | Package | Thesis | Tools |
|---|---|---|---|
| 0 Data acquisition and preprocessing | `src/layer0_data_acquisition` | 3.2.1 | requests, PyGithub |
| 1 AI extraction of `c = <t, τ, v, T>` | `src/layer1_ai_extraction` | 3.2.2 | Qwen2.5-7B-Instruct + LoRA, transformers, peft |
| 2 Evidence integration and normalisation | `src/layer2_evidence_layer` | 3.2.3 | OSV, GitHub Advisory, NVD, PyPI, packaging |
| 3 Rule-based assessment (Algorithm 1, R1–R5) | `src/layer3_rule_assessment` | 3.2.4 | Python, pydantic |
| 4 Explanation and output | `src/layer4_explanation_output` | 3.2.5 | Jinja2, json |

The five verdicts (Table 3.1) are Consistent, Partially Consistent, Potential Inconsistency, Insufficient Evidence
and Unmeasurable Commitment.

## Installation

```powershell
python -m pip install -r requirements.txt
copy .env.example .env    # then set GITHUB_TOKEN (and optionally NVD_API_KEY)
```
Training and the neural evaluation also need `requirements-gpu.txt` and a CUDA GPU (for example Colab T4).

## Using the system

```powershell
python supportlens_cli.py --pypi urllib3              # five-layer audit of one package
python supportlens_cli.py --file path\SECURITY.md      # Layer 1 extraction of a local policy
streamlit run app.py                                   # web interface
```
Without a GPU, Layer 1 uses the regex baseline and every output is marked `degraded`.
Use `--strict` to require the fine-tuned model instead.

## Research workflow (thesis 3.3)

| Step | Command | Output | Who |
|---|---|---|---|
| 1 Collect PyPI policies with a version-support statement | `python scripts/collect_dataset.py --target 300` | `data/processed/pypi_dataset.jsonl`, `data/audit/` | automatic |
| 2 Stratified random sample of 300 | `python scripts/sample_dataset.py` | `data/annotations/sample_300.jsonl` | automatic |
| 3 Blind annotation tasks (Label Studio, codebook) | `python scripts/prepare_annotation.py` | task files per annotator | automatic |
| 4 Independent annotation of `c` and `y` | Label Studio | two annotator exports | **two annotators** |
| 5 Cohen's kappa (target κ ≥ 0.80) | `python scripts/check_agreement.py` | `agreement_report.json` | automatic |
| 6 Adjudication of disagreements | `python scripts/adjudicate.py` | `data/ground_truth/ground_truth.jsonl` | **supervisor** |
| 7 70/15/15 stratified splits, seed 42 | `python scripts/make_splits.py` | `train/val/test.jsonl` | automatic |
| 8 Fine-tune Qwen2.5-7B with LoRA, train RoBERTa | `notebooks/SupportLens_Colab_Train_and_Eval.ipynb` | `models/` | **GPU** |
| 9 Evaluation: extraction (Q1), verdicts (Q2), robustness (Q3), statistics | `python scripts/run_evaluation.py --strict` | `data/evaluation/*.json` | **GPU** |
| 10 Expert rubric (correctness, completeness, traceability) | `python scripts/run_expert_evaluation.py --prepare` / `--ingest` | `expert_evaluation.json` | **expert raters** |
| 11 Check that every reported number matches the run | `python scripts/verify_results.py` | pass or fail | automatic |

No script produces labels, ratings or metrics from heuristics or random numbers. Steps that need people or a GPU
stop with a clear message until their inputs exist.

## Results

The figures below are filled by `python scripts/render_results.py` after step 9 and checked by step 11.

| System | Verdict macro-F1 | Tuple EM | Tuple PM |
|---|---|---|---|
| Regex baseline | <!-- RESULT:verdict_metrics.systems.regex_baseline.macro_f1 -->0.7542<!-- /RESULT --> | <!-- RESULT:extraction_metrics.systems.regex_baseline.tuple.exact_match -->0.0444<!-- /RESULT --> | <!-- RESULT:extraction_metrics.systems.regex_baseline.tuple.partial_match -->0.7222<!-- /RESULT --> |
| Zero-shot Qwen2.5-7B | <!-- RESULT:verdict_metrics.systems.qwen_zero_shot.macro_f1 -->0.5270<!-- /RESULT --> | <!-- RESULT:extraction_metrics.systems.qwen_zero_shot.tuple.exact_match -->0.0000<!-- /RESULT --> | <!-- RESULT:extraction_metrics.systems.qwen_zero_shot.tuple.partial_match -->0.3538<!-- /RESULT --> |
| RoBERTa-base | <!-- RESULT:verdict_metrics.systems.roberta_finetuned.macro_f1 -->0.6485<!-- /RESULT --> | <!-- RESULT:extraction_metrics.systems.roberta_finetuned.tuple.exact_match -->0.4667<!-- /RESULT --> | <!-- RESULT:extraction_metrics.systems.roberta_finetuned.tuple.partial_match -->0.8119<!-- /RESULT --> |
| Qwen2.5-7B + LoRA (proposed) | <!-- RESULT:verdict_metrics.systems.qwen_finetuned_lora.macro_f1 -->0.8409<!-- /RESULT --> | <!-- RESULT:extraction_metrics.systems.qwen_finetuned_lora.tuple.exact_match -->0.8222<!-- /RESULT --> | <!-- RESULT:extraction_metrics.systems.qwen_finetuned_lora.tuple.partial_match -->0.8889<!-- /RESULT --> |

## Tests

```powershell
python -m pytest                 # offline suite
python -m pytest -m network      # live PyPI, GitHub, OSV and NVD checks
```

## Layout

```
src/                   five layers, data_pipeline (collection, annotation, splits), evaluation
scripts/               command-line entry points for the research workflow
notebooks/             Colab notebook for GPU training and evaluation
data/                  seed list, processed dataset, annotations, ground truth, evaluation, reports, audit log
docs/                  interface contract, thesis traceability, thesis text extract
tests/                 offline unit and integration tests (synthetic fixtures only)
```
