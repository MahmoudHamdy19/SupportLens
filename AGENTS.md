# Rules for AI assistants working on SupportLens

Read this whole file before you change anything. The same file is provided as `AGENTS.md`, `GEMINI.md`
and `CLAUDE.md`, so every assistant (Gemini, Claude, Copilot, Codex and others) gets identical rules.
If the user's request conflicts with a rule here, stop and ask the user before acting.

## 1. What this project is

SupportLens is the practical implementation of a master thesis (Master of Applied Artificial Intelligence,
Prince Sattam bin Abdulaziz University). It extracts version-support commitments from PyPI packages'
SECURITY.md files with an LLM and checks them against vulnerability and release evidence. The verdict
comes from deterministic rules, and the explanation comes from templates.

**The thesis is the single source of truth.** Its text is `docs/thesis_text_extract.txt` (Chapters 1–3
are the specification; Chapters 4–6 are still an empty template). The code must implement exactly what
the thesis describes: no more, no less. The user's instruction is that **only the code is changed;
the thesis is never edited by an assistant.**

Read these before working:
- `docs/INTERFACES.md`: binding data shapes and function signatures between the layers.
- `docs/THESIS_TRACEABILITY.md`: each thesis claim and the file that implements it.
- `README.md`: how to run everything, and the research workflow table.

## 2. Non-negotiable rules

1. **Never fabricate data, labels, ratings or results.** No simulated annotators, no heuristic or random
   ground-truth labels, no invented expert ratings, no hand-written or hand-edited numbers in
   `data/evaluation/*.json`, `README.md`, reports or the thesis. Every number must be written by a
   script from a real run. `python scripts/verify_results.py` must pass.
2. **Ground truth comes only from humans.** Two independent annotators label each record, Cohen's kappa
   is measured, and the supervisor adjudicates. The scripts support this workflow; never bypass it.
   `data/ground_truth/` is created only by `scripts/adjudicate.py` from real annotator exports.
3. **Evaluation is strict and end to end.** Each system's own predicted commitment goes through the rule
   engine. Never feed gold commitments to Layer 3 in an evaluation. Never use the regex fallback inside
   an evaluation (`--strict` stays on). Never label output as "Qwen + LoRA" unless the adapter actually ran.
4. **Do not change thesis-defined behaviour.** The tuple `c = <t, τ, v, T>`, the five verdicts (Table 3.1),
   Algorithm 1 (R5 → R4 → R1/R2/R3, then aggregation), the baselines and their hyper-parameters, the
   metrics and the statistical tests are fixed. The documented interpretations below are also fixed.
   Change them only if the user explicitly asks, and then update `docs/THESIS_TRACEABILITY.md`.
5. **Scope is PyPI only.** Do not add other ecosystems (npm, Maven, Go and so on) or non-PyPI repositories.
6. **Reproducibility:** seed 42 everywhere. Never use Python `hash()` for anything that affects results.
   Keep `docs/INTERFACES.md` and the code in sync.
7. **No secrets in files.** `GITHUB_TOKEN` and `NVD_API_KEY` live only in `.env` (git-ignored) or in
   environment variables. Never print, log, commit or zip them.
8. **Tests stay green.** Run `python -m pytest` (offline, about 3 minutes) after every change and add
   tests for new behaviour. Test fixtures are synthetic, live only in `tests/`, and are never copied
   into `data/`.
9. **Keep the project clean.** Do not create scratch files, experiment folders or copies of data inside the
   project. Do not commit caches, model weights or `.env`.
10. **Report honestly.** If something could not be run (no GPU, no network, missing annotations), say so.
    Do not claim results you did not produce.

## 3. Documented interpretations (the thesis is silent; these are fixed decisions)

- The "reasonable timeframe" for R3 is 90 days from publication (`REASONABLE_TIMEFRAME_DAYS`).
- An empty vulnerability list V, or no vulnerability inside the commitment scope, gives Insufficient Evidence.
- A "partial fix" means the vulnerability is fixed on some affected release lines only. A release line is
  the major version, or the major.minor line when the commitment names that minor line (e.g. "2.1.x",
  or a range bounded inside one major such as ">=2.1,<2.3").
- Aggregation: R3 with R1 or R2 gives Partially Consistent. R3 alone gives Potential Inconsistency, even
  when R4 records exist. Otherwise R2 gives Partially Consistent, R4 gives Insufficient Evidence, and
  otherwise the verdict is Consistent.
- Mapping uncertainty (R4) is set when the GitHub repository is a fork or archived. An NVD failure alone
  does not trigger R4 (NVD only enriches dates); OSV, GitHub Advisory or PyPI failures do.
- A bare `N.M` version means the whole N.M line; "latest" never resolves to a pre-release.
- Sampling strata: stars tercile × whether the package has at least one vulnerability record.
- LoRA: q/k/v/o projections, r = 128, alpha = 256 (about 1.06% trainable). The 4-bit base model fits a
  single T4 GPU.
- Layer 1 input is the Layer 0 preprocessed policy text (`policy_text` in the dataset records).
- Annotators also mark the temporal-reference phrase (`temporal_expression`) when the policy states one.

## 4. Project map (who owns what)

| Path | Content |
|---|---|
| `src/layer0_data_acquisition/` | PyPI metadata, GitHub repository verification (PyGithub), SECURITY.md retrieval, preprocessing, exclusion log |
| `src/layer1_ai_extraction/` | schema, Qwen extractor with JSON-constrained decoding, regex and RoBERTa baselines, LoRA and RoBERTa training, extraction metrics |
| `src/layer2_evidence_layer/` | OSV, GitHub Advisory, NVD and PyPI evidence, de-duplication, version normalisation |
| `src/layer3_rule_assessment/` | Algorithm 1 rule engine and the five verdicts |
| `src/layer4_explanation_output/` | Jinja2 explanations and JSON report storage |
| `src/data_pipeline/` | collection, sampling, annotation tasks, agreement, adjudication, splits |
| `src/evaluation/` | end-to-end evaluation, statistics, failure analysis, expert rubric, verification |
| `scripts/` | command-line entry points (one per workflow step) |
| `notebooks/` | Colab notebook for GPU training and evaluation |
| `data/` | seed list, collected dataset, annotations, ground truth, evaluation, reports, audit log (each folder has a README) |

## 5. Current state and next steps (the authoritative, ordered plan is NEXT_STEPS.md)

- **Code: complete.** All layers and workflow tools are implemented and tested.
- **Data collection is in progress.** Count the lines of `data/processed/pypi_dataset.jsonl` for the
  current total; the target is 300. Continue it with:
  `python scripts/collect_dataset.py --target 300`
  It is resumable: it skips everything already in `data/audit/collection_log.jsonl`, so never delete
  `data/processed` or `data/audit` to "start fresh". Only one machine may run the collection at a time.
- Then, in order:
  1. `python scripts/sample_dataset.py`
  2. `python scripts/prepare_annotation.py`
  3. Human annotation by two annotators.
  4. `python scripts/check_agreement.py`
  5. Supervisor adjudication: `python scripts/adjudicate.py --template`, then `python scripts/adjudicate.py`.
  6. `python scripts/make_splits.py`
  7. GPU training and evaluation with `notebooks/SupportLens_Colab_Train_and_Eval.ipynb`.
  8. `python scripts/run_expert_evaluation.py --prepare`, the expert ratings, then `--ingest`.
  9. `python scripts/render_results.py`, then `python scripts/verify_results.py`.
- Steps that need people or a GPU stop with a clear message until their inputs exist. That is
  intended behaviour; do not work around it.

## 6. Commands

```
python -m pip install -r requirements.txt     # core (CPU)
python -m pip install -r requirements-gpu.txt # training and neural evaluation (CUDA GPU)
python -m pytest                              # offline test suite
python supportlens_cli.py --pypi urllib3      # audit one package (regex fallback without a GPU, marked degraded)
streamlit run app.py                          # web interface
```

## 7. Style

- Code, docstrings and documentation in academic English. Short docstrings, no emojis.
- Answer the user in the language they write in (the user usually writes Egyptian Arabic).
