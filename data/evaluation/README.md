# data/evaluation/

Written by `scripts/run_evaluation.py` in one step, from a real run on
`data/ground_truth/test.jsonl`. Nothing is written when the run fails; `--limit` smoke runs go to
`smoke_limit_N/` and cannot be verified.

* `predictions.jsonl`: every prediction with its extractor and model provenance
  (`extractor_engine`, `actual_model_used`, `lora_applied`, `execution_device`, `degraded`),
  the predicted commitment, the verdict, the rules fired, the distinct `reason_codes` of the
  Layer 3 evidence log, the record's `mapping_uncertain` flag and the explanation.
* `extraction_metrics.json`: token-level P/R/F1 for t and v, accuracy and macro-F1 for tau and T,
  tuple exact match and partial match (Q1).
* `verdict_metrics.json`: macro-F1, accuracy, per-class scores and the 5x5 confusion matrix (Q2).
* `statistical_tests.json` (thesis 3.3.4): bootstrap 95% CIs (1,000 resamples, seed 42, one set of
  resample indices shared by every metric and system) of the verdict macro-F1
  (`bootstrap_macro_f1`) and of the extraction scores t F1, v F1, tuple EM and tuple PM
  (`bootstrap_extraction`); McNemar's tests between the fine-tuned Qwen and each baseline (b, c,
  chi2, p) on per-record verdict correctness (`mcnemar`) and on per-record tuple exact match
  (`mcnemar_tuple_exact_match`); Mann-Whitney U tests of verdict correctness over project age,
  stars and policy length with a median split (`mann_whitney`) and between active and abandoned
  projects (`mann_whitney_activity`).
* `failure_analysis.json` (RQ3): per system, the (gold verdict, predicted verdict) pairs of the
  wrong verdicts; for the wrong and for the Insufficient Evidence predictions, how many records
  triggered each Layer 3 rule (R1 to R5) and each evidence-log reason code (each counted once per
  record); extraction errors grouped by the gold commitment type tau; verdict errors among
  active, abandoned and unclassified projects.
* `run_manifest.json`: software and hardware, seed, data hashes, hyper-parameters, run options.
* `expert/`: blind rating sheets and the sample manifest. `expert_evaluation.json`: ratings,
  mean and standard deviation per dimension and Krippendorff's alpha (interval), from
  `scripts/run_expert_evaluation.py --prepare` and `--ingest`.

Active and abandoned projects (Table 3.2): a project is **active** when its repository is not
archived and its last push (`pushed_at`) lies at most 365 days before the evidence collection
time (`collected_at`); **abandoned** when it is archived or its last push is older than that;
**unclassified** (left out of the comparison) when `collected_at` is missing, or `pushed_at` is
missing for a non-archived repository. A missing `archived` field counts as not archived.

The ground-truth field `mapping_uncertain` (written by the collector; False when absent or not
`true`) is passed to the rule engine, so an uncertain package-to-repository mapping triggers
rule R4 (Insufficient Evidence).

Check with `python scripts/verify_results.py`, which recomputes every number above (both kinds
of CI, both McNemar tests, both Mann-Whitney analyses and the failure analysis) from
`predictions.jsonl` and the test split. Tables in README.md and docs use markers such as
`<!-- RESULT:verdict_metrics.systems.qwen_finetuned_lora.macro_f1 -->value<!-- /RESULT -->`
(a dotted path into these files, optionally followed by `|format`), filled by
`scripts/render_results.py`. The top-level names are the file stems: `extraction_metrics`,
`verdict_metrics`, `statistical_tests`, `failure_analysis`, `expert_evaluation`, `run_manifest`,
`agreement_report` and `split_report`.
