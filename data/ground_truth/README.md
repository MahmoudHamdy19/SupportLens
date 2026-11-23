# data/ground_truth/

* `ground_truth.jsonl`: one record per sampled package: the final commitment tuple
  `c = <t, tau, v, T>`, the final verdict, the evidence snapshot and, under `annotation`, both
  annotators' labels and the adjudicator (`adjudicated_by` is `null` when they agreed).
* `ground_truth_meta.json`: counts and SHA-256 hashes of the inputs.
* `train.jsonl`, `val.jsonl`, `test.jsonl`: stratified 70/15/15 split on the verdict, seed 42.
* `split_report.json`: realised class counts per split and how small classes were handled.

Provenance: produced only by `scripts/adjudicate.py` (from the two annotator exports and the
supervisor's adjudication file) and `scripts/make_splits.py`. `adjudicate.py` refuses to write
while a record is missing or a disagreement is unresolved. No label here comes from a heuristic,
a random number or a model. The test set is used only by `scripts/run_evaluation.py`.
