# data/annotations/

Human annotation of the stratified sample (thesis Section 3.3.1).

| File | Meaning | Produced by |
|---|---|---|
| `annotation_guidelines.md` | the codebook both annotators follow | written once |
| `label_studio_config.xml` | labelling interface: spans for t and v, choices for tau, T and the verdict | written once |
| `sample_300.jsonl`, `sample_300.meta.json` | the sample and its design (strata, seed 42, real size) | `scripts/sample_dataset.py` |
| `label_studio/tasks_*.json` | blind task files (shuffled per annotator, no model output, no pre-labels) | `scripts/prepare_annotation.py` |
| annotator exports | one JSON per annotator: a Label Studio export or the simple format described in `src/data_pipeline/annotation.py` | the annotators |
| `agreement_report.json` | Cohen's kappa (target 0.80), per-field agreement, confusion matrix, disagreements | `scripts/check_agreement.py` |
| `adjudication.json` | the supervisor's decision for every disagreement | the supervisor |

Rules: the two annotators work independently on the same tasks; neither sees a model prediction
or the other's labels; kappa is measured before adjudication; only the supervisor resolves
disagreements. The exports and the adjudication file are human work and are never generated.
`scripts/adjudicate.py --template` writes a blank decision sheet that lists the disagreements
and decides nothing.
