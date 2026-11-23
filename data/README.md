# data/

Every file in this tree is either an input supplied by a person, a download from a public
source, or the output of a script in `scripts/`. Nothing is written by hand and no file may be
created from a heuristic, a random number or a model's own output (rules 1 and 2 of
`docs/INTERFACES.md`, section 6).

| Folder | Content | Produced by |
|---|---|---|
| `seed/` | optional list of extra candidate packages | a person |
| `cache/` | downloaded public lists and API responses | `scripts/collect_dataset.py`, Layer 2 |
| `audit/` | exclusion and collection logs (one JSON object per line) | Layer 0, `scripts/collect_dataset.py` |
| `processed/` | the collected PyPI dataset | `scripts/collect_dataset.py` |
| `annotations/` | codebook, Label Studio config and tasks, annotator exports, agreement report | see its README |
| `ground_truth/` | adjudicated ground truth and the stratified splits | `scripts/adjudicate.py`, `scripts/make_splits.py` |
| `evaluation/` | predictions, metrics, statistical tests, expert ratings, run manifest | `scripts/run_evaluation.py`, `scripts/run_expert_evaluation.py` |

Reproduce a number: `python scripts/verify_results.py` recomputes every headline figure from
`data/evaluation/predictions.jsonl` and `data/ground_truth/` and fails on any difference.
