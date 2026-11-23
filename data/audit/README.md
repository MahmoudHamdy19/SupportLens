# data/audit/

Append-only logs, one JSON object per line.

* `exclusion_log.jsonl`: written by Layer 0 for every package that was rejected, with the reason
  (no PyPI package, no GitHub link, no SECURITY.md, no version-support statement, ...).
* `collection_log.jsonl`: written by the collector for every attempt. `outcome` is `INCLUDED`,
  `EXCLUDED`, `TRANSIENT` (network error, rate limit or a failed evidence source; retried by the
  next run) or `ERROR`.

A resumed collection skips names that were included or excluded for a definitive reason.
Do not edit these files.
