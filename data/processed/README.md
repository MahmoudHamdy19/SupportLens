# data/processed/

`pypi_dataset.jsonl`: one record per PyPI package that passed the Layer 0 gates of thesis
Sections 3.2.1 and 3.3.1 (verified GitHub repository, SECURITY.md present, version-support
statement present) and whose OSV and GitHub Advisory sources answered.

Fields: `package_name`, `github_repo`, `policy_text` (raw SECURITY.md, line endings normalised
to LF), `policy_path`, `policy_url`, `releases`, `vulnerabilities`, `source_diagnostics`,
`stars`, `repo_created_at`, `pushed_at`, `policy_word_count`, `collected_at` (UTC; the time of
the evidence snapshot).

Provenance: written only by `scripts/collect_dataset.py` from live PyPI, GitHub, OSV and GitHub
Advisory responses. GITHUB_TOKEN is read from the environment and never stored. The file holds
no labels.
