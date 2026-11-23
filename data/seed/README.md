# data/seed/

`pypi_seed_packages.txt`: one candidate PyPI package name per line, `#` starts a comment. The
collector tries these names first and then the public top-PyPI list. A name is only a
candidate; a package enters the dataset only when Layer 0 verifies its GitHub repository,
finds a SECURITY.md with a version-support statement, and the evidence sources answer.
