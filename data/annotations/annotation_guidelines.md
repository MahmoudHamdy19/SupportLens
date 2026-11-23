# SupportLens annotation guidelines (codebook)

Version 1.0. Authority: thesis Sections 3.1.3, 3.2.4 (Table 3.1) and 3.3.1.

Two annotators label every sampled policy independently. They record the commitment tuple
`c = <t, tau, v, T>` and the expected verdict `y`. A third party (the thesis supervisor)
resolves every disagreement afterwards. Cohen's kappa on the verdict is measured before
adjudication (target kappa >= 0.80).

## 1. Ground rules

1. Work alone. Do not discuss individual policies or labels with the other annotator until
   both exports are handed in. Questions about the codebook go to the supervisor, who answers
   both annotators with the same text.
2. Annotate only what the policy and the evidence panel show. Do not look at model output,
   at the other annotator's labels or at any pre-label (the task files contain none).
3. Do not use external knowledge of the project beyond what is needed to read the evidence.
4. The Label Studio project must use `label_studio_config.xml` unchanged.
5. The commitment text and version expression are selected in the policy text, never typed:
   they must be verbatim substrings of the policy.

## 2. The commitment tuple

The study is restricted to the version-support statement of the policy. Support commitments
about response times, remediation SLAs or end-of-life dates are not annotated.

### 2.1 Commitment text `t` (label `CommitmentText`)

* Select the sentence or sentences that state which versions receive security fixes.
* Select ONE contiguous span, from the first to the last word of the statement. If a table
  or list carries the statement, select it including its header line.
* Include only text that is needed to read the commitment. Do not select the whole file.
* If the policy contains several statements, select the one that defines the currently
  supported versions. If it contains none that can be selected (for example the version
  heading is the only content), leave `t` empty and choose `ambiguous` for tau.

### 2.2 Commitment type `tau` (choice `support_status`)

| Value | Use when | Example |
|---|---|---|
| `latest_only` | only the newest release is supported | "Only the latest release receives security fixes." |
| `explicit_range` | named versions, branches or a version range are supported | "Supported: 2.1.x and 2.0.x"; "versions >= 1.4" |
| `all_versions` | every release is supported | "All versions of this package are supported." |
| `ambiguous` | support is mentioned but cannot be mapped to concrete versions | "We support recent versions when possible." |

Rules:
* A table that marks some versions as supported and others as unsupported is `explicit_range`.
* "Latest minor of each supported major" is `explicit_range` when the majors are named,
  otherwise `ambiguous`.
* Choose `ambiguous` when you cannot write down, from the text alone, which released
  versions are covered.

### 2.3 Version expression `v` (label `VersionExpression`)

* Select each version token or range exactly as written: `2.1.x`, `>= 1.4`, `3.0 - 3.2`,
  `v4`, `latest`.
* One span per token or range. Do not merge separate tokens into one span.
* For `latest_only` or `all_versions` select the phrase that expresses it when the policy
  contains one ("latest", "most recent release", "all versions"). When the policy contains no
  such phrase, select nothing.
* For `explicit_range` at least one token is expected. If the policy names versions only in
  prose that cannot be selected as a token, choose `ambiguous`.
* Tokens that describe unsupported versions (for example a table row marked unsupported) are
  not selected.

### 2.4 Temporal reference `T` (choice `temporal_reference`)

The point in time relative to which support is evaluated.

* `vulnerability_publication_time` (default): support is judged when a vulnerability is
  published. Use it when the policy names versions or says "latest" without a time rule.
* `release_time`: use it when the policy ties support to the release of a version, for
  example "each minor release is supported for 18 months after it is released".

### 2.5 Temporal phrase (label `TemporalExpression`, optional)

* Select the words of the policy that state the temporal reference, for example "the most
  recent release", "at the time of disclosure", "currently", "after it is released".
* At most one span. Leave it unmarked when the policy states no temporal phrase; the choice
  `temporal_reference` is still required.
* The span must be selected in the text, never typed. It is stored as
  `commitment.temporal_expression` (verbatim text, or null).

## 3. Expected verdict `y` (Table 3.1)

Decide `y` by reading the commitment against the evidence panel (releases and vulnerability
records). Apply the questions in this order and stop at the first that applies. The order
follows rules R5, R4, R1, R2 and R3 of Algorithm 1; you judge by reading, you do not run
any software.

1. **Unmeasurable Commitment.** Is tau `ambiguous`, or can the versions not be turned into
   concrete version identifiers? Then the verdict is Unmeasurable Commitment.
2. **Insufficient Evidence.** Is the evidence incomplete or ambiguous?
   * an evidence source is marked FAILURE or RATE_LIMITED, or the release list is empty;
   * the evidence panel lists no vulnerability record at all (there is nothing to test the
     commitment against);
   * none of the listed vulnerabilities affects a version that the commitment covers;
   * release or publication dates that are needed for your judgement are missing.
3. Otherwise there is at least one vulnerability that affects a version covered by the
   commitment. Judge each such vulnerability:
   * **fixed** if a fixed release exists for the covered version line (same major version);
   * **partly fixed** if a fix exists only on some covered release lines;
   * **unfixed beyond the timeframe** (rule R3) if no fix exists although it was published
     more than 90 days before the evidence date (the "reasonable timeframe"; the supervisor
     may change this number before annotation starts and must then tell both annotators);
   * **undecided** (rule R4) if it is unfixed but younger than the timeframe, or a date or
     version mapping you need is missing.
   Then aggregate by highest severity, applying these checks in this exact order and
   stopping at the first that applies:
   1. no covered vulnerability could be judged at all: **Insufficient Evidence**;
   2. at least one vulnerability is unfixed beyond the timeframe and at least one other is
      fixed or partly fixed: **Partially Consistent**;
   3. at least one vulnerability is unfixed beyond the timeframe (even when others are
      undecided): **Potential Inconsistency**;
   4. at least one vulnerability is partly fixed: **Partially Consistent**;
   5. at least one vulnerability is undecided: **Insufficient Evidence**;
   6. otherwise every covered vulnerability is fixed: **Consistent**.
   A confirmed unfixed vulnerability is never hidden by undecided ones.

Write a short justification in the notes field whenever the choice was not obvious. Notes are
read by the adjudicator only.

## 4. Edge cases

* Empty or non-English policy: if the version-support statement cannot be read, tau is
  `ambiguous`, t is empty and the verdict is Unmeasurable Commitment.
* A policy that only links to a web page for supported versions is `ambiguous`.
* A vulnerability without affected versions cannot be tested: ignore it for coverage and
  mention it in the notes.
* Do not guess missing dates. If a date you need is absent, use Insufficient Evidence.

## 5. Files and procedure

1. The supervisor imports `tasks_annotator1.json` and `tasks_annotator2.json` into two
   separate Label Studio projects with `label_studio_config.xml`.
2. Each annotator labels every task and exports the project as JSON (Label Studio export).
3. `scripts/check_agreement.py` computes kappa and lists the disagreements.
4. The supervisor decides each disagreement in the adjudication file
   (`scripts/adjudicate.py --template` writes a blank sheet).
5. `scripts/adjudicate.py` builds `data/ground_truth/ground_truth.jsonl`.

An annotator who works outside Label Studio may deliver the simple JSON format documented in
`src/data_pipeline/annotation.py`. The same validation applies.
