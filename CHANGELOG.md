# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic
Versioning once the first public release is published.

## [Unreleased]

Changes since 0.1.0rc1, developed under the internal v0.2–v0.4 milestones.
The package version remains 0.1.0rc1 until the next release candidate is
cut.

### Removed

- The `request <= 0` guard in the cache artifact `_read_bytes` loop,
  which was unreachable. Its only caller passes a limit validated as a
  positive integer, and the loop raises as soon as the accumulated data
  exceeds that limit, so every iteration starts with
  `len(data) <= limit` and the computed request is always at least 1;
  the loop now asserts that invariant. `_cache_max_bytes` also
  normalizes the limit to a plain `int` (real-type check plus
  `int.__int__` extraction, before the range check), so an `int`
  subclass overriding arithmetic, comparison or `__int__` — or an
  object spoofing `__class__` — cannot break the invariant. Observable
  behaviour is unchanged for plain integers, and the cache tests pin
  the limit cases: growth after `fstat`, short reads, artifacts over
  the byte limit, and an `int`-subclass limit.
- The `relative_delta is None` guard in `_relative_regression`, which
  was unreachable through the public quality-gate APIs. Those APIs
  build their deltas only via `_metric_delta`, whose `_relative`
  helper pairs `None` exclusively with the `baseline_zero` status
  (handled by the preceding branch) and always supplies a float for
  the `defined` status, so the guard could never fire; the branch now
  asserts that invariant. Observable behaviour is unchanged, and the
  existing quality-gate tests pin the defined-status regressed cases
  for both directions, a defined-status improvement, and the
  baseline-zero cases.
- Two unreachable `None` guards in the quality-gate drop-constraint
  blocks (`max_absolute_drop` / `max_relative_drop`). The earlier
  `has_drop` check already requires a baseline, and a gate carrying a
  drop constraint always receives a delta from `_comparison_delta`
  (which returns or raises), so neither value can be `None` there;
  the blocks now assert the same invariant the function already
  asserts when building the comparison. Observable behaviour is
  unchanged.
- Three more unreachable empty guards: the empty-result guard in
  `EvaluationDataset.from_jsonl` (the input text is already checked
  non-empty and every line either raises or appends a query), and the
  `str(Path)` emptiness guards for `cache_dir` in the cache root helper
  and `EvaluationRunner` (blank strings are already rejected and
  `pathlib` normalises an empty path to `.`, so the constructed path
  is never empty under an unmodified `pathlib`). Observable behaviour
  is unchanged.
- Two unreachable empty-result guards in the corpus loaders
  (`load_documents` and `_load_jsonl`). File discovery already raises
  unless at least one supported file is selected, and each loader
  returns at least one document or raises, so the guards could never
  fire; observable behaviour is unchanged.

### Added

- Tests pinning relevance-grade ID validation in `EvaluationDataset`:
  empty, blank, or non-string grade IDs fail the preceding
  exact-match check, while a string-equal non-`str` key (such as
  `collections.UserString`) passes that check and is rejected by the
  non-empty guard with a `DatasetValidationError`. The guard only
  looked unreachable; the new test keeps it covered.
- Tests pinning the config loader's `corpus.include` handling: an
  explicit YAML null is treated exactly like an omitted `include`
  (an empty pattern tuple), matching the loader's existing branch.
- Tests pinning reachable error paths in metadata validation and the
  dataset bundle workflow: a non-string key in nested `Document`
  metadata raises `CorpusValidationError`, and an `OSError` while
  inspecting a dataset bundle directory or reading its manifest is
  converted to `DatasetValidationError`.
- Tests pinning validation paths across shipped modules:
  `KeywordRetriever.index_size_bytes` follows the index lifecycle
  (None before indexing, 0 for an empty index, a deterministic
  positive size afterwards); `load_config` reports an unknown
  built-in retriever type together with its hint;
  `context_precision` / `context_recall` reject a non-collection
  `relevant_ids`; and `load_dataset_draft` rejects a dataset
  manifest larger than the 1 MiB safety limit before parsing it.
- Tests pinning `evaluate_quality_gates` reference and regression
  paths: gates referencing a missing cutoff, a missing metric at an
  existing cutoff, a candidate-only retriever (drop gates), or a
  latency metric the baseline lacks all raise `ConfigurationError`;
  a latency absolute regression beyond its threshold fails with the
  hand-calculated regression amount; and a drop that stays within
  the comparison tolerance counts as zero regression.
- Tests pinning `inspect_dataset` argument validation and skip paths:
  non-dataset input and invalid integer limits are rejected, the
  relevance-concentration check is skipped below the minimum query
  count while the maximum share is still reported, and the verbatim
  check is skipped for chunk-level datasets supplied without chunks
  and for queries shorter than the configured minimum length.
- Tests pinning `BootstrapDelta` field validation (non-empty names,
  positive-or-None cutoff, positive resamples, boolean significance)
  and the bootstrap percentile path when a quantile position lands
  exactly on a resample mean.
- Python 3.14 support: the full test suite, lint, and type checks pass
  on CPython 3.14, the CI matrix covers it on all three operating
  systems, and the package classifiers declare it. Packaging tests pin
  the CI matrix, the declared classifiers, and the technical design
  document's version list to the same set so they cannot drift apart.
- Chunk-level precomputed evaluation: `evaluate_results()` now accepts a
  chunk-level `EvaluationDataset` and matches retrieved chunk IDs directly
  against its chunk-level grades, so rankings produced by an external
  chunk search system can be scored without the built-in retrievers. The
  dataset's relevance level is recorded in the manifest and run ID;
  document-level run IDs are unchanged. Fused evaluation through
  `evaluate_fused_results()` works at chunk level the same way and is
  now pinned by hand-calculated tests.
- Executable example: `examples/chunk_rankings.py` loads a chunk-level
  dataset (`examples/chunked-qrels.jsonl`), scores two external
  systems' chunk rankings with `evaluate_results()`, and fuses them
  with `evaluate_fused_results()`; the example runs as an integration
  test so it cannot drift from the public API.
- Executable example: `examples/standard_loaders.py` loads miniature
  MS MARCO (`examples/msmarco-mini/`) and Natural Questions
  (`examples/nq-mini.jsonl`) fixtures with `load_msmarco_dataset()`
  and `load_natural_questions_dataset()` and evaluates both.
- Executable example: `examples/comparison_statistics.py` compares two
  precomputed runs with `compare_runs()`, attaches a bootstrap
  confidence interval to the recall@1 delta with
  `bootstrap_metric_delta()`, and saves the standalone HTML diff report.
- Executable example: `examples/checkpoint_resume.py` simulates a crash
  during a checkpointed evaluation and resumes with
  `run_with_checkpoint()`, running only the queries missing from the
  checkpoint before scoring the collected rankings.
- Executable example: `examples/benchmark_loaders.py` loads a miniature
  BEIR-format dataset (`examples/beir-mini/`) with `load_beir_dataset()`,
  evaluates the keyword and BM25 strategies on it, and fuses the two
  rankings with `evaluate_fused_results()`; the example runs as an
  integration test so it cannot drift from the public API.
- Executable example: `examples/dataset_authoring.py` walks the v0.2
  dataset-authoring workflow end to end: deterministic synthetic
  generation behind the provider boundary, bundle persistence, the
  persisted review workflow (review completion never rewrites the
  synthetic origin), finalization, dataset inspection, and evaluation;
  the example runs as an integration test so it cannot drift from the
  public API.
- Documentation: a dataset authoring guide (`docs/dataset-authoring.md`,
  in the site navigation and the sdist) covering the draft lifecycle,
  provenance and reliability semantics, the persisted review workflow
  and its CLI, the inspection diagnostics, and the synthetic
  generation trust rules.
- CLI `--version`: `retrieval-lab --version` prints the installed
  package version and exits without running an evaluation.
- Run diff report: `RunComparison.to_html()` / `save_html(path)` (and
  `comparison_html()` in `retrieval_lab.reporting`) render a standalone
  escaped HTML report of a two-run comparison, with comparability
  differences, aggregate metric changes, and per-query metric changes
  classified as improved, regressed, or unchanged.
- Evaluation checkpoints: `EvaluationCheckpoint` records each query's
  ranking to a fingerprint-pinned JSONL file (flushed and fsynced per
  record, torn final lines from a crash are dropped on reopen), and
  `run_with_checkpoint()` resumes a large evaluation by running only
  the queries still missing.
- Benchmark loader: `load_beir_dataset()` reads a BEIR-format
  directory (`corpus.jsonl`, `queries.jsonl`, `qrels/<split>.tsv`)
  from disk into a `BenchmarkDataset` of native documents and an
  `EvaluationDataset` with graded relevance; queries without positive
  judgments in the split are reported as `excluded_query_ids`.
- Benchmark loaders: `load_msmarco_dataset()` reads an MS MARCO
  passage-ranking directory (`collection.tsv`, `queries.<split>.tsv`,
  `qrels.<split>.tsv` in TREC format), and
  `load_natural_questions_dataset()` reads a simplified Natural
  Questions JSONL file, turning each example's page into a corpus
  document and its long-answer annotation into a grade-1 judgment;
  unjudged queries are reported as `excluded_query_ids`.
- RAG quality metrics: `faithfulness()`, `context_precision()`, and
  `context_recall()` compute the RAGAS-style formulas from
  caller-supplied claim assessments (`ClaimAssessment`) and relevance
  judgments, and `evaluate_rag_answer()` bundles them into a
  `RagEvaluation` record per query.
- LLM-as-judge support: `judge_answer()` and `judge_answer_with_panel()`
  run caller-controlled judges over generated answers with a validated
  prompt template (`JudgePromptTemplate`, `DEFAULT_JUDGE_PROMPT_TEMPLATE`),
  multi-judge panels (`PanelJudgment` with mean/spread and token/cost
  totals), and per-judge usage recording (`JudgeUsage`). No provider
  client is bundled; judges return typed `JudgeVerdict` records.
- Comparison statistics: `bootstrap_metric_delta()` resamples the paired
  per-query deltas of a `MetricComparison` and returns a `BootstrapDelta`
  with a percentile bootstrap confidence interval, a two-sided bootstrap
  p-value, and a significance flag, all deterministic under a caller seed.
- Dataset inspection foundation: `inspect_dataset()` validates an evaluation
  dataset against a corpus and returns a typed `DatasetInspectionReport` with
  stable issue codes (v0.2).
- Dataset authoring models with explicit provenance: `DatasetDraft`,
  `DraftQuery`, and `DatasetProvenance` track origin and review status for
  datasets that are still being prepared (v0.2).
- Provider-independent synthetic dataset generation boundary:
  `generate_synthetic_draft()` produces explicitly experimental, unreviewed
  drafts from a caller-supplied generator, with `mark_draft_in_review()` and
  `mark_draft_reviewed()` driving the human review states (v0.2).
- Dataset review workflow: draft persistence, per-query review decisions,
  draft status reporting, and bundle finalization, exposed through the public
  API and thin `dataset` CLI commands (v0.2).
- Reusable public demo API: `compare_retrievers_for_query()` and
  `retrieval_metric_explanations()` provide typed, presentation-ready demo
  results, with a Streamlit example built on top of them (v0.3).
- Deterministic answer evaluation: `evaluate_answer()` and
  `evaluate_answers()` score generated answers with `exact_match` and
  `token_f1`, and a provider-independent generation boundary
  (`CallableGenerator`) supplies answers without provider dependencies (v0.4).

### Fixed

- Removed an unreachable empty-path check in the application layer and
  made `DenseRetriever.search` fail closed with a
  `RetrieverContractError` when an indexed state has no recorded
  dimension, instead of silently returning no results.
- Restored `ruff check` and `ruff format` compliance on `main` after the
  v0.4 changes landed, so the CI quality job is green again (#29).
- Chunk-level precomputed and fused runs are comparable again:
  `evaluate_results()` and `evaluate_fused_results()` now record a
  `chunk_hash` in the manifest — a caller-supplied `chunk_hash` value, or
  a deterministic hash of the dataset's chunk ID set when omitted — so
  chunk-level runs satisfy the strict comparability contract instead of
  always raising `IncomparableRunError`. A derived hash certifies the
  dataset's chunk ID set only, not a chunking configuration.
  Document-level run IDs are unchanged.

## [0.1.0rc1] - 2026-08-10

### Added

- Initial product, API, evaluation, and architecture specifications.
- Packaging, quality-tool, and continuous-integration foundations.
- First deterministic keyword-evaluation vertical slice.
- Strict TXT, Markdown, corpus JSONL, and graded evaluation JSONL loading.
- Dataset-to-corpus validation for document and chunk relevance.
- Evaluation of precomputed document rankings through the public Python API.
- Dependency-free deterministic BM25 with replaceable tokenization.
- Optional exact dense retrieval with typed embedding backends, lazy model loading,
  and reproducible model settings.
- Shared six-metric evaluation engine and canonical dataset hashing.
- Search and build latency statistics, reproducibility runtime metadata, and a
  validated default evaluation seed.
- Immutable typed configuration models, strict schema-versioned YAML loading,
  and `EvaluationRunner.from_config()` with safe relative-path resolution.
- Synchronous `CallableRetriever`, `RetrievedItem`, and corpus-free
  `evaluate_retrievers()` support for existing search APIs and vector databases.
- Provider-independent asynchronous retrieval adapters and corpus-free
  `evaluate_async_retrievers()` with bounded concurrency and cancellation-safe
  execution.
- Safe schema-versioned result reload, atomic JSON/CSV/HTML persistence, and
  standalone dependency-free reports with formula and HTML injection protection.
- Deterministic typed comparison of saved runs with complete comparability
  diagnostics and aggregate/query-level metric deltas.
- Typed absolute and baseline-relative quality-gate evaluation with immutable
  result attachment and validated JSON round-tripping.
- Thin standard-library CLI and typed application services for safe project
  initialization, input validation, configured runs, and report persistence.
- Added thin `inspect`, `compare`, and `gate` CLI commands with deterministic
  text/strict JSON output, query evidence inspection, and stable exit codes.
- Added offline saved-run regression fixtures, a public GitHub Actions gate
  example, active CI gate assertions, and wheel/optional-install smoke checks.
- Added an offline Japanese tutorial corpus, executable Python examples, FAQ,
  and a reproducible BM25 notebook with deterministic validation.
- Added a deterministic cold/warm local benchmark harness; synthetic benchmark
  measurements are validation fixtures, not claims about production performance.
- Added release-candidate package metadata, a strict MkDocs site, and offline
  wheel/sdist validation workflows.

### Migration notes

- Result `schema_version` remains `1` during pre-release development. New
  runner-produced results include per-retriever `latency` and per-query search
  timing/warnings, plus runtime-only manifest metadata. Consumers should ignore
  unknown result fields and continue accepting older results without latency.
