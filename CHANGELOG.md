# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic
Versioning once the first public release is published.

## [Unreleased]

Changes since 0.1.0rc1, developed under the internal v0.2–v0.4 milestones.
The package version remains 0.1.0rc1 until the next release candidate is
cut.

### Added

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

- Restored `ruff check` and `ruff format` compliance on `main` after the
  v0.4 changes landed, so the CI quality job is green again (#29).

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
