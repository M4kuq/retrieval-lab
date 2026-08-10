# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic
Versioning once the first public release is published.

## Unreleased

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

### Migration notes

- Result `schema_version` remains `1` during pre-release development. New
  runner-produced results include per-retriever `latency` and per-query search
  timing/warnings, plus runtime-only manifest metadata. Consumers should ignore
  unknown result fields and continue accepting older results without latency.
