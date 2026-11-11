# API reference

This page is the reference for the stable public API defined in
[API minimum](api-minimum.md). Every name listed here is importable from the
package root, except the six metric functions at the end, which are imported
from `retrieval_lab.metrics` and are not exported from the package root. The
list is pinned to `retrieval_lab.__all__` (plus `retrieval_lab.metrics.__all__`)
by the public-API tests, so it cannot drift from the exported surface.
Anything not listed here is internal implementation and may change without
notice.

## Domain records

Immutable typed records shared by every evaluation path.

- `Document` — A source document to index and evaluate.
- `Chunk` — A contiguous, zero-based slice of a parent document.
- `EvaluationQuery` — An evaluation query with explicit document and chunk relevance.
- `SearchResult` — One ranked chunk returned by a retriever.
- `QueryEvaluation` — Per-query ranking evidence and metric values at each cutoff.
- `RetrieverMetrics` — Macro-averaged metric values for one retriever.
- `EvaluationResult` — A deterministic evaluation result with aggregate and query evidence.
- `LatencyStats` — Immutable aggregate search-latency statistics in milliseconds.
- `ConstraintType` — Literal naming the supported quality-gate constraint kinds.

## Retrievers and chunking

Index-backed retrievers that share one chunk artifact per run.

- `BaseRetriever` — Contract implemented by synchronous, index-backed retrievers.
- `KeywordRetriever` — Rank chunks by distinct normalized query-term substring matches.
- `BM25Retriever` — Rank chunks with deterministic Okapi BM25 scores.
- `DenseRetriever` — Rank chunks by exact inner product over validated embedding vectors.
- `HybridRetriever` — Fuse rankings from two or more indexed retrievers with RRF.
- `EmbeddingBackend` — Typed embedding service used by DenseRetriever.
- `EmbeddingModelMetadata` — Identity and reproducibility metadata for an embedding backend.
- `FixedSizeChunker` — Split documents into deterministic, overlapping character chunks.

## Callable retriever adapters

Bring an existing search system, synchronously or asynchronously, without a
corpus or an index.

- `RetrievedItem` — One item in a provider or vector-database ranking.
- `Retriever` — Minimal synchronous retrieval protocol for external search systems.
- `CallableRetriever` — Adapt a synchronous callable to the provider-independent protocol.
- `evaluate_retrievers` — Evaluate synchronous external retrievers without a corpus or index.
- `AsyncRetriever` — Minimal asynchronous retrieval protocol for external search systems.
- `AsyncCallableRetriever` — Adapt an asynchronous callable to the provider-independent protocol.
- `evaluate_async_retrievers` — Evaluate asynchronous retrievers without a corpus or index.

## Datasets

- `EvaluationDataset` — An immutable set of queries and their graded relevance judgments.
- `RelevanceLevel` — Literal selecting document-level or chunk-level relevance.
- `validate_dataset` — Validate dataset-to-corpus identifiers at the explicit relevance level.
- `RetrievedQueryResult` — An immutable document ranking produced by an existing search system.
- `evaluate_results` — Evaluate existing document rankings using Retrieval Lab's core metrics.
- `load_documents` — Load supported local documents in deterministic source-path order.

## Hybrid fusion of precomputed rankings

- `fuse_rankings` — Fuse named identifier rankings with reciprocal-rank fusion.
- `fuse_results` — Fuse per-query rankings from two or more named sources.
- `evaluate_fused_results` — Evaluate the RRF fusion of precomputed rankings as its own run.

## Configuration

Typed records behind the schema-versioned YAML configuration.

- `RetrievalConfig` — Complete schema-versioned Retrieval Lab configuration.
- `ExperimentConfig` — Experiment identity and reproducibility settings.
- `CorpusConfig` — Corpus source, include filters, and chunking settings.
- `DatasetConfig` — Native JSONL evaluation dataset settings.
- `EvaluationConfig` — Metric cutoffs and v0.1 execution controls.
- `ReportConfig` — Report settings retained for future report execution.
- `ChunkerConfig` — Configuration for the deterministic character chunker.
- `BM25RetrieverConfig` — Configuration for dependency-free BM25 retrieval.
- `KeywordRetrieverConfig` — Configuration for the deterministic keyword retriever.
- `DenseRetrieverConfig` — Configuration for lazy optional exact dense retrieval.
- `HybridRetrieverConfig` — Configuration for reciprocal-rank fusion over named sources.
- `load_config` — Load and validate a schema-versioned YAML configuration.

## Runner and result loading

- `EvaluationRunner` — Evaluate one or more retrievers over shared documents and queries.
- `load_result` — Load and validate a UTF-8 result JSON file under a size limit.

## Comparison

- `ComparisonTolerance` — Absolute and relative tolerances used for equality and classifications.
- `ComparabilityIssue` — One blocking or diagnostic difference between two result manifests.
- `ComparabilityReport` — Complete comparability diagnostics, including non-blocking run changes.
- `MetricDelta` — Aggregate or query-level metric change from baseline to candidate.
- `MetricComparison` — Aggregate metric delta and deterministic per-query evidence.
- `QueryDeltaExtreme` — Best or worst query-level change for one metric.
- `RunComparison` — Deterministic comparison for the common retrievers in two runs.
- `BootstrapDelta` — Bootstrap confidence interval and significance for one metric delta.
- `check_comparability` — Diagnose whether two results satisfy all strict comparison contracts.
- `compare_runs` — Compare two compatible results and return deterministic metric deltas.
- `bootstrap_metric_delta` — Resample per-query deltas for a confidence interval and p-value.

## Quality gates

- `QualityGateConfig` — A validated but not-yet-executed quality gate.
- `QualityGateCheck` — One evaluated constraint for one retriever metric.
- `QualityGateResult` — All constraints evaluated for one configured gate.
- `QualityGateReport` — Deterministic results for every configured quality gate.
- `evaluate_quality_gates` — Evaluate all configured absolute and baseline-relative constraints.

## Application services

The typed services behind the CLI; they never print.

- `InitializedProject` — Files created by initialize_project.
- `ValidationResult` — Validated configuration and the input counts discovered by the runner.
- `ExperimentOutput` — An evaluation result and the report files written for it.
- `InspectionOutput` — Loaded result metadata and optional per-query evidence.
- `QueryEvidence` — Safe, deterministic evidence for one retriever/query pair.
- `ComparisonOutput` — Comparable saved runs and deterministic aggregate metric rows.
- `ComparisonRow` — One aggregate metric delta prepared for deterministic display.
- `GateOutput` — Configured quality-gate evaluation and its pass/fail state.
- `initialize_project` — Create a safe, offline, runnable Retrieval Lab project template.
- `validate_config_inputs` — Validate configuration, corpus, dataset, and gold IDs without searching.
- `run_configured_experiment` — Run a configured experiment and persist the selected report formats.
- `inspect_result` — Load a saved result and prepare deterministic inspection data.
- `compare_result_files` — Load and compare two saved results through the strict public APIs.
- `evaluate_configured_quality_gates` — Evaluate configured gates from a file or a candidate result manifest.

## Answer evaluation

Deterministic, rule-based scoring of generated answers; see also the
answer-evaluation guide.

- `AnswerEvaluation` — Rule-based answer-quality measurements for one generated answer.
- `AnswerEvaluationResult` — Aggregate answer-evaluation result with per-query evidence.
- `AnswerFailureType` — Literal classifying an answer outcome: success, retrieval failure, generation failure, citation failure, or unscored.
- `AnswerReference` — Ground-truth information used by deterministic answer evaluation.
- `evaluate_answer` — Evaluate one generated answer with deterministic rules and citations.
- `evaluate_answers` — Evaluate a batch while preserving input order and query identity.
- `exact_match` — Return normalized Exact Match as 0.0 or 1.0.
- `token_f1` — Return multiset Token F1 using deterministic multilingual tokenization.

## Answer generation

Caller-controlled generators; the library ships no provider client.

- `Generator` — Minimal answer-generator contract; providers remain caller-controlled.
- `GenerateCallable` — Callable shape accepted for one generation request.
- `CallableGenerator` — Adapt an arbitrary synchronous callable to the public Generator contract.
- `MockGenerator` — Deterministic offline generator for tests, examples, and pipelines.
- `GenerationRequest` — A query plus an ordered context list for answer generation.
- `GenerationContext` — One retrieved context item supplied to an answer generator.
- `GeneratedAnswer` — An answer and explicit document citations returned by a generator.
- `GenerationOutput` — A validated generated answer plus observed generation latency.
- `generate_answer` — Generate one answer and measure latency using a caller-controllable clock.

## Evaluation checkpoints

Crash-safe resume for large evaluations: each query's ranking is
appended (and fsynced) to a JSONL checkpoint pinned to a
caller-supplied run fingerprint, and a resumed run executes only the
queries still missing.

- `EvaluationCheckpoint` — An append-only JSONL store of per-query rankings for one run.
- `run_with_checkpoint` — Run the queries missing from checkpoint and return all rankings.

## Benchmark loaders

Offline loaders that convert standard benchmark layouts into native
records; downloading stays with the caller.

- `BenchmarkDataset` — A loaded benchmark: its corpus documents and evaluation dataset.
- `load_beir_dataset` — Load a BEIR-format dataset directory from disk.
- `load_msmarco_dataset` — Load an MS MARCO passage-ranking dataset directory from disk.
- `load_natural_questions_dataset` — Load a simplified Natural Questions JSONL file from disk.

## RAG quality metrics

RAGAS-style formulas computed from structured inputs — which context
items were relevant and which answer claims the context supports — so
every score is reproducible offline; claim extraction and support
judgments stay caller-controlled.

- `faithfulness` — Return the fraction of assessed claims supported by the context.
- `context_precision` — Return rank-weighted context precision over the full context list.
- `context_recall` — Return the fraction of relevant items present in the context list.
- `ClaimAssessment` — One answer claim and whether the retrieved context supports it.
- `RagEvaluation` — RAG quality measurements for one answered query.
- `evaluate_rag_answer` — Evaluate one RAG answer's context quality and claim faithfulness.

## LLM-as-judge

Caller-controlled judges in the same style as answer generators: the
library renders the prompt, measures latency, and records verdicts and
reported usage; it ships no provider client and never parses free-text
judge output.

- `Judge` — Minimal judge contract; providers remain caller-controlled.
- `JudgeCallable` — Callable shape accepted for one judge request and its rendered prompt.
- `CallableJudge` — Adapt an arbitrary synchronous callable to the public Judge contract.
- `MockJudge` — Deterministic offline judge for tests, examples, and pipelines.
- `JudgePromptTemplate` — A judge prompt template with a closed set of placeholders.
- `DEFAULT_JUDGE_PROMPT_TEMPLATE` — A judge prompt template with a closed set of placeholders.
- `JudgeRequest` — One generated answer awaiting judgment, with its evidence.
- `JudgeVerdict` — One judge's validated score for one answer.
- `JudgeUsage` — Token and cost usage reported by one judge invocation.
- `JudgeOutput` — A validated verdict plus the prompt and latency that produced it.
- `PanelJudgment` — Verdicts from several judges for one answer, with panel totals.
- `judge_answer` — Judge one answer and measure latency using a caller-controllable clock.
- `judge_answer_with_panel` — Judge one answer with several judges and aggregate their verdicts.

## Dataset authoring, inspection, and workflow

Drafts stay unevaluable until they are finalized through human review.

- `DatasetDraft` — Immutable authoring state that cannot be evaluated until finalized.
- `DraftQuery` — One editable query whose relevance judgments may still be incomplete.
- `DatasetOrigin` — Literal naming a dataset origin: unknown, human, synthetic, or mixed.
- `DatasetProvenance` — Dataset-level origin and human-review state.
- `DatasetReviewStatus` — Literal naming the human-review state recorded in a dataset's provenance.
- `DatasetInspectionCode` — Literal naming the deterministic dataset-inspection finding kinds.
- `DatasetInspectionIssue` — One deterministic dataset-quality finding that requires human review.
- `DatasetInspectionReport` — Summary of non-destructive quality diagnostics for one dataset.
- `inspect_dataset` — Return conservative, deterministic candidates for human dataset review.
- `DatasetDraftStatus` — Safe, deterministic status for a persisted dataset draft.
- `dataset_draft_status` — Load a bundle and return authoring/review status.
- `load_dataset_draft` — Load and validate a bundle produced by DatasetDraft.save_bundle.
- `review_dataset_query` — Replace one query's judgments, validate them against the corpus, and save.
- `finalize_dataset_bundle` — Validate that a persisted draft is ready for EvaluationDataset loading.

## Synthetic query drafts

- `SyntheticQueryGenerator` — Contract for a caller-controlled synthetic query generator.
- `SyntheticGenerateCallable` — Callable shape accepted for synthetic draft generation.
- `CallableSyntheticGenerator` — Adapt a synchronous Python callable to SyntheticQueryGenerator.
- `generate_synthetic_draft` — Generate an explicitly experimental, unreviewed document-level draft.
- `mark_draft_in_review` — Mark a dataset draft as undergoing human review.
- `mark_draft_reviewed` — Mark human review complete without changing the dataset's origin.

## Demo comparison

- `DemoComparison` — Side-by-side retrieval results for one query.
- `DemoRetrieverView` — Results and observed search latency for one retriever.
- `DemoSearchHit` — One presentation-safe search hit returned by a demo comparison.
- `compare_retrievers_for_query` — Search already-indexed retrievers under one shared query and cutoff.
- `retrieval_metric_explanations` — Return concise explanations suitable for notebooks and demo UIs.

## Exceptions

Every public boundary raises these typed errors; they are also importable
from the exceptions module, and the errors module is a compatibility alias.

- `RetrievalLabError` — Base class for errors raised by Retrieval Lab public APIs.
- `ConfigurationError` — Raised when an evaluation configuration is invalid.
- `CorpusValidationError` — Raised when a document or chunk violates the corpus contract.
- `DatasetValidationError` — Raised when an evaluation dataset violates its public contract.
- `EvaluationError` — Raised when an evaluation cannot produce a valid result.
- `IncomparableRunError` — Raised when two evaluation runs cannot be compared safely.
- `OptionalDependencyError` — Raised when an optional Retrieval Lab feature is not installed.
- `RetrieverContractError` — Raised when a retriever result violates the ranking contract.

## Metric functions

Pure per-query metric primitives. Import them from `retrieval_lab.metrics`;
they are not exported from the package root.

- `average_precision_at_k` — Return AP@k with denominator min(number of relevant items, k).
- `hit_rate_at_k` — Return 1.0 when a relevant identifier occurs in the first k results.
- `ndcg_at_k` — Return graded nDCG@k using gain 2**relevance - 1.
- `precision_at_k` — Return relevant results divided by k; missing positions are misses.
- `recall_at_k` — Return the fraction of relevant identifiers retrieved in the first k.
- `reciprocal_rank` — Return the reciprocal rank of the first relevant result, optionally at k.
