# Minimum public API for the first vertical slice

Status: accepted and extended through the data-input, precomputed-evaluation, and
BM25 milestone.

This document fixes the smallest useful public API before implementation. The
interface can grow during v0.1, but existing behavior must not be moved behind a
private module without a deprecation path.

## Public imports

The following names are importable from `retrieval_lab`:

- `Document`
- `Chunk`
- `EvaluationQuery`
- `SearchResult`
- `QueryEvaluation`
- `RetrieverMetrics`
- `EvaluationResult`
- `BaseRetriever`
- `KeywordRetriever`
- `BM25Retriever`
- `DenseRetriever`
- `HybridRetriever`
- `RetrievedItem`, `Retriever`, `CallableRetriever`, `evaluate_retrievers`
- `AsyncRetriever`, `AsyncCallableRetriever`, `evaluate_async_retrievers`
- `EmbeddingBackend`
- `EmbeddingModelMetadata`
- `OptionalDependencyError`
- `FixedSizeChunker`
- `EvaluationDataset`
- `RelevanceLevel`
- `RetrievedQueryResult`
- `EvaluationRunner`
- `RetrievalConfig` and its typed nested configuration records
- `load_config`
- `load_result`
- `ComparisonTolerance`, `ComparabilityIssue`, `ComparabilityReport`
- `MetricDelta`, `QueryDeltaExtreme`, `MetricComparison`, `RunComparison`
- `check_comparability`, `compare_runs`
- `QualityGateConfig`, `QualityGateCheck`, `QualityGateResult`, `QualityGateReport`
- `evaluate_quality_gates`
- `InitializedProject`, `ValidationResult`, `ExperimentOutput`
- `initialize_project`, `validate_config_inputs`, `run_configured_experiment`
- `InspectionOutput`, `QueryEvidence`, `ComparisonOutput`, `ComparisonRow`,
  `GateOutput`
- `inspect_result`, `compare_result_files`, `evaluate_configured_quality_gates`
- `load_documents`
- `validate_dataset`
- `evaluate_results`

Metric functions are importable from `retrieval_lab.metrics` and may also be
re-exported later when the metric surface is stable.

## Domain contracts

All domain records are typed dataclasses. Identifiers and text must be non-empty.
Offsets are zero-based, the end offset is exclusive, and `0 <= start < end`.

`EvaluationQuery` stores document and chunk relevance separately. At least one
positive relevant identifier is required. Inputs supplied as mutable sets are
normalized to `frozenset` values.

`SearchResult.rank` is one-based. Scores must be finite. A result identifies both
its chunk and parent document so evaluation can explicitly choose document or
chunk relevance.

## Retriever contract

`BaseRetriever` exposes:

```python
class BaseRetriever(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def index(self, chunks: Sequence[Chunk]) -> None: ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[SearchResult]: ...
```

Results are ordered best-first and ranks are contiguous from one. Implementations
must use deterministic tie-breaking.

`BaseRetriever.settings` returns deterministic, JSON-compatible implementation
settings for the run manifest. The manifest keeps its existing `retrievers` list
and records detailed values separately under `retriever_settings`.

`DenseRetriever` performs exact inner-product search. It defaults to
`intfloat/multilingual-e5-small`, applies `query: ` and `passage: ` prefixes, and
normalizes document and query vectors unless `normalize_embeddings=False` is set.
It accepts a typed `EmbeddingBackend` for custom embedding services. The default
sentence-transformers adapter is lazy; install it only when needed with
`pip install retrieval-lab[dense]`.

`HybridRetriever` combines two or more already-configured public retrievers with
reciprocal-rank fusion (RRF). Its constructor takes a `sources` sequence of
`BaseRetriever` instances and the positive integer options `rrf_k` (the RRF
constant, default `60`) and `candidate_k` (the number requested from each source,
default `100`). `candidate_k` must be at least the evaluation `top_k`. The hybrid
retriever passes the exact same shared chunk sequence to every source when
`index(chunks)` is called, then requests each source's ranking with the same
`candidate_k`; source scores are ignored and one-based ranks are fused
deterministically. Source names must be unique and nested hybrids are rejected.
The public constructor and settings are available without optional dependencies:

```python
from retrieval_lab import BM25Retriever, HybridRetriever, KeywordRetriever

hybrid = HybridRetriever(
    [KeywordRetriever(), BM25Retriever()],
    rrf_k=60,
    candidate_k=100,
)
assert hybrid.settings["rrf_k"] == 60
assert hybrid.settings["candidate_k"] == 100
```

Custom BM25 tokenizer functions receive a privacy-preserving fingerprint of their
code, defaults, and immutable closure state. Stateful callable objects, or
functions whose captured state cannot be fingerprinted safely, must pass a stable
`tokenizer_identity`; only its SHA-256 digest is recorded.

Custom `EmbeddingBackend` implementations must expose a stable, non-null,
JSON-compatible `cache_identity` that changes whenever embedding behavior or
provider configuration changes. Only its SHA-256 digest is stored in cache keys
and manifests; raw provider settings and credentials are never serialized.

For an existing search API, use the synchronous callable contract:

```python
class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedItem]: ...
```

`CallableRetriever(name, callable)` validates immutable `RetrievedItem` records,
including finite scores, optional contiguous ranks, duplicate IDs, and the
requested cutoff. `evaluate_retrievers(dataset=..., retrievers=..., top_k=...)`
evaluates these adapters without a corpus or index. Fully scored results are
normalized by `(-score, id)`; results without complete scores preserve the
returned sequence. Both use the shared metrics/latency/result schema.
Runner-produced query results also store `retrieved_ids_by_cutoff`, so each
cutoff metric has its exact ranking evidence; legacy artifacts without this
optional field remain loadable.
For chunk-level relevance, pass the stable chunk-artifact identity as
`chunk_hash`. Strict comparison requires matching chunk provenance;
document-level comparison keeps chunk changes experimental.

For an async provider, implement or wrap the separate protocol:

```python
class AsyncRetriever(Protocol):
    @property
    def name(self) -> str: ...

    async def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedItem]: ...
```

`AsyncCallableRetriever(name, callable)` and
`await evaluate_async_retrievers(dataset=..., retrievers=..., top_k=...)` use
the same item validation and result schema. `concurrency` bounds in-flight
retrievals, `repetitions` checks deterministic rankings, and `timeout_s` is an
optional per-call deadline. Results retain dataset query and sorted retriever
order regardless of completion order. A failed query fails the whole run;
caller `asyncio.CancelledError` is cleaned up and re-raised. This async entry
point must be awaited from the caller's existing event loop; the synchronous
entry point never starts an event loop or silently uses threads.

## Runner contract

The first vertical slice supports:

```python
EvaluationRunner.quick_evaluate(
    *,
    documents: Sequence[Document],
    queries: Sequence[EvaluationQuery],
    strategies: Sequence[str],
    top_k: Sequence[int],
) -> EvaluationResult
```

The `keyword` and `bm25` strategies are accepted. The runner uses a shared
`FixedSizeChunker` and searches once at `max(top_k)` for each query. Unsupported
strategies fail explicitly.

`EvaluationRunner.from_dataset()` preserves graded relevance loaded from JSONL.
Document relevance collapses repeated parent documents before scoring. Chunk
relevance evaluates chunk identifiers directly and validates that every gold chunk
exists in the shared chunk artifact.

`EvaluationRunner` and `from_dataset()` accept an optional `cache_dir`. When set,
the runner persists validated, content-addressed chunk and Dense-index artifacts as
canonical JSON. Invalid or incompatible cache data is reported in runtime manifest
events and rebuilt. `cache_max_bytes` bounds each cache read and streamed write;
it defaults to 64 MiB and must be a positive integer. An artifact that exceeds the
limit is left uncached and reported as `skipped`, without failing the evaluation.
`from_config()` uses the default limit. Cache state does not change the
deterministic `run_id`.

`EvaluationRunner(...).run()` uses the same application path as
`quick_evaluate`; the classmethod is convenience syntax, not separate logic.

`EvaluationRunner.from_config(path_or_config)` uses that same application path.
YAML requires `schema_version: 1`, rejects duplicate and unknown fields, uses a
safe loader, does not expand environment variables, and resolves relative paths
from the configuration file's parent. The normalized configuration is recorded
under `manifest["config"]` without absolute paths. Runtime paths and report-output
choices do not change the deterministic run ID.

The v0.1 YAML adapter supports the fixed character chunker exposed as
`recursive_characters`, native evaluation JSONL, canonical built-in retriever
names, RRF Hybrid references, and all six implemented metrics. The YAML runner
remains the synchronous built-in-index entry point; async
external retrieval uses the explicit callable API above. Report and
quality-gate blocks are strictly validated and retained for their dedicated
execution APIs.

## File and precomputed-result contracts

`load_documents()` accepts a single supported file or a directory. TXT and
Markdown use normalized relative POSIX paths as stable document IDs. Corpus JSONL
uses explicit `id`, `text`, and optional `metadata` fields.

`EvaluationDataset.from_jsonl()` accepts the canonical `query_id`, `query`, and
graded `relevant` array. `validate()` checks every gold document or chunk ID
against the supplied corpus.

`evaluate_results()` accepts exactly one `RetrievedQueryResult` for every dataset
query. It computes the same metrics and dataset hash as `EvaluationRunner` without
indexing or retrieval. Precomputed chunk rankings are not yet supported.

## Result contract

`EvaluationResult.schema_version` is `1`. It exposes per-retriever aggregate
metrics, nearest-rank search latency (`mean_ms`, `p50_ms`, `p95_ms`, `max_ms`,
`sample_count`, and `failure_count`), and per-query evidence. Each successful
query records `search_latency_ms`; `warnings` notes unstable p95 estimates below
20 samples. `RetrieverMetrics.recall_at(k)` returns the macro average Recall@k
and raises `KeyError` when k was not evaluated.

Latency and environment observations are runtime-only and do not contribute to
`run_id`. The runner uses seed `42` by default; changing it changes the
deterministic run identity. Result schema version `1` remains unchanged while
these additive fields are pre-release additions.

`to_dict()`, `to_json()`, and `save_json(path)` use one canonical JSON-compatible
schema. JSON output is UTF-8, preserves Japanese text, sorts keys, and has a final
newline. Parent directories are created by `save_json`, which atomically replaces
the destination after flushing and syncing a same-directory temporary file.

Results can be loaded through `EvaluationResult.from_dict(payload)`,
`EvaluationResult.from_json(text)`, `EvaluationResult.load_json(path)`, or the
package-root alias `load_result(path)`. JSON loading accepts schema version 1,
rejects duplicate keys, non-finite numbers, malformed metrics, inconsistent
aggregates, and mixed partial per-query latency data when an aggregate exists.
Aggregate-only legacy results (all query timing fields absent) remain readable;
when timing fields are present they must not mix timed and untimed queries.
`failure_count` records failed calls in the aggregate and does not authorize an
ambiguous mixed per-query representation. Unknown additive fields are ignored;
typed non-empty `quality_gates` results are validated and restored as immutable
records. `load_json` defaults to a 64 MiB file limit.

`evaluate_quality_gates(candidate, gates, baseline=...)` evaluates every
configured constraint in order. `min_value` and `max_value` apply to the
candidate value. `max_absolute_drop` and `max_relative_drop` require a baseline
and use the saved-run comparison direction, with lower latency treated as an
improvement. `QualityGateReport.passed`, `.failed`, `.to_dict()`, and `.to_json()`
provide library-level results without printing. Attach results immutably with
`EvaluationResult.with_quality_gates(report)`; the canonical root
`quality_gates` list round-trips through the existing result loader.
An empty gate sequence produces an empty passing report and preserves the
existing empty `quality_gates` result shape.

## Application services and CLI

The package provides typed application services. `initialize_project(target)`
creates a fixed offline sample without overwriting owned files unless
`force=True`. `validate_config_inputs(config_path)` validates configuration,
corpus, dataset, and relevance IDs without executing retrieval.
`run_configured_experiment(config_path, ...)` runs through
`EvaluationRunner.from_config()` and writes deterministic `result.json`, CSV,
and standalone HTML outputs. The services never print and return typed result
records. `inspect_result(path, query_id=...)` loads a strict result and
prepares metadata/evidence, `compare_result_files(baseline, candidate)`
prepares deterministic deltas, and `evaluate_configured_quality_gates(config,
candidate, baseline_path=...)` evaluates an explicit config. Passing only the
candidate restores the normalized gates embedded in its run manifest.

The `retrieval-lab` console adapter is intentionally thin:

```console
retrieval-lab init ./my-evaluation
retrieval-lab validate -c ./my-evaluation/retrieval-lab.yaml
retrieval-lab run -c ./my-evaluation/retrieval-lab.yaml
retrieval-lab inspect ./artifacts/result.json
retrieval-lab compare ./baseline/result.json ./candidate/result.json
retrieval-lab gate ./candidate/result.json --baseline ./baseline/result.json
# Optional override: retrieval-lab gate -c retrieval-lab.yaml candidate/result.json
```

Artifact-only gating treats the baseline's embedded policy as authoritative and
requires the candidate policy hash to match it. Candidate-only artifact gating
assumes the result is from a trusted producer; use an explicit `--config` when
the policy must be supplied independently of the artifact.

`init` rejects existing owned files and symlinked template targets. `run`
accepts repeatable `--format json|csv|html` and an optional `--output-dir`.
Success is written to stdout; concise errors are written to stderr without
tracebacks or absolute paths.

`inspect` accepts `--query-id` to display deterministic per-retriever query
evidence. `inspect`, `compare`, and `gate` accept `--json` for strict,
machine-readable output and `--debug` to opt into tracebacks. `compare` prints
baseline/candidate values and absolute/relative deltas, including latency with
lower-is-better semantics, and reports non-blocking experimental variable
differences. `gate` returns exit status `1` only when a quality
gate fails. Malformed input and incomparable saved artifacts return `2`;
retrieval execution failures return `3`.

The JSON command shapes are stable: `inspect` returns `command`, `run_id`,
`schema_version`, `retrievers`, `quality_gates`, and a deterministic `summary`
(plus `query` with `query_id` and per-retriever `evidence` when requested);
`compare` returns `baseline_run_id`, `candidate_run_id`, `common_retrievers`,
`diagnostics`, `variable_differences`, and metric rows. Experimental-variable
rows contain only `field` and `reason`; manifest values are intentionally omitted
so paths, credentials, and other private configuration cannot enter comparison
output. `gate` returns `candidate_run_id`, optional
`baseline_run_id`, `passed`, and the canonical `quality_gates` rows.

Saved runs can be compared without re-running retrieval:

```python
from retrieval_lab import compare_runs, load_result

comparison = compare_runs(
    load_result("baseline/result.json"),
    load_result("candidate/result.json"),
)
```

`check_comparability()` returns all blocking mismatches as typed issues without
raising. `compare_runs()` raises `IncomparableRunError` with the complete issue
tuple when dataset hash, query IDs, relevance level, metric version, top-k, or
metric shapes are incompatible. Retriever, corpus, chunk, configuration, seed,
run identity, timestamp, and latency-presence differences are reported as
experimental differences; only common retrievers receive metric deltas.

`summary()` returns deterministic plain text and never prints. `to_summary_csv()`
and `to_per_query_csv()` return UTF-8 long-form CSV; the corresponding save
methods and `save_csv(output_dir)` use atomic writes. `to_html()` and
`save_html(path)` produce a standalone escaped report with inline CSS only.
Reports show metrics, latency, warnings, safe normalized configuration fields,
and attention query IDs, but never query text, retrieved IDs, document text,
absolute paths, secrets, or external resources.

Saved-run CI usage is documented in `docs/ci-regression.md`. The checked-in
fixtures and `examples/github-actions/retrieval-quality-gate.yml` show the
offline `gate` command, baseline artifact contract, and stable exit statuses.

The offline first-run walkthrough is in `docs/tutorial.md`; operational caveats
for metrics, relevance levels, latency, Dense dependencies, caches, and
comparability are collected in `docs/faq.md`. All tutorial code uses the public
package API.
