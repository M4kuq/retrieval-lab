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
evaluates these adapters without a corpus or index, using each returned
sequence as its ranking and the shared metrics/latency/result schema.

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
events and rebuilt. Cache state does not change the deterministic `run_id`.

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
names, RRF Hybrid references, and all six implemented metrics. `repetitions` and
`concurrency` must both be `1` until the async execution API is available. Report
and quality-gate blocks are strictly validated and retained for their dedicated
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
newline. Parent directories are created by `save_json`.
