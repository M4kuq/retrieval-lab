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
- `EmbeddingBackend`
- `EmbeddingModelMetadata`
- `OptionalDependencyError`
- `FixedSizeChunker`
- `EvaluationDataset`
- `RelevanceLevel`
- `RetrievedQueryResult`
- `EvaluationRunner`
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

`EvaluationRunner(...).run()` uses the same application path as
`quick_evaluate`; the classmethod is convenience syntax, not separate logic.

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
metrics and per-query evidence. `RetrieverMetrics.recall_at(k)` returns the macro
average Recall@k and raises `KeyError` when k was not evaluated.

`to_dict()`, `to_json()`, and `save_json(path)` use one canonical JSON-compatible
schema. JSON output is UTF-8, preserves Japanese text, sorts keys, and has a final
newline. Parent directories are created by `save_json`.
