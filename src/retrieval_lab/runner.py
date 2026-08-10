"""Public application service for retrieval evaluation."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from retrieval_lab.artifacts.cache import (
    CacheRead,
    CacheStatus,
    dense_index_hash,
    read_chunk_artifact,
    read_dense_index_artifact,
    write_chunk_artifact,
    write_dense_index_artifact,
)
from retrieval_lab.chunkers import FixedSizeChunker
from retrieval_lab.datasets import EvaluationDataset, RelevanceLevel
from retrieval_lab.domain import (
    Chunk,
    Document,
    EvaluationQuery,
    EvaluationResult,
    JSONValue,
    QueryEvaluation,
    RetrieverMetrics,
)
from retrieval_lab.evaluation.engine import (
    aggregate_metrics,
    content_hash,
    dataset_payload,
    evaluate_ranking,
    normalize_top_k,
    plain_json,
)
from retrieval_lab.evaluation.ranking import (
    collapse_to_documents,
    stable_rank_results,
)
from retrieval_lab.exceptions import (
    ConfigurationError,
    CorpusValidationError,
    DatasetValidationError,
    EvaluationError,
    RetrievalLabError,
)
from retrieval_lab.retrievers import (
    BaseRetriever,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    KeywordRetriever,
)


class EvaluationRunner:
    """Evaluate one or more retrievers over shared documents and queries."""

    def __init__(
        self,
        *,
        documents: Sequence[Document],
        queries: Sequence[EvaluationQuery] | None = None,
        dataset: EvaluationDataset | None = None,
        retrievers: Sequence[BaseRetriever] | None = None,
        strategies: Sequence[str] | None = None,
        top_k: Sequence[int] = (1, 3, 5, 10),
        chunker: FixedSizeChunker | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        """Validate an in-memory retrieval experiment."""

        self._documents = _validate_documents(documents)
        if (queries is None) == (dataset is None):
            raise ConfigurationError("provide exactly one of queries or dataset")
        self._dataset: EvaluationDataset | None = dataset
        self._relevance_level: RelevanceLevel
        if dataset is not None:
            if not isinstance(dataset, EvaluationDataset):
                raise DatasetValidationError("dataset must be an EvaluationDataset")
            self._queries = dataset.queries
            self._relevance_level = dataset.relevance_level
            self._relevance_grades_by_query = dataset.relevance_grades_by_query
            if dataset.relevance_level == "document":
                dataset.validate(documents=self._documents)
        else:
            assert queries is not None
            self._queries = _validate_queries(queries)
            self._relevance_level = "document"
            self._relevance_grades_by_query = {
                query.id: {
                    identifier: 1 for identifier in sorted(query.relevant_document_ids)
                }
                for query in self._queries
            }
        self._top_k = normalize_top_k(top_k)
        self._chunker = FixedSizeChunker() if chunker is None else chunker
        if not isinstance(self._chunker, FixedSizeChunker):
            raise ConfigurationError("chunker must be a FixedSizeChunker")
        self._retrievers = _resolve_retrievers(retrievers, strategies)
        if cache_dir is None:
            self._cache_dir: Path | None = None
        else:
            if isinstance(cache_dir, str) and not cache_dir.strip():
                raise ConfigurationError("cache_dir must not be empty")
            try:
                self._cache_dir = Path(cache_dir)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("cache_dir must be a valid path") from exc
            if not str(self._cache_dir):
                raise ConfigurationError("cache_dir must not be empty")

    @classmethod
    def quick_evaluate(
        cls,
        *,
        documents: Sequence[Document],
        queries: Sequence[EvaluationQuery],
        strategies: Sequence[str] = ("keyword",),
        top_k: Sequence[int] = (1, 3, 5, 10),
    ) -> EvaluationResult:
        """Run the supported built-in strategies with their default settings."""

        return cls(
            documents=documents,
            queries=queries,
            strategies=strategies,
            top_k=top_k,
        ).run()

    @classmethod
    def from_dataset(
        cls,
        *,
        documents: Sequence[Document],
        dataset: EvaluationDataset,
        retrievers: Sequence[BaseRetriever] | None = None,
        strategies: Sequence[str] | None = None,
        top_k: Sequence[int] = (1, 3, 5, 10),
        chunker: FixedSizeChunker | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> EvaluationRunner:
        """Create a runner without discarding graded dataset relevance."""

        return cls(
            documents=documents,
            dataset=dataset,
            retrievers=retrievers,
            strategies=strategies,
            top_k=top_k,
            chunker=chunker,
            cache_dir=cache_dir,
        )

    def run(self) -> EvaluationResult:
        """Build shared chunks, retrieve once per query, and evaluate all cutoffs."""

        chunks: Sequence[Chunk] = self._chunker.chunk(self._documents)
        if not chunks:
            raise CorpusValidationError("chunking produced no evaluable chunks")
        if self._dataset is not None and self._relevance_level == "chunk":
            self._dataset.validate(chunks=chunks)

        metrics: dict[str, RetrieverMetrics] = {}
        query_results: dict[str, tuple[QueryEvaluation, ...]] = {}
        max_k = max(self._top_k)

        chunk_hash = _chunk_hash(
            documents=self._documents,
            chunks=chunks,
            chunker=self._chunker,
        )
        cache_events: list[dict[str, JSONValue]] = []
        index_hashes: dict[str, JSONValue] = {}
        if self._cache_dir is not None:
            chunks, chunk_event = self._prepare_chunk_cache(
                chunks,
                chunk_hash=chunk_hash,
            )
            cache_events.append(chunk_event)
            index_hashes, dense_events = self._prepare_dense_cache(
                chunks,
                chunk_hash=chunk_hash,
            )
            cache_events.extend(dense_events)

        for retriever in self._retrievers:
            try:
                retriever.index(chunks)
                evaluations = tuple(
                    self._evaluate_query(retriever, query, max_k=max_k)
                    for query in self._queries
                )
            except RetrievalLabError:
                raise
            except Exception as exc:
                raise EvaluationError(
                    f"retriever {retriever.name!r} failed during evaluation"
                ) from exc

            metrics[retriever.name] = aggregate_metrics(evaluations, self._top_k)
            query_results[retriever.name] = evaluations

        manifest, run_id = _build_manifest(
            documents=self._documents,
            queries=self._queries,
            chunks=chunks,
            retrievers=self._retrievers,
            top_k=self._top_k,
            chunker=self._chunker,
            relevance_level=self._relevance_level,
            relevance_grades_by_query=self._relevance_grades_by_query,
            chunk_hash=chunk_hash,
            index_hashes=index_hashes,
            cache_events=cache_events,
        )
        return EvaluationResult(
            run_id=run_id,
            metrics=metrics,
            query_results=query_results,
            manifest=manifest,
        )

    def _prepare_chunk_cache(
        self,
        chunks: Sequence[Chunk],
        *,
        chunk_hash: str,
    ) -> tuple[tuple[Chunk, ...], dict[str, JSONValue]]:
        """Read a safe chunk artifact, rebuilding it on any invalid outcome."""

        assert self._cache_dir is not None
        read = read_chunk_artifact(self._cache_dir, chunk_hash)
        if read.status is CacheStatus.HIT:
            cached = read.payload
            if isinstance(cached, tuple) and cached == tuple(chunks):
                return tuple(cached), {"artifact": "chunks", "status": "hit"}
            read = CacheRead(CacheStatus.CORRUPT, reason="chunk content mismatch")
        write_chunk_artifact(self._cache_dir, chunk_hash, chunks)
        event: dict[str, JSONValue] = {
            "artifact": "chunks",
            "status": read.status.value,
        }
        if read.reason is not None:
            event["reason"] = read.reason
        return tuple(chunks), event

    def _prepare_dense_cache(
        self,
        chunks: Sequence[Chunk],
        *,
        chunk_hash: str,
    ) -> tuple[dict[str, JSONValue], list[dict[str, JSONValue]]]:
        """Restore or build every shared DenseRetriever before evaluation."""

        assert self._cache_dir is not None
        index_hashes: dict[str, JSONValue] = {}
        events: list[dict[str, JSONValue]] = []
        seen: set[int] = set()
        for retriever in _find_dense_retrievers(self._retrievers):
            if id(retriever) in seen:
                continue
            seen.add(id(retriever))
            settings = plain_json(retriever.settings)
            index_hash = dense_index_hash(chunk_hash, settings)
            index_hashes[retriever.name] = index_hash
            identity: dict[str, JSONValue] = {
                "name": retriever.name,
                "settings": settings,
            }
            read = read_dense_index_artifact(
                self._cache_dir,
                index_hash=index_hash,
                chunk_hash=chunk_hash,
                retriever_identity=identity,
                chunks=chunks,
                model_id=str(settings["model_id"]),
                requested_revision=_optional_string(settings.get("requested_revision")),
                expected_resolved_revision=_optional_string(
                    settings.get("resolved_revision")
                ),
            )
            event: dict[str, JSONValue] = {
                "artifact": "dense_index",
                "retriever": retriever.name,
                "status": read.status.value,
                "index_hash": index_hash,
            }
            if read.reason is not None:
                event["reason"] = read.reason
            if read.status is CacheStatus.HIT and isinstance(read.payload, dict):
                try:
                    retriever._cache_restore(
                        chunks,
                        read.payload["vectors"],
                        resolved_revision=read.payload["resolved_revision"],
                    )
                except Exception as exc:
                    event["status"] = CacheStatus.CORRUPT.value
                    event["reason"] = str(exc)
                else:
                    events.append(event)
                    continue

            try:
                retriever.index(chunks)
                indexed_chunks, vectors, dimension, metadata = retriever._cache_export()
                write_dense_index_artifact(
                    self._cache_dir,
                    index_hash=index_hash,
                    chunk_hash=chunk_hash,
                    retriever_identity=identity,
                    chunks=indexed_chunks,
                    vectors=vectors,
                    dimension=dimension,
                    model_id=metadata.model_id,
                    requested_revision=metadata.requested_revision,
                    resolved_revision=metadata.resolved_revision,
                )
            except RetrievalLabError:
                raise
            except Exception as exc:
                raise EvaluationError(
                    f"dense retriever {retriever.name!r} cache rebuild failed"
                ) from exc
            events.append(event)
        return index_hashes, events

    def _evaluate_query(
        self,
        retriever: BaseRetriever,
        query: EvaluationQuery,
        *,
        max_k: int,
    ) -> QueryEvaluation:
        raw_results = retriever.search(query.query, max_k)
        ranked_chunks = stable_rank_results(raw_results, top_k=max_k)
        if self._relevance_level == "document":
            if not query.relevant_document_ids:
                raise DatasetValidationError(
                    f"EvaluationQuery[{query.id!r}] has no document relevance for "
                    "this document-level runner"
                )
            ranked_documents = collapse_to_documents(ranked_chunks)
            retrieved_ids = tuple(item.document_id for item in ranked_documents)
        else:
            if not query.relevant_chunk_ids:
                raise DatasetValidationError(
                    f"EvaluationQuery[{query.id!r}] has no chunk relevance for "
                    "this chunk-level runner"
                )
            retrieved_ids = tuple(item.chunk_id for item in ranked_chunks)

        return evaluate_ranking(
            query_id=query.id,
            retrieved_ids=retrieved_ids,
            relevance_grades=self._relevance_grades_by_query[query.id],
            top_k=self._top_k,
        )


def _validate_documents(documents: Sequence[Document]) -> tuple[Document, ...]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise CorpusValidationError("documents must be a sequence of Document values")
    normalized = tuple(documents)
    if not normalized:
        raise CorpusValidationError("documents must not be empty")
    if not all(isinstance(document, Document) for document in normalized):
        raise CorpusValidationError("documents must contain only Document values")
    identifiers = [document.id for document in normalized]
    if len(set(identifiers)) != len(identifiers):
        raise CorpusValidationError("document IDs must be unique")
    return normalized


def _validate_queries(
    queries: Sequence[EvaluationQuery],
) -> tuple[EvaluationQuery, ...]:
    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
        raise DatasetValidationError(
            "queries must be a sequence of EvaluationQuery values"
        )
    normalized = tuple(queries)
    if not normalized:
        raise DatasetValidationError("queries must not be empty")
    if not all(isinstance(query, EvaluationQuery) for query in normalized):
        raise DatasetValidationError("queries must contain only EvaluationQuery values")
    identifiers = [query.id for query in normalized]
    if len(set(identifiers)) != len(identifiers):
        raise DatasetValidationError("query IDs must be unique")
    return normalized


def _resolve_retrievers(
    retrievers: Sequence[BaseRetriever] | None,
    strategies: Sequence[str] | None,
) -> tuple[BaseRetriever, ...]:
    resolved: list[BaseRetriever] = []
    if strategies is not None:
        if isinstance(strategies, (str, bytes)) or not isinstance(strategies, Sequence):
            raise ConfigurationError("strategies must be a sequence of names")
        if not strategies:
            raise ConfigurationError("strategies must not be empty")
        for strategy in strategies:
            if strategy == "keyword":
                resolved.append(KeywordRetriever())
            elif strategy == "bm25":
                resolved.append(BM25Retriever())
            else:
                raise ConfigurationError(
                    f"unsupported strategy {strategy!r}; supported built-ins are "
                    "'keyword' and 'bm25'"
                )

    if retrievers is not None:
        if isinstance(retrievers, (str, bytes)) or not isinstance(retrievers, Sequence):
            raise ConfigurationError("retrievers must be a sequence")
        if not all(isinstance(retriever, BaseRetriever) for retriever in retrievers):
            raise ConfigurationError(
                "retrievers must contain only BaseRetriever implementations"
            )
        resolved.extend(retrievers)

    if not resolved:
        resolved.append(KeywordRetriever())
    names = [retriever.name for retriever in resolved]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ConfigurationError("retriever names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ConfigurationError("retriever names must be unique")
    return tuple(resolved)


def _build_manifest(
    *,
    documents: Sequence[Document],
    queries: Sequence[EvaluationQuery],
    chunks: Sequence[Chunk],
    retrievers: Sequence[BaseRetriever],
    top_k: Sequence[int],
    chunker: FixedSizeChunker,
    relevance_level: RelevanceLevel,
    relevance_grades_by_query: Mapping[str, Mapping[str, int]],
    chunk_hash: str | None = None,
    index_hashes: Mapping[str, JSONValue] | None = None,
    cache_events: Sequence[Mapping[str, JSONValue]] = (),
) -> tuple[dict[str, JSONValue], str]:
    corpus_payload: list[JSONValue] = []
    for document in documents:
        corpus_record: dict[str, JSONValue] = {
            "id": document.id,
            "metadata": plain_json(document.metadata),
            "source": document.source,
            "text": document.text,
        }
        corpus_payload.append(corpus_record)

    chunks_payload: list[JSONValue] = []
    for chunk in chunks:
        chunk_record: dict[str, JSONValue] = {
            "document_id": chunk.document_id,
            "end_offset": chunk.end_offset,
            "id": chunk.id,
            "start_offset": chunk.start_offset,
            "text": chunk.text,
        }
        chunks_payload.append(chunk_record)
    corpus_hash = content_hash(corpus_payload)
    dataset_hash = content_hash(dataset_payload(queries, relevance_grades_by_query))
    calculated_chunk_hash = content_hash(
        {
            "chunks": chunks_payload,
            "corpus_hash": corpus_hash,
            "settings": {"overlap": chunker.overlap, "size": chunker.size},
        }
    )
    if chunk_hash is not None and chunk_hash != calculated_chunk_hash:
        raise EvaluationError("chunk hash changed while building the manifest")
    chunk_hash = calculated_chunk_hash
    retriever_names = tuple(retriever.name for retriever in retrievers)
    retriever_settings: dict[str, JSONValue] = {
        retriever.name: plain_json(retriever.settings) for retriever in retrievers
    }
    run_payload: dict[str, JSONValue] = {
        "chunk_hash": chunk_hash,
        "dataset_hash": dataset_hash,
        "metric_version": 1,
        "relevance_level": relevance_level,
        "retrievers": _json_values(retriever_names),
        "retriever_settings": retriever_settings,
        "top_k": _json_values(top_k),
    }
    run_id = content_hash(run_payload)
    manifest: dict[str, JSONValue] = {
        "chunk_count": len(chunks),
        "chunk_hash": chunk_hash,
        "chunking": {"overlap": chunker.overlap, "size": chunker.size},
        "corpus_hash": corpus_hash,
        "dataset_hash": dataset_hash,
        "document_count": len(documents),
        "metric_version": 1,
        "query_count": len(queries),
        "query_ids": _json_values(tuple(query.id for query in queries)),
        "relevance_level": relevance_level,
        "retrievers": _json_values(retriever_names),
        "retriever_settings": retriever_settings,
        "top_k": _json_values(top_k),
    }
    if index_hashes:
        manifest["index_hashes"] = {
            key: index_hashes[key] for key in sorted(index_hashes)
        }
    if cache_events:
        # Runtime cache observations are intentionally outside run_payload, so
        # hit/miss/corruption never changes the reproducible run ID.
        manifest["runtime"] = {
            "cache_events": [
                {key: event[key] for key in sorted(event)} for event in cache_events
            ]
        }
    return manifest, run_id


def _json_values(values: Sequence[str | int]) -> list[JSONValue]:
    result: list[JSONValue] = []
    result.extend(values)
    return result


def _chunk_hash(
    *,
    documents: Sequence[Document],
    chunks: Sequence[Chunk],
    chunker: FixedSizeChunker,
) -> str:
    """Return the same content hash recorded in the deterministic manifest."""

    corpus_payload: list[JSONValue] = [
        {
            "id": document.id,
            "metadata": plain_json(document.metadata),
            "source": document.source,
            "text": document.text,
        }
        for document in documents
    ]
    chunks_payload: list[JSONValue] = [
        {
            "document_id": chunk.document_id,
            "end_offset": chunk.end_offset,
            "id": chunk.id,
            "start_offset": chunk.start_offset,
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    corpus_hash = content_hash(corpus_payload)
    return content_hash(
        {
            "chunks": chunks_payload,
            "corpus_hash": corpus_hash,
            "settings": {"overlap": chunker.overlap, "size": chunker.size},
        }
    )


def _find_dense_retrievers(
    retrievers: Sequence[BaseRetriever],
) -> tuple[DenseRetriever, ...]:
    """Find built-in Dense instances, including shared Hybrid sources."""

    found: list[DenseRetriever] = []
    for retriever in retrievers:
        if isinstance(retriever, DenseRetriever):
            found.append(retriever)
        elif isinstance(retriever, HybridRetriever):
            found.extend(
                source
                for source in retriever._sources
                if isinstance(source, DenseRetriever)
            )
    return tuple(found)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = ["EvaluationRunner"]
