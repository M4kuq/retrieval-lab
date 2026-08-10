"""Public application service for retrieval evaluation."""

from __future__ import annotations

import math
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING

from retrieval_lab.artifacts.cache import (
    CACHE_MAX_BYTES,
    CacheCapacityError,
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
    LatencyStats,
    QueryEvaluation,
    RetrieverMetrics,
)
from retrieval_lab.evaluation.engine import (
    aggregate_metrics,
    content_hash,
    dataset_payload,
    evaluate_cutoff_rankings,
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
    RetrieverContractError,
)
from retrieval_lab.loaders import load_documents
from retrieval_lab.retrievers import (
    BaseRetriever,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    KeywordRetriever,
)

if TYPE_CHECKING:
    from retrieval_lab.config import RetrievalConfig

DEFAULT_CACHE_MAX_BYTES = CACHE_MAX_BYTES


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
        cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES,
        seed: int = 42,
        _config_settings: Mapping[str, JSONValue] | None = None,
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
        self._seed = _validate_seed(seed)
        self._chunker = FixedSizeChunker() if chunker is None else chunker
        if not isinstance(self._chunker, FixedSizeChunker):
            raise ConfigurationError("chunker must be a FixedSizeChunker")
        self._retrievers = _resolve_retrievers(retrievers, strategies)
        self._cache_max_bytes = _validate_cache_max_bytes(cache_max_bytes)
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
        self._config_settings = (
            plain_json(_config_settings) if _config_settings is not None else None
        )

    @classmethod
    def from_config(
        cls,
        path: str | os.PathLike[str] | RetrievalConfig,
    ) -> EvaluationRunner:
        """Build a runner from a strict YAML file or typed ``RetrievalConfig``.

        YAML paths are resolved relative to the configuration file.  Quality
        gates and report settings are validated and retained in the manifest,
        but are not executed by this evaluation API.
        """

        from retrieval_lab.config import RetrievalConfig, load_config

        config = path if isinstance(path, RetrievalConfig) else load_config(Path(path))
        return cls._from_typed_config(config)

    @classmethod
    def _from_typed_config(cls, config: object) -> EvaluationRunner:
        from retrieval_lab.config import RetrievalConfig

        if not isinstance(config, RetrievalConfig):
            raise ConfigurationError("config must be a RetrievalConfig")
        documents = load_documents(
            config.corpus.path,
            include=config.corpus.include,
        )
        dataset = EvaluationDataset.from_jsonl(
            config.dataset.path,
            relevance_level=config.dataset.relevance_level,
        )
        retriever_by_name: dict[str, BaseRetriever] = {}
        for item in config.retrievers:
            if item.type == "keyword":
                retriever_by_name[item.name] = KeywordRetriever()
            elif item.type == "bm25":
                retriever_by_name[item.name] = BM25Retriever(k1=item.k1, b=item.b)
            elif item.type == "dense":
                retriever_by_name[item.name] = DenseRetriever(
                    model_id=item.model,
                    revision=item.model_revision,
                    normalize_embeddings=item.normalize_embeddings,
                    batch_size=item.batch_size,
                    query_prompt=item.query_prompt,
                    document_prompt=item.document_prompt,
                )
        for item in config.retrievers:
            if item.type == "hybrid":
                sources = [retriever_by_name[name] for name in item.sources]
                retriever_by_name[item.name] = HybridRetriever(
                    sources,
                    rrf_k=item.rrf_k,
                    candidate_k=item.candidate_k,
                )
        return cls(
            documents=documents,
            dataset=dataset,
            retrievers=tuple(
                retriever_by_name[item.name] for item in config.retrievers
            ),
            top_k=config.evaluation.top_k,
            chunker=FixedSizeChunker(
                size=config.corpus.chunker.size,
                overlap=config.corpus.chunker.overlap,
            ),
            cache_dir=config.cache_dir,
            seed=config.experiment.seed,
            _config_settings=config.normalized_settings(),
        )

    @classmethod
    def quick_evaluate(
        cls,
        *,
        documents: Sequence[Document],
        queries: Sequence[EvaluationQuery],
        strategies: Sequence[str] = ("keyword",),
        top_k: Sequence[int] = (1, 3, 5, 10),
        seed: int = 42,
    ) -> EvaluationResult:
        """Run the supported built-in strategies with their default settings."""

        return cls(
            documents=documents,
            queries=queries,
            strategies=strategies,
            top_k=top_k,
            seed=seed,
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
        cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES,
        seed: int = 42,
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
            cache_max_bytes=cache_max_bytes,
            seed=seed,
        )

    def run(self) -> EvaluationResult:
        """Build shared chunks, retrieve once per query, and evaluate all cutoffs."""
        started_at = _utc_timestamp()
        chunks: Sequence[Chunk] = self._chunker.chunk(self._documents)
        if not chunks:
            raise CorpusValidationError("chunking produced no evaluable chunks")
        if self._dataset is not None and self._relevance_level == "chunk":
            self._dataset.validate(chunks=chunks)

        metrics: dict[str, RetrieverMetrics] = {}
        query_results: dict[str, tuple[QueryEvaluation, ...]] = {}
        latency_stats: dict[str, LatencyStats] = {}
        build_ms: dict[str, float] = {}
        index_sizes_bytes: dict[str, int] = {}
        max_k = max(self._top_k)

        chunk_hash = _chunk_hash(
            documents=self._documents,
            chunks=chunks,
            chunker=self._chunker,
        )
        cache_events: list[dict[str, JSONValue]] = []
        cache_build_ms: dict[int, float] = {}
        shared_cache_ms = 0.0
        if self._cache_dir is not None:
            chunks, chunk_event = self._prepare_chunk_cache(
                chunks,
                chunk_hash=chunk_hash,
            )
            cache_events.append(chunk_event)
            shared_cache_ms = _event_duration_ms(chunk_event)
            index_hashes, dense_events, dense_build_ms = self._prepare_dense_cache(
                chunks,
                chunk_hash=chunk_hash,
            )
            cache_events.extend(dense_events)
            cache_build_ms.update(dense_build_ms)

        indexed_retriever_ids: set[int] = (
            {
                id(retriever)
                for _identity, retriever in _dense_retriever_entries(self._retrievers)
            }
            if self._cache_dir is not None
            else set()
        )
        for retriever in self._retrievers:
            build_started_ns = perf_counter_ns()
            try:
                if isinstance(retriever, HybridRetriever):
                    retriever._index_sources_once(chunks, indexed_retriever_ids)
                elif id(retriever) not in indexed_retriever_ids:
                    retriever.index(chunks)
                    indexed_retriever_ids.add(id(retriever))
            except RetrievalLabError:
                raise
            except Exception as exc:
                raise EvaluationError(
                    f"retriever {retriever.name!r} failed during indexing"
                ) from exc
            build_ms[retriever.name] = (
                _elapsed_ms(build_started_ns)
                + shared_cache_ms
                + _cached_dense_build_ms(retriever, cache_build_ms)
            )
            try:
                index_size = _index_size_bytes(retriever)
            except RetrievalLabError:
                raise
            except Exception as exc:
                raise EvaluationError(
                    f"retriever {retriever.name!r} failed while measuring index size"
                ) from exc
            if index_size is not None:
                if (
                    isinstance(index_size, bool)
                    or not isinstance(index_size, int)
                    or index_size < 0
                ):
                    raise EvaluationError(
                        f"retriever {retriever.name!r} reported an invalid index size"
                    )
                index_sizes_bytes[retriever.name] = index_size

            samples: list[float] = []
            failures = [0]
            evaluations: list[QueryEvaluation] = []
            try:
                for query in self._queries:
                    evaluations.append(
                        self._evaluate_query(
                            retriever,
                            query,
                            max_k=max_k,
                            latency_samples=samples,
                            failure_count=failures,
                        )
                    )
            except RetrievalLabError:
                raise
            except Exception as exc:
                raise EvaluationError(
                    f"retriever {retriever.name!r} failed during evaluation"
                ) from exc

            stats = LatencyStats.from_samples(
                samples,
                failure_count=failures[0],
            )
            if stats.warnings:
                evaluations = [
                    replace(evaluation, warnings=stats.warnings)
                    for evaluation in evaluations
                ]
            metrics[retriever.name] = aggregate_metrics(evaluations, self._top_k)
            query_results[retriever.name] = tuple(evaluations)
            latency_stats[retriever.name] = stats

        index_hashes = _dense_index_hashes(
            self._retrievers,
            chunk_hash=chunk_hash,
        )
        manifest, run_id = _build_manifest(
            documents=self._documents,
            queries=self._queries,
            chunks=chunks,
            retrievers=self._retrievers,
            top_k=self._top_k,
            chunker=self._chunker,
            relevance_level=self._relevance_level,
            relevance_grades_by_query=self._relevance_grades_by_query,
            seed=self._seed,
            chunk_hash=chunk_hash,
            index_hashes=index_hashes,
            cache_events=cache_events,
            runtime=_runtime_manifest(
                retrievers=self._retrievers,
                started_at=started_at,
                build_ms=build_ms,
                index_sizes_bytes=index_sizes_bytes,
                cache_events=cache_events,
            ),
            config_settings=self._config_settings,
        )
        return EvaluationResult(
            run_id=run_id,
            metrics=metrics,
            query_results=query_results,
            manifest=manifest,
            latency=latency_stats,
        )

    def _prepare_chunk_cache(
        self,
        chunks: Sequence[Chunk],
        *,
        chunk_hash: str,
    ) -> tuple[tuple[Chunk, ...], dict[str, JSONValue]]:
        """Read a safe chunk artifact, rebuilding it on any invalid outcome."""

        assert self._cache_dir is not None
        started_ns = perf_counter_ns()
        try:
            read = read_chunk_artifact(
                self._cache_dir,
                chunk_hash,
                max_bytes=self._cache_max_bytes,
            )
            if read.status is CacheStatus.HIT:
                cached = read.payload
                if isinstance(cached, tuple) and cached == tuple(chunks):
                    event: dict[str, JSONValue] = {
                        "artifact": "chunks",
                        "status": "hit",
                    }
                    event["duration_ms"] = _elapsed_ms(started_ns)
                    return tuple(cached), event
                read = CacheRead(CacheStatus.CORRUPT, reason="chunk content mismatch")
            try:
                write_chunk_artifact(
                    self._cache_dir,
                    chunk_hash,
                    chunks,
                    max_bytes=self._cache_max_bytes,
                )
            except CacheCapacityError:
                event = {
                    "artifact": "chunks",
                    "status": "skipped",
                    "reason": "cache artifact exceeds configured capacity",
                    "duration_ms": _elapsed_ms(started_ns),
                }
                return tuple(chunks), event
            event = {
                "artifact": "chunks",
                "status": read.status.value,
            }
            if read.reason is not None:
                event["reason"] = read.reason
            event["duration_ms"] = _elapsed_ms(started_ns)
            return tuple(chunks), event
        except Exception:
            # The caller owns the public error translation; timing must not
            # mask the cache failure with a clock exception.
            raise

    def _prepare_dense_cache(
        self,
        chunks: Sequence[Chunk],
        *,
        chunk_hash: str,
    ) -> tuple[
        dict[str, JSONValue],
        list[dict[str, JSONValue]],
        dict[int, float],
    ]:
        """Restore or build every shared DenseRetriever before evaluation."""

        assert self._cache_dir is not None
        index_hashes: dict[str, JSONValue] = {}
        events: list[dict[str, JSONValue]] = []
        build_ms: dict[int, float] = {}
        for identity_key, retriever in _dense_retriever_entries(self._retrievers):
            settings = _retriever_settings(retriever)
            storage_settings = _dense_storage_settings(settings)
            storage_index_hash = dense_index_hash(chunk_hash, storage_settings)
            identity: dict[str, JSONValue] = {
                "name": retriever.name,
                "settings": storage_settings,
            }
            started_ns = perf_counter_ns()
            read = read_dense_index_artifact(
                self._cache_dir,
                index_hash=storage_index_hash,
                chunk_hash=chunk_hash,
                retriever_identity=identity,
                chunks=chunks,
                model_id=str(settings["model_id"]),
                requested_revision=_optional_string(settings.get("requested_revision")),
                expected_resolved_revision=_optional_string(
                    settings.get("resolved_revision")
                ),
                max_bytes=self._cache_max_bytes,
            )
            event: dict[str, JSONValue] = {
                "artifact": "dense_index",
                "retriever": identity_key,
                "status": read.status.value,
                "storage_index_hash": storage_index_hash,
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
                    logical_index_hash = dense_index_hash(
                        chunk_hash, _retriever_settings(retriever)
                    )
                    index_hashes[identity_key] = logical_index_hash
                    event["index_hash"] = logical_index_hash
                    event["duration_ms"] = _elapsed_ms(started_ns)
                    build_ms[id(retriever)] = _event_duration_ms(event)
                    events.append(event)
                    continue

            try:
                retriever.index(chunks)
                indexed_chunks, vectors, dimension, metadata = retriever._cache_export()
                try:
                    write_dense_index_artifact(
                        self._cache_dir,
                        index_hash=storage_index_hash,
                        chunk_hash=chunk_hash,
                        retriever_identity=identity,
                        chunks=indexed_chunks,
                        vectors=vectors,
                        dimension=dimension,
                        model_id=metadata.model_id,
                        requested_revision=metadata.requested_revision,
                        resolved_revision=metadata.resolved_revision,
                        max_bytes=self._cache_max_bytes,
                    )
                except CacheCapacityError:
                    event["status"] = "skipped"
                    event["reason"] = "cache artifact exceeds configured capacity"
            except RetrievalLabError:
                raise
            except Exception as exc:
                raise EvaluationError(
                    f"dense retriever {retriever.name!r} cache rebuild failed"
                ) from exc
            logical_index_hash = dense_index_hash(
                chunk_hash, _retriever_settings(retriever)
            )
            index_hashes[identity_key] = logical_index_hash
            event["index_hash"] = logical_index_hash
            event["duration_ms"] = _elapsed_ms(started_ns)
            build_ms[id(retriever)] = _event_duration_ms(event)
            events.append(event)
        return index_hashes, events, build_ms

    def _evaluate_query(
        self,
        retriever: BaseRetriever,
        query: EvaluationQuery,
        *,
        max_k: int,
        latency_samples: list[float],
        failure_count: list[int],
    ) -> QueryEvaluation:
        started_ns = perf_counter_ns()
        try:
            raw_results = retriever.search(query.query, max_k)
        except Exception:
            _elapsed_ms(started_ns)
            failure_count[0] += 1
            raise
        ranked_chunks = stable_rank_results(raw_results, top_k=max_k)
        search_latency_ms = _elapsed_ms(started_ns)
        latency_samples.append(search_latency_ms)
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
        if self._relevance_level == "document":
            return evaluate_cutoff_rankings(
                query_id=query.id,
                retrieved_ids=retrieved_ids,
                retrieved_ids_by_cutoff={
                    cutoff: tuple(
                        item.document_id
                        for item in collapse_to_documents(ranked_chunks[:cutoff])
                    )
                    for cutoff in self._top_k
                },
                relevance_grades=self._relevance_grades_by_query[query.id],
                top_k=self._top_k,
                search_latency_ms=search_latency_ms,
            )
        return evaluate_ranking(
            query_id=query.id,
            retrieved_ids=retrieved_ids,
            relevance_grades=self._relevance_grades_by_query[query.id],
            top_k=self._top_k,
            search_latency_ms=search_latency_ms,
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


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigurationError("seed must be a non-negative integer")
    return seed


def _validate_cache_max_bytes(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError("cache_max_bytes must be a positive integer")
    return value


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _elapsed_ms(started_ns: int) -> float:
    elapsed = perf_counter_ns() - started_ns
    # A monotonic clock should never go backwards, but injected clocks in tests
    # can be imperfect. Do not emit a negative runtime value.
    return max(0.0, elapsed / 1_000_000.0)


def _event_duration_ms(event: Mapping[str, JSONValue]) -> float:
    value = event.get("duration_ms", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _index_size_bytes(retriever: BaseRetriever) -> int | None:
    """Return a deterministic size for indexes owned by built-in retrievers.

    Custom ``BaseRetriever`` implementations deliberately have no size probe:
    arbitrary attribute access could execute provider code and is outside the
    retriever contract.
    """

    if type(retriever) is KeywordRetriever:
        return retriever.index_size_bytes
    if type(retriever) is BM25Retriever:
        return retriever.index_size_bytes
    if type(retriever) is DenseRetriever:
        dimension = retriever._dimension
        if dimension is None:
            return None
        return len(retriever._vectors) * dimension * 8
    if type(retriever) is HybridRetriever:
        sizes: list[int] = []
        seen: set[int] = set()
        for source in retriever._sources:
            if id(source) in seen:
                continue
            seen.add(id(source))
            size = _index_size_bytes(source)
            if size is not None:
                sizes.append(size)
        return sum(sizes) if sizes else None
    return None


def _cached_dense_build_ms(
    retriever: BaseRetriever,
    build_ms_by_id: Mapping[int, float],
) -> float:
    """Attribute cached Dense preparation to each top-level strategy using it."""

    if isinstance(retriever, DenseRetriever):
        return build_ms_by_id.get(id(retriever), 0.0)
    if not isinstance(retriever, HybridRetriever):
        return 0.0

    duration_ms = 0.0
    seen: set[int] = set()
    for source in retriever._sources:
        if not isinstance(source, DenseRetriever) or id(source) in seen:
            continue
        seen.add(id(source))
        duration_ms += build_ms_by_id.get(id(source), 0.0)
    return duration_ms


def _optional_dependency_versions(
    retrievers: Sequence[BaseRetriever],
) -> dict[str, JSONValue]:
    """Read metadata versions without importing optional heavy modules."""

    used_dense = any(
        (isinstance(retriever, DenseRetriever) and retriever.uses_default_backend)
        or (
            isinstance(retriever, HybridRetriever)
            and any(
                isinstance(source, DenseRetriever) and source.uses_default_backend
                for source in retriever._sources
            )
        )
        for retriever in retrievers
    )
    if not used_dense:
        return {}
    try:
        version = importlib_metadata.version("sentence-transformers")
    except importlib_metadata.PackageNotFoundError:
        return {}
    return {"sentence-transformers": version}


def _runtime_manifest(
    *,
    retrievers: Sequence[BaseRetriever],
    started_at: str,
    build_ms: Mapping[str, float],
    index_sizes_bytes: Mapping[str, int],
    cache_events: Sequence[Mapping[str, JSONValue]],
) -> dict[str, JSONValue]:
    """Create non-deterministic environment and runtime observations."""

    try:
        retrieval_lab_version = importlib_metadata.version("retrieval-lab")
    except importlib_metadata.PackageNotFoundError:
        retrieval_lab_version = "0.1.0.dev0"
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    dependencies = _optional_dependency_versions(retrievers)
    return {
        "build_ms": {name: build_ms[name] for name in sorted(build_ms)},
        "cache_events": [
            {key: event[key] for key in sorted(event)} for event in cache_events
        ],
        "dependencies": dependencies,
        "finished_at_utc": _utc_timestamp(),
        "index_sizes_bytes": {
            name: index_sizes_bytes[name] for name in sorted(index_sizes_bytes)
        },
        "os": {"machine": machine, "release": release, "system": system},
        "python_version": platform.python_version(),
        "retrieval_lab_version": retrieval_lab_version,
        "started_at_utc": started_at,
    }


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
    seed: int = 42,
    chunk_hash: str | None = None,
    index_hashes: Mapping[str, JSONValue] | None = None,
    cache_events: Sequence[Mapping[str, JSONValue]] = (),
    runtime: Mapping[str, JSONValue] | None = None,
    config_settings: Mapping[str, JSONValue] | None = None,
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
        retriever.name: _retriever_settings(retriever) for retriever in retrievers
    }
    normalized_index_hashes = (
        dict(index_hashes)
        if index_hashes is not None
        else _dense_index_hashes(retrievers, chunk_hash=chunk_hash)
    )
    run_payload: dict[str, JSONValue] = {
        "chunk_hash": chunk_hash,
        "dataset_hash": dataset_hash,
        "index_hashes": {
            key: normalized_index_hashes[key] for key in sorted(normalized_index_hashes)
        },
        "metric_version": 1,
        "relevance_level": relevance_level,
        "retrievers": _json_values(retriever_names),
        "retriever_settings": retriever_settings,
        "seed": seed,
        "top_k": _json_values(top_k),
    }
    quality_gate_policy_hash: str | None = None
    if config_settings is not None:
        raw_policy = config_settings.get("quality_gates", [])
        if raw_policy:
            quality_gate_policy_hash = content_hash(raw_policy)
            run_payload["quality_gate_policy_hash"] = quality_gate_policy_hash
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
        "seed": seed,
        "top_k": _json_values(top_k),
    }
    if quality_gate_policy_hash is not None:
        manifest["quality_gate_policy_hash"] = quality_gate_policy_hash
    manifest["index_hashes"] = {
        key: normalized_index_hashes[key] for key in sorted(normalized_index_hashes)
    }
    if cache_events:
        # Runtime cache observations are intentionally outside run_payload, so
        # hit/miss/corruption never changes the reproducible run ID.
        manifest["runtime"] = {
            "cache_events": [
                {key: event[key] for key in sorted(event)} for event in cache_events
            ]
        }
    if runtime is not None:
        manifest["runtime"] = plain_json(runtime)
    if config_settings is not None:
        manifest["config"] = plain_json(config_settings)
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


def _dense_retriever_entries(
    retrievers: Sequence[BaseRetriever],
) -> tuple[tuple[str, DenseRetriever], ...]:
    """Find Dense instances with stable, collision-free strategy paths."""

    found: list[tuple[str, DenseRetriever]] = []
    seen: set[int] = set()
    for retriever in retrievers:
        if isinstance(retriever, DenseRetriever) and id(retriever) not in seen:
            found.append((retriever.name, retriever))
            seen.add(id(retriever))
    for retriever in retrievers:
        if not isinstance(retriever, HybridRetriever):
            continue
        for source_name, source in zip(
            retriever._source_names, retriever._sources, strict=True
        ):
            if isinstance(source, DenseRetriever) and id(source) not in seen:
                found.append((f"{retriever.name}.sources.{source_name}", source))
                seen.add(id(source))
    return tuple(found)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _retriever_settings(retriever: BaseRetriever) -> dict[str, JSONValue]:
    """Read and normalize a custom retriever's settings at the API boundary."""

    try:
        raw_settings = retriever.settings
    except RetrievalLabError:
        raise
    except Exception as exc:
        raise RetrieverContractError(
            f"retriever {retriever.name!r} settings could not be read"
        ) from exc
    try:
        normalized = _normalize_retriever_json(
            raw_settings,
            location=f"retriever {retriever.name!r} settings",
        )
    except RetrieverContractError:
        raise
    except (RecursionError, TypeError, ValueError, OverflowError) as exc:
        raise RetrieverContractError(
            f"retriever {retriever.name!r} settings must be JSON-compatible"
        ) from exc
    if not isinstance(normalized, dict):
        raise RetrieverContractError(
            f"retriever {retriever.name!r} settings must be a mapping"
        )
    return normalized


def _normalize_retriever_json(value: object, *, location: str) -> JSONValue:
    """Copy JSON-compatible settings with deterministic mapping ordering."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RetrieverContractError(f"{location} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        keys = tuple(value)
        if not all(isinstance(key, str) for key in keys):
            raise RetrieverContractError(f"{location} keys must be strings")
        return {
            key: _normalize_retriever_json(value[key], location=f"{location}.{key}")
            for key in sorted(keys)
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalize_retriever_json(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise RetrieverContractError(
        f"{location} contains unsupported JSON value type {type(value).__name__}"
    )


def _dense_index_hashes(
    retrievers: Sequence[BaseRetriever],
    *,
    chunk_hash: str,
) -> dict[str, JSONValue]:
    """Compute stable cache identities even when cache persistence is disabled."""

    result: dict[str, JSONValue] = {}
    for identity_key, retriever in _dense_retriever_entries(retrievers):
        settings = _retriever_settings(retriever)
        result[identity_key] = dense_index_hash(chunk_hash, settings)
    return result


def _dense_storage_settings(settings: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Return stable cache-address settings before a lazy model is loaded.

    The resolved revision is validated inside the artifact and participates in
    the final logical index hash, but it cannot address the artifact because a
    lazy backend may not know it until document encoding has completed.
    """

    return {
        key: settings[key] for key in sorted(settings) if key != "resolved_revision"
    }


__all__ = ["EvaluationRunner"]
