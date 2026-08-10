"""Deterministic reciprocal-rank-fusion retrieval over shared chunks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

from retrieval_lab.domain import JSONValue, SearchResult
from retrieval_lab.evaluation.ranking import validate_search_results
from retrieval_lab.exceptions import RetrievalLabError, RetrieverContractError
from retrieval_lab.models import Chunk
from retrieval_lab.retrievers.base import BaseRetriever


class HybridRetriever(BaseRetriever):
    """Fuse rankings from two or more indexed retrievers with RRF.

    Every source receives the same shared chunk sequence and the same
    ``candidate_k``. Fusion uses only one-based source ranks; source scores are
    deliberately not part of the calculation.
    """

    def __init__(
        self,
        sources: Sequence[BaseRetriever],
        *,
        rrf_k: int = 60,
        candidate_k: int = 100,
    ) -> None:
        """Create an unindexed hybrid retriever with validated sources."""
        if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
            raise RetrieverContractError(
                "sources must be a sequence of BaseRetriever implementations"
            )
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
            raise RetrieverContractError("rrf_k must be a positive integer")
        if (
            isinstance(candidate_k, bool)
            or not isinstance(candidate_k, int)
            or candidate_k <= 0
        ):
            raise RetrieverContractError("candidate_k must be a positive integer")

        normalized_sources = tuple(sources)
        if len(normalized_sources) < 2:
            raise RetrieverContractError(
                "hybrid retrieval requires at least two sources"
            )

        names: list[str] = []
        for position, source in enumerate(normalized_sources):
            if not isinstance(source, BaseRetriever):
                raise RetrieverContractError(
                    f"sources[{position}] must be a BaseRetriever implementation"
                )
            if isinstance(source, HybridRetriever):
                raise RetrieverContractError(
                    "HybridRetriever sources are not allowed; nested hybrid "
                    "composition could be recursive"
                )
            try:
                source_name = source.name
            except Exception as exc:
                raise RetrieverContractError(
                    f"sources[{position}] could not provide a name"
                ) from exc
            if not isinstance(source_name, str) or not source_name.strip():
                raise RetrieverContractError(
                    f"sources[{position}] name must be a non-empty string"
                )
            names.append(source_name)

        if len(set(names)) != len(names):
            raise RetrieverContractError("hybrid source names must be unique")

        self._sources = normalized_sources
        self._source_names = tuple(names)
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k
        self._indexed = False
        self._indexed_chunks: tuple[Chunk, ...] | None = None

    @property
    def name(self) -> str:
        """Return the stable hybrid strategy name."""
        return "hybrid"

    @property
    def settings(self) -> Mapping[str, JSONValue]:
        """Return source-order-independent, JSON-compatible RRF settings."""
        source_entries: list[JSONValue] = []
        for source_name, source in sorted(
            zip(self._source_names, self._sources, strict=True),
            key=lambda item: item[0],
        ):
            try:
                source_settings = source.settings
            except RetrievalLabError:
                raise
            except Exception as exc:
                raise RetrieverContractError(
                    f"hybrid source {source_name!r} settings could not be read"
                ) from exc
            try:
                normalized_settings = _normalize_json_mapping(source_settings)
            except RetrieverContractError:
                raise
            except Exception as exc:
                raise RetrieverContractError(
                    f"hybrid source {source_name!r} settings must be JSON-compatible"
                ) from exc
            source_entries.append(
                {
                    "name": source_name,
                    "settings": normalized_settings,
                }
            )
        return {
            "candidate_k": self._candidate_k,
            "name": self.name,
            "rrf_k": self._rrf_k,
            "sources": source_entries,
            "type": "hybrid",
        }

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Index every source with the same supplied chunks."""
        indexed = _validate_shared_chunks(chunks)
        if self._indexed and self._indexed_chunks == indexed:
            return
        self._indexed = False
        for source in self._sources:
            source.index(chunks)
        self._indexed = True
        self._indexed_chunks = indexed

    def _index_sources_once(
        self,
        chunks: Sequence[Chunk],
        indexed_retriever_ids: set[int],
    ) -> None:
        """Index shared sources once when a runner exposes them separately."""

        indexed = _validate_shared_chunks(chunks)
        if self._indexed and self._indexed_chunks == indexed:
            return
        self._indexed = False
        for source in self._sources:
            if id(source) not in indexed_retriever_ids:
                source.index(chunks)
                indexed_retriever_ids.add(id(source))
        self._indexed = True
        self._indexed_chunks = indexed

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return the top fused candidates, sorted by RRF score then chunk ID."""
        if not self._indexed:
            raise RetrieverContractError(
                "hybrid retriever is not indexed; call index(chunks) before search"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RetrieverContractError(
                "top_k must be a positive integer; for example, top_k=5"
            )
        if self._candidate_k < top_k:
            raise RetrieverContractError(
                "candidate_k must be greater than or equal to top_k"
            )
        if not isinstance(query, str):
            raise RetrieverContractError("query must be a string")

        rankings: dict[str, tuple[SearchResult, ...]] = {}
        for source_name, source in zip(self._source_names, self._sources, strict=True):
            try:
                raw_results = source.search(query, self._candidate_k)
                if isinstance(raw_results, (str, bytes)) or not isinstance(
                    raw_results, Sequence
                ):
                    raise RetrieverContractError(
                        "source search must return a sequence of SearchResult values"
                    )
                if not all(isinstance(result, SearchResult) for result in raw_results):
                    raise RetrieverContractError(
                        "source search must return a sequence of SearchResult values"
                    )
                rankings[source_name] = validate_search_results(
                    raw_results,
                    top_k=self._candidate_k,
                )
            except RetrievalLabError:
                raise
            except Exception as exc:
                raise RetrieverContractError(
                    f"hybrid source {source_name!r} returned an invalid ranking"
                ) from exc

        fused: dict[str, float] = {}
        candidates: dict[str, SearchResult] = {}
        # Iterating names in sorted order makes both floating-point accumulation
        # and the copied candidate values independent of constructor source order.
        for source_name in sorted(rankings):
            for result in rankings[source_name]:
                fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + (
                    1.0 / (self._rrf_k + result.rank)
                )
                existing = candidates.get(result.chunk_id)
                if existing is not None and not _same_chunk_payload(existing, result):
                    raise RetrieverContractError(
                        "hybrid sources returned conflicting payloads for "
                        f"chunk_id {result.chunk_id!r}"
                    )
                candidates.setdefault(result.chunk_id, result)

        ordered_ids = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))
        return [
            replace(
                candidates[chunk_id],
                score=fused[chunk_id],
                rank=rank,
            )
            for rank, chunk_id in enumerate(ordered_ids[:top_k], start=1)
        ]


def _normalize_json_mapping(value: object) -> dict[str, JSONValue]:
    """Defensively copy and deterministically order a JSON-compatible mapping."""
    if not isinstance(value, Mapping):
        raise RetrieverContractError("retriever settings must be a mapping")
    keys = tuple(value.keys())
    if not all(isinstance(key, str) for key in keys):
        raise RetrieverContractError("retriever settings keys must be strings")
    return {
        key: _normalize_json_value(value[key], location=f"settings.{key}")
        for key in sorted(keys)
    }


def _same_chunk_payload(left: SearchResult, right: SearchResult) -> bool:
    """Return whether two source results describe the same shared chunk."""

    return (
        left.document_id == right.document_id
        and left.text == right.text
        and left.metadata == right.metadata
    )


def _normalize_json_value(value: object, *, location: str) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RetrieverContractError(f"{location} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        normalized = _normalize_json_mapping(value)
        return normalized
    if isinstance(value, list):
        return [
            _normalize_json_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise RetrieverContractError(
        f"{location} contains unsupported JSON value type {type(value).__name__}"
    )


def _validate_shared_chunks(chunks: Sequence[Chunk]) -> tuple[Chunk, ...]:
    """Validate the shared index boundary before invoking any source."""

    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise RetrieverContractError("chunks must be a sequence of Chunk records")
    indexed = tuple(chunks)
    seen: set[str] = set()
    for position, chunk in enumerate(indexed):
        if not isinstance(chunk, Chunk):
            raise RetrieverContractError(f"chunks[{position}] must be a Chunk record")
        if chunk.id in seen:
            raise RetrieverContractError(
                f"hybrid index received duplicate chunk identifier: {chunk.id}"
            )
        seen.add(chunk.id)
    return indexed


__all__ = ["HybridRetriever"]
