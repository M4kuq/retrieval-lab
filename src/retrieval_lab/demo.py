"""Reusable query-comparison models for notebooks and public demos."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from retrieval_lab.domain.json_types import JSONValue
from retrieval_lab.exceptions import EvaluationError, RetrieverContractError
from retrieval_lab.models import SearchResult
from retrieval_lab.retrievers import BaseRetriever

Clock = Callable[[], float]


@dataclass(frozen=True)
class DemoSearchHit:
    """One presentation-safe search hit returned by a demo comparison."""

    document_id: str
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: Mapping[str, JSONValue]

    @classmethod
    def from_result(cls, result: SearchResult) -> DemoSearchHit:
        """Copy one validated ``SearchResult`` into immutable demo data."""

        if not isinstance(result, SearchResult):
            raise RetrieverContractError(
                "demo retrievers must return SearchResult values"
            )
        return cls(
            document_id=result.document_id,
            chunk_id=result.chunk_id,
            text=result.text,
            score=result.score,
            rank=result.rank,
            metadata=MappingProxyType(dict(result.metadata)),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-compatible hit record."""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "metadata": dict(self.metadata),
            "rank": self.rank,
            "score": self.score,
            "text": self.text,
        }


@dataclass(frozen=True)
class DemoRetrieverView:
    """Results and observed search latency for one retriever."""

    retriever: str
    latency_ms: float
    results: tuple[DemoSearchHit, ...]

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-compatible retriever comparison view."""

        return {
            "latency_ms": self.latency_ms,
            "results": [result.to_dict() for result in self.results],
            "retriever": self.retriever,
        }


@dataclass(frozen=True)
class DemoComparison:
    """Side-by-side retrieval results for one query."""

    query: str
    top_k: int
    views: tuple[DemoRetrieverView, ...]

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a deterministic JSON-compatible comparison payload."""

        return {
            "query": self.query,
            "top_k": self.top_k,
            "views": [view.to_dict() for view in self.views],
        }


def compare_retrievers_for_query(
    retrievers: Sequence[BaseRetriever],
    query: str,
    *,
    top_k: int = 5,
    clock: Clock = time.perf_counter,
) -> DemoComparison:
    """Search already-indexed retrievers under one shared query and cutoff."""

    normalized_query = _require_query(query)
    normalized_top_k = _require_top_k(top_k)
    normalized_retrievers = _require_retrievers(retrievers)
    if not callable(clock):
        raise EvaluationError("clock must be callable")

    views: list[DemoRetrieverView] = []
    for retriever in normalized_retrievers:
        start = _read_clock(clock)
        try:
            results = retriever.search(normalized_query, normalized_top_k)
        except (RetrieverContractError, EvaluationError):
            raise
        except Exception as exc:
            raise RetrieverContractError(
                f"retriever {retriever.name!r} failed during demo search: {exc}"
            ) from exc
        end = _read_clock(clock)
        latency_ms = (end - start) * 1000.0
        if latency_ms < 0.0:
            raise EvaluationError("clock moved backwards during demo search")
        validated = _validate_results(
            results,
            retriever_name=retriever.name,
            top_k=normalized_top_k,
        )
        views.append(
            DemoRetrieverView(
                retriever=retriever.name,
                latency_ms=latency_ms,
                results=tuple(DemoSearchHit.from_result(item) for item in validated),
            )
        )
    return DemoComparison(
        query=normalized_query,
        top_k=normalized_top_k,
        views=tuple(views),
    )


def retrieval_metric_explanations() -> Mapping[str, str]:
    """Return concise explanations suitable for notebooks and demo UIs."""

    return MappingProxyType(
        {
            "hit_rate": "Whether at least one relevant item appears within the cutoff.",
            "map": "Mean average precision across queries, rewarding relevant items ranked early.",
            "mrr": "Mean reciprocal rank of the first relevant result.",
            "ndcg": "Normalized discounted cumulative gain, supporting graded relevance.",
            "precision": "Share of retrieved items within the cutoff that are relevant.",
            "recall": "Share of known relevant items recovered within the cutoff.",
        }
    )


def _require_query(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError("query must be a non-empty string")
    return value


def _require_top_k(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvaluationError("top_k must be a positive integer")
    return value


def _require_retrievers(
    values: Sequence[BaseRetriever],
) -> tuple[BaseRetriever, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise EvaluationError("retrievers must be a sequence of BaseRetriever values")
    retrievers = tuple(values)
    if not retrievers:
        raise EvaluationError("retrievers must not be empty")
    names: set[str] = set()
    for position, retriever in enumerate(retrievers):
        if not isinstance(retriever, BaseRetriever):
            raise EvaluationError(
                f"retrievers[{position}] must be a BaseRetriever instance"
            )
        name = retriever.name
        if not isinstance(name, str) or not name.strip():
            raise RetrieverContractError(
                f"retrievers[{position}].name must be a non-empty string"
            )
        if name in names:
            raise EvaluationError(f"retriever names must be unique; duplicate {name!r}")
        names.add(name)
    return retrievers


def _read_clock(clock: Clock) -> float:
    try:
        value = float(clock())
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError("clock must return a finite number") from exc
    if not math.isfinite(value):
        raise EvaluationError("clock must return a finite number")
    return value


def _validate_results(
    values: object,
    *,
    retriever_name: str,
    top_k: int,
) -> tuple[SearchResult, ...]:
    if not isinstance(values, list):
        raise RetrieverContractError(
            f"retriever {retriever_name!r} must return list[SearchResult]"
        )
    if len(values) > top_k:
        raise RetrieverContractError(
            f"retriever {retriever_name!r} returned more than top_k results"
        )
    seen: set[str] = set()
    validated: list[SearchResult] = []
    for position, item in enumerate(values, start=1):
        if not isinstance(item, SearchResult):
            raise RetrieverContractError(
                f"retriever {retriever_name!r} result {position} must be SearchResult"
            )
        if item.rank != position:
            raise RetrieverContractError(
                f"retriever {retriever_name!r} ranks must be contiguous from 1"
            )
        if item.chunk_id in seen:
            raise RetrieverContractError(
                f"retriever {retriever_name!r} returned duplicate chunk ID "
                f"{item.chunk_id!r}"
            )
        seen.add(item.chunk_id)
        validated.append(item)
    return tuple(validated)


__all__ = [
    "Clock",
    "DemoComparison",
    "DemoRetrieverView",
    "DemoSearchHit",
    "compare_retrievers_for_query",
    "retrieval_metric_explanations",
]
