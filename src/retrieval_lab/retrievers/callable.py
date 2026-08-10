"""Provider-independent synchronous retriever adapters and evaluation."""

from __future__ import annotations

import builtins
import math
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from time import perf_counter_ns
from typing import Protocol, runtime_checkable

from retrieval_lab.datasets import EvaluationDataset
from retrieval_lab.domain import (
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
    evaluate_ranking,
    normalize_top_k,
)
from retrieval_lab.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    RetrieverContractError,
)


@dataclass(frozen=True, slots=True)
class RetrievedItem:
    """One item in a provider or vector-database ranking.

    The sequence returned by a retriever is the ranking order.  ``score`` and
    ``rank`` are optional because many existing search APIs expose only IDs.
    When ranks are present, :class:`CallableRetriever` requires every item to
    provide contiguous one-based ranks in sequence order.
    """

    id: str
    parent_document_id: str | None = None
    score: float | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise RetrieverContractError("RetrievedItem.id must be a non-empty string")
        if self.parent_document_id is not None and (
            not isinstance(self.parent_document_id, str)
            or not self.parent_document_id.strip()
        ):
            raise RetrieverContractError(
                "RetrievedItem.parent_document_id must be a non-empty string or None"
            )
        if self.score is not None:
            if (
                isinstance(self.score, bool)
                or not isinstance(self.score, (int, float))
                or not math.isfinite(float(self.score))
            ):
                raise RetrieverContractError(
                    "RetrievedItem.score must be a finite number or None"
                )
            object.__setattr__(self, "score", float(self.score))
        if self.rank is not None and (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank <= 0
        ):
            raise RetrieverContractError(
                "RetrievedItem.rank must be a positive integer or None"
            )


@runtime_checkable
class Retriever(Protocol):
    """Minimal synchronous retrieval protocol for external search systems."""

    @property
    def name(self) -> str:
        """Return the stable retriever name."""

    def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedItem]:
        """Return a best-first sequence of retrieved items."""


class _SearchCallable(Protocol):
    def __call__(self, query: str, *, top_k: int) -> Sequence[RetrievedItem]:
        """Call a user-owned synchronous search function."""


class CallableRetriever:
    """Adapt a synchronous callable to the provider-independent protocol."""

    def __init__(self, name: str, callable: _SearchCallable) -> None:
        """Create an adapter without invoking the callable."""

        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("CallableRetriever.name must be non-empty")
        if not builtins.callable(callable):
            raise ConfigurationError("CallableRetriever.callable must be callable")
        self._name = name
        self._callable = callable

    @property
    def name(self) -> str:
        """Return the configured stable adapter name."""

        return self._name

    def retrieve(self, query: str, *, top_k: int) -> tuple[RetrievedItem, ...]:
        """Invoke and validate one ranking, translating low-level failures."""

        if not isinstance(query, str):
            raise RetrieverContractError("retrieval query must be a string")
        _validate_top_k(top_k)
        try:
            raw_items = self._callable(query, top_k=top_k)
        except Exception as exc:
            raise RetrieverContractError(
                f"retriever {self.name!r} callable failed"
            ) from exc
        return _validate_items(raw_items, top_k=top_k)


def evaluate_retrievers(
    *,
    dataset: EvaluationDataset,
    retrievers: Mapping[str, Retriever],
    top_k: Sequence[int] = (1, 3, 5, 10),
    clock: Callable[[], int] | None = None,
) -> EvaluationResult:
    """Evaluate synchronous external retrievers without a corpus or index.

    Each retriever is called once per query at ``max(top_k)``.  The ranking
    order returned by the external system is preserved; scores are evidence
    only and are never used to reorder results.
    """

    if not isinstance(dataset, EvaluationDataset):
        raise DatasetValidationError("dataset must be an EvaluationDataset")
    normalized_top_k = normalize_top_k(top_k)
    normalized_retrievers = _validate_retrievers(retrievers)
    clock_fn = perf_counter_ns if clock is None else _validate_clock(clock)
    max_k = max(normalized_top_k)
    metrics: dict[str, RetrieverMetrics] = {}
    query_results: dict[str, tuple[QueryEvaluation, ...]] = {}
    latency: dict[str, LatencyStats] = {}
    rankings_by_name: dict[str, list[dict[str, JSONValue]]] = {}
    started_at = _utc_timestamp()

    for name, retriever in normalized_retrievers:
        evaluations: list[QueryEvaluation] = []
        latency_samples: list[float] = []
        rankings_payload: list[dict[str, JSONValue]] = []
        for query in dataset.queries:
            started_ns = _read_clock(clock_fn)
            try:
                raw_items = retriever.retrieve(query.query, top_k=max_k)
                items = _validate_items(raw_items, top_k=max_k)
            except Exception as exc:
                raise RetrieverContractError(
                    f"retriever {name!r} failed for query {query.id!r}"
                ) from exc
            finished_ns = _read_clock(clock_fn)
            if finished_ns < started_ns:
                raise ConfigurationError("clock readings must be monotonic")
            elapsed_ms = float(finished_ns - started_ns) / 1_000_000.0
            latency_samples.append(elapsed_ms)
            retrieved_ids = _evaluation_ids(
                items,
                relevance_level=dataset.relevance_level,
            )
            evaluations.append(
                evaluate_ranking(
                    query_id=query.id,
                    retrieved_ids=retrieved_ids,
                    relevance_grades=dataset.relevance_grades_by_query[query.id],
                    top_k=normalized_top_k,
                    search_latency_ms=elapsed_ms,
                )
            )
            rankings_payload.append(
                {
                    "items": [_item_payload(item) for item in items],
                    "query_id": query.id,
                }
            )
        stats = LatencyStats.from_samples(latency_samples)
        if stats.warnings:
            evaluations = [
                _with_warnings(evaluation, stats.warnings) for evaluation in evaluations
            ]
        query_evaluations = tuple(evaluations)
        metrics[name] = aggregate_metrics(query_evaluations, normalized_top_k)
        query_results[name] = query_evaluations
        latency[name] = stats
        rankings_by_name[name] = rankings_payload

    manifest, run_id = _build_manifest(
        dataset=dataset,
        retriever_names=tuple(name for name, _ in normalized_retrievers),
        top_k=normalized_top_k,
        rankings_by_name=rankings_by_name,
        started_at=started_at,
    )
    return EvaluationResult(
        run_id=run_id,
        metrics=metrics,
        query_results=query_results,
        manifest=manifest,
        latency=latency,
    )


def _validate_retrievers(
    retrievers: Mapping[str, Retriever],
) -> tuple[tuple[str, Retriever], ...]:
    if not isinstance(retrievers, Mapping) or not retrievers:
        raise ConfigurationError("retrievers must be a non-empty mapping")
    normalized: list[tuple[str, Retriever]] = []
    for raw_name, retriever in retrievers.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ConfigurationError("retriever mapping keys must be non-empty strings")
        try:
            name = retriever.name
            retrieve = retriever.retrieve
        except Exception as exc:
            raise ConfigurationError(
                f"retriever {raw_name!r} must implement the Retriever protocol"
            ) from exc
        if not builtins.callable(retrieve):
            raise ConfigurationError(
                f"retriever {raw_name!r} must implement the Retriever protocol"
            )
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("retriever names must be non-empty strings")
        if name != raw_name:
            raise ConfigurationError(
                f"retriever mapping key {raw_name!r} does not match name {name!r}"
            )
        normalized.append((name, retriever))
    normalized.sort(key=lambda item: item[0])
    return tuple(normalized)


def _validate_clock(clock: Callable[[], int]) -> Callable[[], int]:
    if not callable(clock):
        raise ConfigurationError("clock must be callable")
    return clock


def _read_clock(clock: Callable[[], int]) -> int:
    try:
        value = clock()
    except Exception as exc:
        raise ConfigurationError("clock callable failed") from exc
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError("clock must return an integer nanosecond value")
    return value


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise RetrieverContractError("top_k must be a positive integer")


def _validate_items(
    raw_items: object,
    *,
    top_k: int,
) -> tuple[RetrievedItem, ...]:
    _validate_top_k(top_k)
    if isinstance(raw_items, (str, bytes)) or not isinstance(raw_items, Sequence):
        raise RetrieverContractError(
            "retriever must return a sequence of RetrievedItem values"
        )
    try:
        items = tuple(raw_items)
    except Exception as exc:
        raise RetrieverContractError(
            "retriever returned an unreadable result sequence"
        ) from exc
    if len(items) > top_k:
        raise RetrieverContractError(
            f"retriever returned {len(items)} results for top_k={top_k}"
        )
    if not all(isinstance(item, RetrievedItem) for item in items):
        raise RetrieverContractError(
            "retriever must return a sequence of RetrievedItem values"
        )
    identifiers = [item.id for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise RetrieverContractError("retriever returned duplicate item IDs")
    ranks = [item.rank for item in items]
    if any(rank is not None for rank in ranks):
        if any(rank is None for rank in ranks):
            raise RetrieverContractError(
                "retriever ranks must be specified for every returned item"
            )
        expected = tuple(range(1, len(items) + 1))
        if tuple(ranks) != expected:
            raise RetrieverContractError(
                "retriever ranks must be contiguous, one-based, and match "
                "sequence order"
            )
    return items


def _evaluation_ids(
    items: Sequence[RetrievedItem],
    *,
    relevance_level: str,
) -> tuple[str, ...]:
    identifiers: list[str] = []
    seen: set[str] = set()
    for item in items:
        identifier = (
            item.id
            if relevance_level == "chunk"
            else item.parent_document_id or item.id
        )
        if identifier in seen:
            if relevance_level == "chunk":
                raise RetrieverContractError(
                    f"duplicate chunk evaluation ID: {identifier!r}"
                )
            continue
        seen.add(identifier)
        identifiers.append(identifier)
    return tuple(identifiers)


def _item_payload(item: RetrievedItem) -> dict[str, JSONValue]:
    return {
        "id": item.id,
        "parent_document_id": item.parent_document_id,
        "rank": item.rank,
        "score": item.score,
    }


def _with_warnings(
    evaluation: QueryEvaluation,
    warnings: Sequence[str],
) -> QueryEvaluation:
    return QueryEvaluation(
        query_id=evaluation.query_id,
        retrieved_ids=evaluation.retrieved_ids,
        metrics_by_cutoff=evaluation.metrics_by_cutoff,
        search_latency_ms=evaluation.search_latency_ms,
        warnings=tuple(warnings),
    )


def _build_manifest(
    *,
    dataset: EvaluationDataset,
    retriever_names: Sequence[str],
    top_k: Sequence[int],
    rankings_by_name: Mapping[str, Sequence[Mapping[str, JSONValue]]],
    started_at: str,
) -> tuple[dict[str, JSONValue], str]:
    grades = dataset.relevance_grades_by_query
    dataset_hash = content_hash(dataset_payload(dataset.queries, grades))
    ranking_payload: dict[str, JSONValue] = {}
    for name in sorted(rankings_by_name):
        records: list[JSONValue] = []
        records.extend(
            dict(record)
            for record in sorted(
                rankings_by_name[name], key=lambda record: str(record["query_id"])
            )
        )
        ranking_payload[name] = records
    rankings_hash = content_hash(ranking_payload)
    query_ids: list[JSONValue] = []
    query_ids.extend(sorted(query.id for query in dataset.queries))
    run_payload: dict[str, JSONValue] = {
        "dataset_hash": dataset_hash,
        "evaluation_mode": "callable",
        "metric_version": 1,
        "query_ids": query_ids,
        "relevance_level": dataset.relevance_level,
        "retrievers": list(retriever_names),
        "retrieved_rankings_hash": rankings_hash,
        "top_k": list(top_k),
    }
    run_id = content_hash(run_payload)
    manifest: dict[str, JSONValue] = {
        **run_payload,
        "query_count": len(dataset.queries),
        "runtime": _runtime_manifest(started_at),
    }
    return manifest, run_id


def _runtime_manifest(started_at: str) -> dict[str, JSONValue]:
    try:
        version = importlib_metadata.version("retrieval-lab")
    except importlib_metadata.PackageNotFoundError:
        version = "0.1.0.dev0"
    return {
        "finished_at_utc": _utc_timestamp(),
        "os": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "python_version": platform.python_version(),
        "retrieval_lab_version": version,
        "started_at_utc": started_at,
    }


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "CallableRetriever",
    "RetrievedItem",
    "Retriever",
    "evaluate_retrievers",
]
