"""Provider-independent asynchronous retriever adapters and evaluation."""

from __future__ import annotations

import asyncio
import builtins
import math
from collections.abc import Awaitable, Mapping, Sequence
from time import perf_counter_ns
from typing import Protocol, cast

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
    evaluate_cutoff_rankings,
    evaluate_ranking,
    normalize_top_k,
)
from retrieval_lab.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    RetrieverContractError,
)
from retrieval_lab.retrievers.callable import (
    RetrievedItem,
    _build_manifest,
    _evaluation_ids,
    _item_payload,
    _normalize_scored_items,
    _ranking_signature,
    _utc_timestamp,
    _validate_chunk_hash,
    _validate_items,
    _validate_top_k,
    _with_warnings,
)


class AsyncRetriever(Protocol):
    """Minimal asynchronous retrieval protocol for external search systems."""

    @property
    def name(self) -> str:
        """Return the stable retriever name."""

    async def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedItem]:
        """Return a best-first sequence of retrieved items."""


class _AsyncSearchCallable(Protocol):
    def __call__(
        self,
        query: str,
        *,
        top_k: int,
    ) -> Awaitable[Sequence[RetrievedItem]]:
        """Call a user-owned asynchronous search function."""


class AsyncCallableRetriever:
    """Adapt an asynchronous callable to the provider-independent protocol.

    The callable is invoked only when :meth:`retrieve` is awaited.  A
    :class:`asyncio.CancelledError` from the caller is deliberately allowed to
    propagate so cooperative cancellation remains intact; ordinary callable
    failures are translated to :class:`RetrieverContractError` with their
    original exception as ``__cause__``.
    """

    def __init__(self, name: str, callable: _AsyncSearchCallable) -> None:
        """Create an adapter without invoking the callable."""

        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("AsyncCallableRetriever.name must be non-empty")
        if not builtins.callable(callable):
            raise ConfigurationError("AsyncCallableRetriever.callable must be callable")
        self._name = name
        self._callable = callable

    @property
    def name(self) -> str:
        """Return the configured stable adapter name."""

        return self._name

    async def retrieve(self, query: str, *, top_k: int) -> tuple[RetrievedItem, ...]:
        """Invoke and validate one ranking, translating low-level failures."""

        if not isinstance(query, str):
            raise RetrieverContractError("retrieval query must be a string")
        _validate_top_k(top_k)
        try:
            raw_items = await self._callable(query, top_k=top_k)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RetrieverContractError(
                f"retriever {self.name!r} callable failed"
            ) from exc
        return _validate_items(raw_items, top_k=top_k)


async def evaluate_async_retrievers(
    *,
    dataset: EvaluationDataset,
    retrievers: Mapping[str, AsyncRetriever],
    top_k: Sequence[int] = (1, 3, 5, 10),
    concurrency: int = 1,
    repetitions: int = 1,
    timeout_s: float | None = None,
    chunk_hash: str | None = None,
) -> EvaluationResult:
    """Evaluate asynchronous retrievers without a corpus or index.

    Each retriever/query/repetition call receives ``max(top_k)`` and is
    bounded by the shared ``concurrency`` limit.  The first repetition is the
    ranking used for quality metrics; later repetitions must have identical
    ranking identity (item IDs, parent IDs, ranks, and order), while scores
    may vary.  Returned results are ordered by retriever name and dataset query
    order, regardless of completion order.

    This function is an async entry point and must be awaited by the caller.
    It does not create or replace an event loop.  If the awaiting task is
    externally cancelled, worker tasks are cleaned up and
    :class:`asyncio.CancelledError` is re-raised rather than translated.
    """

    if not isinstance(dataset, EvaluationDataset):
        raise DatasetValidationError("dataset must be an EvaluationDataset")
    normalized_top_k = normalize_top_k(top_k)
    normalized_retrievers = _validate_async_retrievers(retrievers)
    normalized_concurrency = _validate_positive_int(concurrency, "concurrency")
    normalized_repetitions = _validate_positive_int(repetitions, "repetitions")
    normalized_timeout = _validate_timeout(timeout_s)
    normalized_chunk_hash = _validate_chunk_hash(chunk_hash)

    max_k = max(normalized_top_k)
    completed: dict[tuple[str, int, int], tuple[tuple[RetrievedItem, ...], float]] = {}
    parent_task = asyncio.current_task()
    started_at = _utc_timestamp()

    async def run_one(
        name: str,
        retriever: AsyncRetriever,
        query_index: int,
        repetition: int,
    ) -> None:
        query = dataset.queries[query_index]
        started_ns = perf_counter_ns()
        try:
            if normalized_timeout is None:
                raw_items = await retriever.retrieve(query.query, top_k=max_k)
            else:
                async with asyncio.timeout(normalized_timeout):
                    raw_items = await retriever.retrieve(query.query, top_k=max_k)
        except asyncio.CancelledError as exc:
            if (
                parent_task is not None
                and parent_task.cancelling() > parent_cancellation_count
            ):
                raise
            raise RetrieverContractError(
                f"retriever {name!r} was cancelled for query {query.id!r}"
            ) from exc
        except TimeoutError as exc:
            raise RetrieverContractError(
                f"retriever {name!r} timed out for query {query.id!r}"
            ) from exc
        except RetrieverContractError as exc:
            raise RetrieverContractError(
                f"retriever {name!r} failed for query {query.id!r}: {exc}"
            ) from exc
        except Exception as exc:
            raise RetrieverContractError(
                f"retriever {name!r} failed for query {query.id!r}"
            ) from exc
        finished_ns = perf_counter_ns()
        if finished_ns < started_ns:
            raise ConfigurationError("clock readings must be monotonic")
        try:
            items = _normalize_scored_items(_validate_items(raw_items, top_k=max_k))
        except RetrieverContractError as exc:
            raise RetrieverContractError(
                f"retriever {name!r} violated the result contract for query "
                f"{query.id!r}"
            ) from exc
        completed[(name, query_index, repetition)] = (
            items,
            float(finished_ns - started_ns) / 1_000_000.0,
        )

    jobs = iter(
        (name, retriever, query_index, repetition)
        for name, retriever in normalized_retrievers
        for query_index in range(len(dataset.queries))
        for repetition in range(normalized_repetitions)
    )

    async def worker() -> None:
        while True:
            try:
                job = next(jobs)
            except StopIteration:
                return
            await run_one(*job)

    job_count = (
        len(normalized_retrievers) * len(dataset.queries) * normalized_repetitions
    )
    worker_count = min(normalized_concurrency, job_count)

    worker_tasks: list[asyncio.Task[None]] = []
    parent_cancellation_count = (
        parent_task.cancelling() if parent_task is not None else 0
    )
    task_error: BaseException | None = None
    try:
        worker_tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]
        await asyncio.gather(*worker_tasks)
    except asyncio.CancelledError:
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        raise
    except Exception as exc:
        # A provider can raise while unwinding its cancellation cleanup.  A
        # gather of the fixed worker set lets us clean up those workers while
        # retaining the caller's cancellation request, rather than allowing a
        # cleanup error to replace it as a TaskGroup ExceptionGroup.
        cancellation_requested = (
            parent_task is not None
            and parent_task.cancelling() > parent_cancellation_count
        )
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        if cancellation_requested:
            raise asyncio.CancelledError from None
        task_error = _first_task_error(exc)

    if task_error is not None:
        if isinstance(task_error, RetrieverContractError):
            cause = task_error.__cause__
            if cause is not None:
                raise task_error from cause
            raise task_error from None
        if isinstance(task_error, ConfigurationError):
            raise task_error from None
        raise RetrieverContractError("asynchronous retrieval failed") from task_error

    rankings_by_name: dict[str, list[dict[str, JSONValue]]] = {}
    metrics: dict[str, RetrieverMetrics] = {}
    query_results: dict[str, tuple[QueryEvaluation, ...]] = {}
    latency: dict[str, LatencyStats] = {}
    for name, _retriever in normalized_retrievers:
        evaluations: list[QueryEvaluation] = []
        latency_samples: list[float] = []
        rankings_payload: list[dict[str, JSONValue]] = []
        for query_index, query in enumerate(dataset.queries):
            first_items, _ = completed[(name, query_index, 0)]
            for repetition in range(1, normalized_repetitions):
                repeated_items, _ = completed[(name, query_index, repetition)]
                if _ranking_signature(repeated_items) != _ranking_signature(
                    first_items
                ):
                    raise RetrieverContractError(
                        f"retriever {name!r} returned ranking drift for query "
                        f"{query.id!r} across repetitions"
                    )

            samples = [
                completed[(name, query_index, repetition)][1]
                for repetition in range(normalized_repetitions)
            ]
            latency_samples.extend(samples)
            retrieved_ids = _evaluation_ids(
                first_items,
                relevance_level=dataset.relevance_level,
            )
            search_latency_ms = float(sum(samples) / len(samples))
            if dataset.relevance_level == "document":
                evaluations.append(
                    evaluate_cutoff_rankings(
                        query_id=query.id,
                        retrieved_ids=retrieved_ids,
                        retrieved_ids_by_cutoff={
                            cutoff: _evaluation_ids(
                                first_items[:cutoff], relevance_level="document"
                            )
                            for cutoff in normalized_top_k
                        },
                        relevance_grades=dataset.relevance_grades_by_query[query.id],
                        top_k=normalized_top_k,
                        search_latency_ms=search_latency_ms,
                    )
                )
            else:
                evaluations.append(
                    evaluate_ranking(
                        query_id=query.id,
                        retrieved_ids=retrieved_ids,
                        relevance_grades=dataset.relevance_grades_by_query[query.id],
                        top_k=normalized_top_k,
                        search_latency_ms=search_latency_ms,
                    )
                )
            rankings_payload.append(
                {
                    "items": [_item_payload(item) for item in first_items],
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
        chunk_hash=normalized_chunk_hash,
    )
    runtime = cast(
        dict[str, JSONValue], cast(Mapping[str, JSONValue], manifest["runtime"])
    )
    runtime["async"] = {
        "concurrency": normalized_concurrency,
        "repetitions": normalized_repetitions,
        "timeout_s": normalized_timeout,
    }
    manifest["runtime"] = runtime
    return EvaluationResult(
        run_id=run_id,
        metrics=metrics,
        query_results=query_results,
        manifest=manifest,
        latency=latency,
    )


def _validate_async_retrievers(
    retrievers: Mapping[str, AsyncRetriever],
) -> tuple[tuple[str, AsyncRetriever], ...]:
    if not isinstance(retrievers, Mapping) or not retrievers:
        raise ConfigurationError("retrievers must be a non-empty mapping")
    normalized: list[tuple[str, AsyncRetriever]] = []
    for raw_name, retriever in retrievers.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ConfigurationError("retriever mapping keys must be non-empty strings")
        try:
            name = retriever.name
            retrieve = retriever.retrieve
        except Exception as exc:
            raise ConfigurationError(
                f"retriever {raw_name!r} must implement the AsyncRetriever protocol"
            ) from exc
        if not builtins.callable(retrieve):
            raise ConfigurationError(
                f"retriever {raw_name!r} must implement the AsyncRetriever protocol"
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


def _validate_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{field_name} must be a positive integer")
    return value


def _validate_timeout(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError("timeout_s must be None or a finite positive number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ConfigurationError(
            "timeout_s must be None or a finite positive number"
        ) from exc
    if normalized <= 0.0 or not math.isfinite(normalized):
        raise ConfigurationError("timeout_s must be None or a finite positive number")
    return normalized


def _first_task_error(error: BaseException) -> BaseException:
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            candidate = _first_task_error(child)
            if isinstance(candidate, (RetrieverContractError, ConfigurationError)):
                return candidate
        return _first_task_error(error.exceptions[0]) if error.exceptions else error
    return error


__all__ = ["AsyncCallableRetriever", "AsyncRetriever", "evaluate_async_retrievers"]
