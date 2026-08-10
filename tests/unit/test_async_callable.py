"""Tests for provider-independent asynchronous callable retrieval."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from contextlib import suppress
from typing import cast

import pytest

import retrieval_lab.retrievers.async_callable as async_callable_module
import retrieval_lab.retrievers.callable as callable_module
from retrieval_lab import (
    AsyncCallableRetriever,
    AsyncRetriever,
    CallableRetriever,
    EvaluationDataset,
    EvaluationQuery,
    EvaluationResult,
    RetrievedItem,
    evaluate_async_retrievers,
    evaluate_retrievers,
)
from retrieval_lab.exceptions import ConfigurationError, RetrieverContractError


def _run(coroutine: object) -> object:
    return asyncio.run(cast("asyncio.Future[object]", coroutine))


def _document_dataset() -> EvaluationDataset:
    return EvaluationDataset(
        queries=[
            EvaluationQuery(
                id="q-1", query="first", relevant_document_ids={"doc-a", "doc-b"}
            ),
            EvaluationQuery(id="q-2", query="second", relevant_document_ids={"doc-c"}),
        ],
        relevance_grades_by_query={
            "q-1": {"doc-a": 2, "doc-b": 1},
            "q-2": {"doc-c": 1},
        },
    )


def _chunk_dataset() -> EvaluationDataset:
    return EvaluationDataset(
        [
            EvaluationQuery(
                id="q-1", query="first", relevant_chunk_ids={"chunk-a", "chunk-b"}
            )
        ],
        relevance_level="chunk",
    )


def test_async_callable_invokes_keyword_top_k_and_uses_structural_typing() -> None:
    calls: list[tuple[str, int]] = []

    async def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        calls.append((query, top_k))
        return [RetrievedItem("second"), RetrievedItem("first")]

    retriever = AsyncCallableRetriever("production", search)
    assert retriever.name == "production"
    assert callable(retriever.retrieve)
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(retriever, AsyncRetriever)
    result = _run(retriever.retrieve("query", top_k=2))
    assert [item.id for item in cast(tuple[RetrievedItem, ...], result)] == [
        "second",
        "first",
    ]
    assert calls == [("query", 2)]


def test_async_evaluation_normalizes_scored_ties_by_item_id() -> None:
    async def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        return [
            RetrievedItem("z", score=1.0, rank=1),
            RetrievedItem("a", score=1.0, rank=2),
        ]

    dataset = EvaluationDataset(
        [EvaluationQuery(id="q", query="query", relevant_chunk_ids={"a"})],
        relevance_level="chunk",
    )
    result = cast(
        EvaluationResult,
        _run(
            evaluate_async_retrievers(
                dataset=dataset,
                retrievers={"scored": AsyncCallableRetriever("scored", search)},
                top_k=[1, 2],
            )
        ),
    )

    assert result.query_results["scored"][0].retrieved_ids == ("a", "z")
    assert result.metrics["scored"].recall_at(1) == 1.0


def test_async_chunk_hash_and_document_cutoffs_are_preserved() -> None:
    async def chunk_search(query: str, *, top_k: int) -> list[RetrievedItem]:
        return [RetrievedItem("chunk-a")]

    chunk_result = cast(
        EvaluationResult,
        _run(
            evaluate_async_retrievers(
                dataset=_chunk_dataset(),
                retrievers={"chunks": AsyncCallableRetriever("chunks", chunk_search)},
                top_k=[1],
                chunk_hash="chunk-definition-v1",
            )
        ),
    )
    assert chunk_result.manifest["chunk_hash"] == "chunk-definition-v1"
    with pytest.raises(ConfigurationError, match="chunk_hash"):
        _run(
            evaluate_async_retrievers(
                dataset=_chunk_dataset(),
                retrievers={"chunks": AsyncCallableRetriever("chunks", chunk_search)},
                top_k=[1],
                chunk_hash="",
            )
        )

    async def document_search(query: str, *, top_k: int) -> list[RetrievedItem]:
        return [
            RetrievedItem("a-1", parent_document_id="doc-a"),
            RetrievedItem("a-2", parent_document_id="doc-a"),
            RetrievedItem("b-1", parent_document_id="doc-b"),
        ][:top_k]

    dataset = EvaluationDataset(
        [EvaluationQuery(id="q", query="query", relevant_document_ids={"doc-b"})]
    )
    retriever = AsyncCallableRetriever("documents", document_search)
    narrow = cast(
        EvaluationResult,
        _run(
            evaluate_async_retrievers(
                dataset=dataset,
                retrievers={"documents": retriever},
                top_k=[2],
            )
        ),
    )
    wide = cast(
        EvaluationResult,
        _run(
            evaluate_async_retrievers(
                dataset=dataset,
                retrievers={"documents": retriever},
                top_k=[2, 3],
            )
        ),
    )
    assert narrow.metrics["documents"].recall_at(2) == 0.0
    assert wide.metrics["documents"].recall_at(2) == 0.0
    assert wide.metrics["documents"].recall_at(3) == 1.0
    evidence = wide.query_results["documents"][0]
    assert evidence.retrieved_ids_by_cutoff == {
        2: ("doc-a",),
        3: ("doc-a", "doc-b"),
    }


@pytest.mark.parametrize(
    "raw_result",
    [
        "not a sequence",
        object(),
        [object()],
        [RetrievedItem("same"), RetrievedItem("same")],
    ],
)
def test_async_callable_rejects_invalid_result_shapes(raw_result: object) -> None:
    async def search(query: str, *, top_k: int) -> object:
        return raw_result

    with pytest.raises(RetrieverContractError):
        _run(AsyncCallableRetriever("bad", search).retrieve("query", top_k=2))


def test_async_callable_rejects_overreturn_and_rank_contract() -> None:
    async def too_many(query: str, *, top_k: int) -> list[RetrievedItem]:
        return [RetrievedItem("a"), RetrievedItem("b")]

    async def bad_rank(query: str, *, top_k: int) -> list[RetrievedItem]:
        return [RetrievedItem("a", rank=2), RetrievedItem("b", rank=1)]

    with pytest.raises(RetrieverContractError, match="top_k=1"):
        _run(AsyncCallableRetriever("too-many", too_many).retrieve("q", top_k=1))
    with pytest.raises(RetrieverContractError, match="sequence order"):
        _run(AsyncCallableRetriever("rank", bad_rank).retrieve("q", top_k=2))


def test_async_callable_empty_results_and_constructor_arguments() -> None:
    async def empty(query: str, *, top_k: int) -> list[RetrievedItem]:
        return []

    assert _run(AsyncCallableRetriever("empty", empty).retrieve("q", top_k=1)) == ()
    with pytest.raises(ConfigurationError, match="name"):
        AsyncCallableRetriever("", empty)
    with pytest.raises(ConfigurationError, match="callable"):
        AsyncCallableRetriever("bad", cast(object, object()))
    with pytest.raises(RetrieverContractError, match="query"):
        _run(
            AsyncCallableRetriever("valid", empty).retrieve(
                cast(str, object()), top_k=1
            )
        )
    with pytest.raises(RetrieverContractError, match="top_k"):
        _run(AsyncCallableRetriever("valid", empty).retrieve("q", top_k=0))


def test_async_callable_chains_low_level_exception() -> None:
    cause = RuntimeError("provider outage")

    async def failing(query: str, *, top_k: int) -> Sequence[RetrievedItem]:
        raise cause

    with pytest.raises(RetrieverContractError, match="callable failed") as raised:
        _run(AsyncCallableRetriever("production", failing).retrieve("q", top_k=1))
    assert raised.value.__cause__ is cause


def test_async_evaluation_runs_inside_existing_loop_and_orders_results() -> None:
    calls: list[tuple[str, str, int]] = []

    async def search_a(query: str, *, top_k: int) -> list[RetrievedItem]:
        await asyncio.sleep(0.01 if query == "first" else 0.0)
        calls.append(("a", query, top_k))
        return [RetrievedItem("doc-a" if query == "first" else "doc-c")]

    async def search_b(query: str, *, top_k: int) -> list[RetrievedItem]:
        await asyncio.sleep(0.0 if query == "first" else 0.01)
        calls.append(("b", query, top_k))
        return [RetrievedItem("doc-a" if query == "first" else "doc-c")]

    async def scenario() -> object:
        return await evaluate_async_retrievers(
            dataset=_document_dataset(),
            retrievers={
                "b": AsyncCallableRetriever("b", search_b),
                "a": AsyncCallableRetriever("a", search_a),
            },
            top_k=[1, 3],
            concurrency=2,
        )

    result = cast("object", _run(scenario()))
    assert list(cast(object, result).query_results) == ["a", "b"]  # type: ignore[union-attr]
    assert [q.query_id for q in cast(object, result).query_results["a"]] == [  # type: ignore[union-attr]
        "q-1",
        "q-2",
    ]
    assert sorted(calls) == sorted(
        [(name, query, 3) for name in ("a", "b") for query in ("first", "second")]
    )


def test_async_evaluation_bounds_concurrency_and_repeats_once_per_setting() -> None:
    active = 0
    maximum = 0
    calls: list[tuple[str, int]] = []

    async def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        calls.append((query, top_k))
        await asyncio.sleep(0.005)
        active -= 1
        return [RetrievedItem("doc-a")]

    result = cast(
        object,
        _run(
            evaluate_async_retrievers(
                dataset=_document_dataset(),
                retrievers={"a": AsyncCallableRetriever("a", search)},
                top_k=[1],
                concurrency=2,
                repetitions=3,
            )
        ),
    )
    assert maximum == 2
    assert len(calls) == 6
    assert cast(object, result).latency["a"].sample_count == 6  # type: ignore[union-attr]
    assert cast(object, result).query_results["a"][0].search_latency_ms is not None  # type: ignore[union-attr]


def test_async_evaluation_rejects_ranking_drift() -> None:
    count = 0

    async def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        nonlocal count
        count += 1
        return [RetrievedItem("doc-a" if count == 1 else "doc-b")]

    with pytest.raises(RetrieverContractError, match="drift"):
        _run(
            evaluate_async_retrievers(
                dataset=_document_dataset(),
                retrievers={"a": AsyncCallableRetriever("a", search)},
                top_k=[1],
                repetitions=2,
            )
        )


def test_async_evaluation_allows_score_drift_when_ranking_identity_is_stable() -> None:
    score = 0.0

    async def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        nonlocal score
        score += 1.0
        return [
            RetrievedItem("chunk-a", parent_document_id="doc-a", score=score, rank=1)
        ]

    result = _run(
        evaluate_async_retrievers(
            dataset=_chunk_dataset(),
            retrievers={"a": AsyncCallableRetriever("a", search)},
            top_k=[1],
            repetitions=3,
        )
    )
    assert result.query_results["a"][0].retrieved_ids == ("chunk-a",)  # type: ignore[union-attr]


def test_async_score_magnitude_does_not_change_ranking_run_identity() -> None:
    def evaluate(start: float):
        score = start

        async def search(query: str, *, top_k: int) -> list[RetrievedItem]:
            nonlocal score
            score += 1.0
            return [RetrievedItem("chunk-a", score=score, rank=1)]

        return _run(
            evaluate_async_retrievers(
                dataset=_chunk_dataset(),
                retrievers={"a": AsyncCallableRetriever("a", search)},
                top_k=[1],
                repetitions=2,
            )
        )

    first = evaluate(0.0)
    second = evaluate(100.0)

    assert first.run_id == second.run_id  # type: ignore[union-attr]
    assert (  # type: ignore[union-attr]
        first.manifest["retrieved_rankings_hash"]
        == second.manifest["retrieved_rankings_hash"]
    )


def test_async_external_cancellation_wins_over_provider_cleanup_failure() -> None:
    started = asyncio.Event()

    async def failing_cleanup(query: str, *, top_k: int) -> list[RetrievedItem]:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            raise RuntimeError("cleanup failed")

    async def scenario() -> None:
        task = asyncio.create_task(
            evaluate_async_retrievers(
                dataset=_chunk_dataset(),
                retrievers={"a": AsyncCallableRetriever("a", failing_cleanup)},
                top_k=[1],
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _run(scenario())


def test_async_provider_cancellation_after_suppressed_cancel_is_wrapped() -> None:
    async def provider_cancelled(query: str, *, top_k: int) -> list[RetrievedItem]:
        raise asyncio.CancelledError

    async def scenario() -> None:
        parent = asyncio.current_task()
        assert parent is not None
        parent.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.sleep(0)
        try:
            with pytest.raises(RetrieverContractError, match="was cancelled"):
                await evaluate_async_retrievers(
                    dataset=_chunk_dataset(),
                    retrievers={"a": AsyncCallableRetriever("a", provider_cancelled)},
                    top_k=[1],
                )
        finally:
            parent.uncancel()

    _run(scenario())


def test_async_evaluation_schedules_only_a_fixed_worker_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = 0
    original_create_task = asyncio.create_task

    def create_task(coroutine: object) -> asyncio.Task[object]:
        nonlocal created
        created += 1
        return original_create_task(coroutine)  # type: ignore[arg-type]

    monkeypatch.setattr(async_callable_module.asyncio, "create_task", create_task)

    async def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        await asyncio.sleep(0)
        return [RetrievedItem("chunk-a")]

    _run(
        evaluate_async_retrievers(
            dataset=_chunk_dataset(),
            retrievers={"a": AsyncCallableRetriever("a", search)},
            top_k=[1],
            concurrency=3,
            repetitions=20,
        )
    )
    assert created == 3


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"concurrency": 0}, "concurrency"),
        ({"concurrency": True}, "concurrency"),
        ({"repetitions": 0}, "repetitions"),
        ({"repetitions": False}, "repetitions"),
        ({"timeout_s": 0}, "timeout_s"),
        ({"timeout_s": math.inf}, "timeout_s"),
        ({"timeout_s": 10**10000}, "timeout_s"),
        ({"timeout_s": True}, "timeout_s"),
    ],
)
def test_async_evaluation_rejects_invalid_execution_settings(
    kwargs: dict[str, object], field: str
) -> None:
    async def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        return []

    with pytest.raises(ConfigurationError, match=field):
        _run(
            evaluate_async_retrievers(
                dataset=_chunk_dataset(),
                retrievers={"a": AsyncCallableRetriever("a", search)},
                top_k=[1],
                **kwargs,
            )
        )


def test_async_timeout_chains_timeout_and_cancels_call() -> None:
    cancelled = False

    async def slow(query: str, *, top_k: int) -> list[RetrievedItem]:
        nonlocal cancelled
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            cancelled = True
            raise
        return []

    with pytest.raises(RetrieverContractError, match="timed out") as raised:
        _run(
            evaluate_async_retrievers(
                dataset=_chunk_dataset(),
                retrievers={"a": AsyncCallableRetriever("a", slow)},
                top_k=[1],
                timeout_s=0.001,
            )
        )
    assert isinstance(raised.value.__cause__, TimeoutError)
    assert cancelled


def test_async_external_cancellation_cleans_siblings_and_is_reraised() -> None:
    cleaned = asyncio.Event()

    async def slow(query: str, *, top_k: int) -> list[RetrievedItem]:
        try:
            await asyncio.sleep(10.0)
        finally:
            cleaned.set()
        return []

    async def scenario() -> None:
        task = asyncio.create_task(
            evaluate_async_retrievers(
                dataset=_document_dataset(),
                retrievers={"a": AsyncCallableRetriever("a", slow)},
                top_k=[1],
                concurrency=1,
            )
        )
        await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()

    _run(scenario())


def test_async_provider_cancellation_is_a_contract_error_and_cleans_siblings() -> None:
    sibling_started = asyncio.Event()
    sibling_cleaned = asyncio.Event()

    async def cancelled(query: str, *, top_k: int) -> list[RetrievedItem]:
        if query == "first":
            await sibling_started.wait()
            raise asyncio.CancelledError
        sibling_started.set()
        try:
            await asyncio.sleep(10.0)
        finally:
            sibling_cleaned.set()
        return []

    with pytest.raises(RetrieverContractError, match="cancelled") as raised:
        _run(
            evaluate_async_retrievers(
                dataset=_document_dataset(),
                retrievers={"a": AsyncCallableRetriever("a", cancelled)},
                top_k=[1],
                concurrency=2,
            )
        )
    assert isinstance(raised.value.__cause__, asyncio.CancelledError)
    assert sibling_cleaned.is_set()


def test_async_partial_failure_cleans_sibling_tasks() -> None:
    cleaned = asyncio.Event()

    async def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        if query == "first":
            raise RuntimeError("failure")
        try:
            await asyncio.sleep(10.0)
        finally:
            cleaned.set()
        return []

    with pytest.raises(RetrieverContractError):
        _run(
            evaluate_async_retrievers(
                dataset=_document_dataset(),
                retrievers={"a": AsyncCallableRetriever("a", search)},
                top_k=[1],
                concurrency=2,
            )
        )
    assert cleaned.is_set()


def test_async_evaluation_adds_query_context_to_adapter_failures() -> None:
    cause = RuntimeError("provider failed")

    async def failing(query: str, *, top_k: int) -> list[RetrievedItem]:
        raise cause

    with pytest.raises(RetrieverContractError, match="query 'q-1'") as raised:
        _run(
            evaluate_async_retrievers(
                dataset=_chunk_dataset(),
                retrievers={"a": AsyncCallableRetriever("a", failing)},
                top_k=[1],
            )
        )
    adapter_error = raised.value.__cause__
    assert isinstance(adapter_error, RetrieverContractError)
    assert adapter_error.__cause__ is cause


def test_async_evaluation_averages_repetition_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = iter([0, 1_000_000, 2_000_000, 5_000_000])
    monkeypatch.setattr(async_callable_module, "perf_counter_ns", readings.__next__)

    async def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        return [RetrievedItem("chunk-a")]

    result = _run(
        evaluate_async_retrievers(
            dataset=_chunk_dataset(),
            retrievers={"a": AsyncCallableRetriever("a", search)},
            top_k=[1],
            repetitions=2,
        )
    )
    assert result.query_results["a"][0].search_latency_ms == 2.0  # type: ignore[union-attr]
    assert result.latency["a"].mean_ms == 2.0  # type: ignore[union-attr]


def test_async_runtime_start_is_captured_before_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(async_callable_module, "_utc_timestamp", lambda: "started")
    monkeypatch.setattr(callable_module, "_utc_timestamp", lambda: "finished")

    async def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        return [RetrievedItem("chunk-a")]

    result = _run(
        evaluate_async_retrievers(
            dataset=_chunk_dataset(),
            retrievers={"a": AsyncCallableRetriever("a", search)},
            top_k=[1],
        )
    )
    runtime = result.manifest["runtime"]  # type: ignore[union-attr]
    assert runtime["started_at_utc"] == "started"  # type: ignore[index]
    assert runtime["finished_at_utc"] == "finished"  # type: ignore[index]


def test_async_document_collapse_and_chunk_relevance() -> None:
    async def documents(query: str, *, top_k: int) -> list[RetrievedItem]:
        return [
            RetrievedItem("chunk-a-1", parent_document_id="doc-a"),
            RetrievedItem("chunk-a-2", parent_document_id="doc-a"),
            RetrievedItem("chunk-b", parent_document_id="doc-b"),
        ]

    document_result = cast(
        object,
        _run(
            evaluate_async_retrievers(
                dataset=_document_dataset(),
                retrievers={"a": AsyncCallableRetriever("a", documents)},
                top_k=[1, 3],
            )
        ),
    )
    assert cast(object, document_result).query_results["a"][0].retrieved_ids == (  # type: ignore[union-attr]
        "doc-a",
        "doc-b",
    )

    async def chunks(query: str, *, top_k: int) -> list[RetrievedItem]:
        return [RetrievedItem("chunk-a"), RetrievedItem("chunk-b")]

    chunk_result = cast(
        object,
        _run(
            evaluate_async_retrievers(
                dataset=_chunk_dataset(),
                retrievers={"a": AsyncCallableRetriever("a", chunks)},
                top_k=[1, 2],
            )
        ),
    )
    assert cast(object, chunk_result).query_results["a"][0].retrieved_ids == (  # type: ignore[union-attr]
        "chunk-a",
        "chunk-b",
    )


def test_async_chunk_duplicate_evaluation_ids_are_rejected() -> None:
    async def duplicate(query: str, *, top_k: int) -> list[RetrievedItem]:
        return [RetrievedItem("chunk-a"), RetrievedItem("chunk-a")]

    with pytest.raises(RetrieverContractError, match="duplicate"):
        _run(
            evaluate_async_retrievers(
                dataset=_chunk_dataset(),
                retrievers={"a": AsyncCallableRetriever("a", duplicate)},
                top_k=[2],
            )
        )


def test_async_run_id_and_metrics_match_sync_for_same_ranking() -> None:
    dataset = _document_dataset()

    def sync_search(query: str, *, top_k: int) -> list[RetrievedItem]:
        return [RetrievedItem("doc-a")] if query == "first" else []

    async def async_search(query: str, *, top_k: int) -> list[RetrievedItem]:
        return [RetrievedItem("doc-a")] if query == "first" else []

    sync_result = evaluate_retrievers(
        dataset=dataset,
        retrievers={"same": CallableRetriever("same", sync_search)},
        top_k=[1, 2],
    )
    async_result = cast(
        object,
        _run(
            evaluate_async_retrievers(
                dataset=dataset,
                retrievers={"same": AsyncCallableRetriever("same", async_search)},
                top_k=[1, 2],
                concurrency=3,
                repetitions=2,
                timeout_s=2.0,
            )
        ),
    )
    assert sync_result.run_id == cast(object, async_result).run_id  # type: ignore[union-attr]
    assert sync_result.metrics == cast(object, async_result).metrics  # type: ignore[union-attr]
    assert cast(object, async_result).manifest["runtime"]["async"] == {  # type: ignore[union-attr]
        "concurrency": 3,
        "repetitions": 2,
        "timeout_s": 2.0,
    }


def test_async_retriever_mapping_contract_and_custom_protocol() -> None:
    class Custom:
        name = "custom"

        async def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedItem]:
            return [RetrievedItem("chunk-a")]

    assert _run(
        evaluate_async_retrievers(
            dataset=_chunk_dataset(), retrievers={"custom": Custom()}, top_k=[1]
        )
    )

    with pytest.raises(ConfigurationError, match="match"):
        _run(
            evaluate_async_retrievers(
                dataset=_chunk_dataset(),
                retrievers={"wrong": Custom()},
                top_k=[1],
            )
        )
