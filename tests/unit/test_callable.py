"""Tests for the provider-independent synchronous callable adapter."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from typing import cast

import pytest

from retrieval_lab import (
    CallableRetriever,
    Document,
    EvaluationDataset,
    EvaluationQuery,
    EvaluationRunner,
    RetrievedItem,
    Retriever,
    check_comparability,
    evaluate_retrievers,
)
from retrieval_lab.evaluation.precomputed import RetrievedQueryResult, evaluate_results
from retrieval_lab.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    RetrieverContractError,
)


def _document_dataset() -> EvaluationDataset:
    return EvaluationDataset(
        queries=[
            EvaluationQuery(
                id="q-1",
                query="first",
                relevant_document_ids={"doc-a", "doc-b"},
            ),
            EvaluationQuery(
                id="q-2",
                query="second",
                relevant_document_ids={"doc-c"},
            ),
        ],
        relevance_level="document",
        relevance_grades_by_query={
            "q-1": {"doc-a": 2, "doc-b": 1},
            "q-2": {"doc-c": 1},
        },
    )


def _chunk_dataset() -> EvaluationDataset:
    return EvaluationDataset(
        queries=[
            EvaluationQuery(
                id="q-1",
                query="first",
                relevant_chunk_ids={"chunk-a", "chunk-b"},
            )
        ],
        relevance_level="chunk",
    )


def test_retrieved_item_is_immutable_and_fully_validated() -> None:
    item = RetrievedItem(
        id="chunk-a",
        parent_document_id="doc-a",
        score=2,
        rank=1,
    )
    assert item.score == 2.0
    with pytest.raises(AttributeError):
        item.id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"id": ""}, "id"),
        ({"id": "a", "parent_document_id": ""}, "parent_document_id"),
        ({"id": "a", "score": math.nan}, "score"),
        ({"id": "a", "score": math.inf}, "score"),
        ({"id": "a", "score": 10**10000}, "score"),
        ({"id": "a", "rank": 0}, "rank"),
        ({"id": "a", "rank": True}, "rank"),
    ],
)
def test_retrieved_item_rejects_invalid_fields(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(RetrieverContractError, match=message):
        RetrievedItem(**kwargs)  # type: ignore[arg-type]


def test_callable_retriever_preserves_sequence_order_and_keyword_top_k() -> None:
    calls: list[tuple[str, int]] = []

    def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        calls.append((query, top_k))
        return [RetrievedItem("second"), RetrievedItem("first")]

    retriever = CallableRetriever("production", search)
    assert retriever.name == "production"
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(retriever, Retriever)
    assert [item.id for item in retriever.retrieve("query", top_k=2)] == [
        "second",
        "first",
    ]
    assert calls == [("query", 2)]


@pytest.mark.parametrize(
    "raw_result",
    [
        "not a sequence",
        object(),
        [object()],
        [RetrievedItem("same"), RetrievedItem("same")],
    ],
)
def test_callable_retriever_rejects_invalid_result_shapes(raw_result: object) -> None:
    retriever = CallableRetriever("production", lambda query, top_k: raw_result)
    with pytest.raises(RetrieverContractError):
        retriever.retrieve("query", top_k=2)


def test_callable_retriever_rejects_too_many_and_inconsistent_ranks() -> None:
    too_many = CallableRetriever(
        "too-many",
        lambda query, top_k: [RetrievedItem("a"), RetrievedItem("b")],
    )
    with pytest.raises(RetrieverContractError, match="top_k=1"):
        too_many.retrieve("query", top_k=1)

    mixed = CallableRetriever(
        "mixed",
        lambda query, top_k: [RetrievedItem("a", rank=1), RetrievedItem("b")],
    )
    with pytest.raises(RetrieverContractError, match="every"):
        mixed.retrieve("query", top_k=2)

    out_of_order = CallableRetriever(
        "out-of-order",
        lambda query, top_k: [RetrievedItem("a", rank=2), RetrievedItem("b", rank=1)],
    )
    with pytest.raises(RetrieverContractError, match="sequence order"):
        out_of_order.retrieve("query", top_k=2)


def test_callable_retriever_allows_empty_results() -> None:
    retriever = CallableRetriever("empty", lambda query, top_k: [])
    assert retriever.retrieve("query", top_k=1) == ()


def test_callable_retriever_chains_low_level_exception() -> None:
    cause = RuntimeError("provider outage")

    def failing(query: str, *, top_k: int) -> Sequence[RetrievedItem]:
        raise cause

    with pytest.raises(RetrieverContractError, match="callable failed") as raised:
        CallableRetriever("production", failing).retrieve("query", top_k=1)
    assert raised.value.__cause__ is cause


def test_callable_retriever_rejects_constructor_and_call_arguments() -> None:
    with pytest.raises(ConfigurationError, match="name"):
        CallableRetriever("", lambda query, top_k: [])
    with pytest.raises(ConfigurationError, match="callable"):
        CallableRetriever("bad", cast(object, object()))

    retriever = CallableRetriever("valid", lambda query, top_k: [])
    with pytest.raises(RetrieverContractError, match="query"):
        retriever.retrieve(cast(str, object()), top_k=1)
    with pytest.raises(RetrieverContractError, match="top_k"):
        retriever.retrieve("query", top_k=0)


def test_evaluate_retrievers_wraps_custom_failures_and_validates_clock() -> None:
    class FailingRetriever:
        name = "failing"

        def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedItem]:
            raise RuntimeError("backend failure")

    with pytest.raises(RetrieverContractError, match="failed") as raised:
        evaluate_retrievers(
            dataset=_chunk_dataset(),
            retrievers={"failing": FailingRetriever()},  # type: ignore[dict-item]
            top_k=[1],
        )
    assert isinstance(raised.value.__cause__, RuntimeError)
    with pytest.raises(ConfigurationError, match="clock"):
        evaluate_retrievers(
            dataset=_chunk_dataset(),
            retrievers={"x": CallableRetriever("x", lambda query, top_k: [])},
            top_k=[1],
            clock=cast(object, object()),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "clock",
    [
        lambda: cast(int, "not an integer"),
        lambda: cast(int, True),
        iter([2, 1]).__next__,
    ],
)
def test_evaluate_retrievers_rejects_invalid_clock_readings(
    clock: object,
) -> None:
    with pytest.raises(ConfigurationError, match="clock"):
        evaluate_retrievers(
            dataset=_chunk_dataset(),
            retrievers={"x": CallableRetriever("x", lambda query, top_k: [])},
            top_k=[1],
            clock=cast(object, clock),  # type: ignore[arg-type]
        )


def test_evaluate_retrievers_chains_clock_failure() -> None:
    cause = RuntimeError("clock unavailable")

    def failing_clock() -> int:
        raise cause

    with pytest.raises(ConfigurationError, match="clock callable failed") as raised:
        evaluate_retrievers(
            dataset=_chunk_dataset(),
            retrievers={"x": CallableRetriever("x", lambda query, top_k: [])},
            top_k=[1],
            clock=failing_clock,
        )
    assert raised.value.__cause__ is cause


def test_unreadable_sequence_is_rejected_at_the_boundary() -> None:
    class Unreadable(Sequence[RetrievedItem]):
        def __getitem__(self, index: int) -> RetrievedItem:
            raise RuntimeError("cannot iterate")

        def __len__(self) -> int:
            return 1

    retriever = CallableRetriever("bad", lambda query, top_k: Unreadable())
    with pytest.raises(RetrieverContractError, match="unreadable"):
        retriever.retrieve("query", top_k=1)


def test_evaluate_retrievers_collapses_parent_documents_and_calls_max_k_once() -> None:
    calls: list[tuple[str, int]] = []

    def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        calls.append((query, top_k))
        if query == "first":
            return [
                RetrievedItem("chunk-a-1", parent_document_id="doc-a"),
                RetrievedItem("chunk-a-2", parent_document_id="doc-a"),
                RetrievedItem("chunk-b", parent_document_id="doc-b"),
            ]
        return [RetrievedItem("chunk-c", parent_document_id="doc-c")]

    result = evaluate_retrievers(
        dataset=_document_dataset(),
        retrievers={"production": CallableRetriever("production", search)},
        top_k=[1, 3],
        clock=iter([0, 1_000_000, 2_000_000, 3_000_000]).__next__,
    )
    assert calls == [("first", 3), ("second", 3)]
    first = result.query_results["production"][0]
    assert first.retrieved_ids == ("doc-a", "doc-b")
    assert result.metrics["production"].recall_at(3) == 1.0
    assert result.latency["production"].sample_count == 2


def test_evaluate_retrievers_uses_item_ids_for_chunk_relevance() -> None:
    retriever = CallableRetriever(
        "chunks",
        lambda query, top_k: [
            RetrievedItem("chunk-a", parent_document_id="doc-a"),
            RetrievedItem("chunk-b", parent_document_id="doc-a"),
        ],
    )
    result = evaluate_retrievers(
        dataset=_chunk_dataset(),
        retrievers={"chunks": retriever},
        top_k=[1, 2],
        chunk_hash="chunk-definition-v1",
    )
    assert result.query_results["chunks"][0].retrieved_ids == (
        "chunk-a",
        "chunk-b",
    )
    assert result.metrics["chunks"].recall_at(2) == 1.0
    assert result.manifest["chunk_hash"] == "chunk-definition-v1"


def test_callable_chunk_hash_controls_strict_comparability_and_run_id() -> None:
    retriever = CallableRetriever(
        "chunks",
        lambda query, top_k: [RetrievedItem("chunk-a")],
    )
    first = evaluate_retrievers(
        dataset=_chunk_dataset(),
        retrievers={"chunks": retriever},
        top_k=[1],
        chunk_hash="chunk-definition-v1",
    )
    same = evaluate_retrievers(
        dataset=_chunk_dataset(),
        retrievers={"chunks": retriever},
        top_k=[1],
        chunk_hash="chunk-definition-v1",
    )
    different = evaluate_retrievers(
        dataset=_chunk_dataset(),
        retrievers={"chunks": retriever},
        top_k=[1],
        chunk_hash="chunk-definition-v2",
    )
    missing = evaluate_retrievers(
        dataset=_chunk_dataset(),
        retrievers={"chunks": retriever},
        top_k=[1],
    )

    assert first.run_id == same.run_id
    assert first.run_id != different.run_id
    assert check_comparability(first, same).comparable
    assert not check_comparability(first, different).comparable
    assert not check_comparability(first, missing).comparable
    with pytest.raises(ConfigurationError, match="chunk_hash"):
        evaluate_retrievers(
            dataset=_chunk_dataset(),
            retrievers={"chunks": retriever},
            top_k=[1],
            chunk_hash="",
        )


def test_document_metrics_do_not_change_when_a_larger_cutoff_is_added() -> None:
    dataset = EvaluationDataset(
        [EvaluationQuery(id="q", query="query", relevant_document_ids={"doc-b"})]
    )

    def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        return [
            RetrievedItem("a-1", parent_document_id="doc-a"),
            RetrievedItem("a-2", parent_document_id="doc-a"),
            RetrievedItem("b-1", parent_document_id="doc-b"),
        ][:top_k]

    retriever = CallableRetriever("documents", search)
    narrow = evaluate_retrievers(
        dataset=dataset,
        retrievers={"documents": retriever},
        top_k=[2],
    )
    wide = evaluate_retrievers(
        dataset=dataset,
        retrievers={"documents": retriever},
        top_k=[2, 3],
    )

    assert narrow.metrics["documents"].recall_at(2) == 0.0
    assert wide.metrics["documents"].recall_at(2) == 0.0
    assert wide.metrics["documents"].recall_at(3) == 1.0
    evidence = wide.query_results["documents"][0]
    assert evidence.retrieved_ids_by_cutoff == {
        2: ("doc-a",),
        3: ("doc-a", "doc-b"),
    }


def test_evaluate_retrievers_normalizes_scored_ties_by_item_id() -> None:
    dataset = EvaluationDataset(
        [EvaluationQuery(id="q", query="query", relevant_chunk_ids={"a"})],
        relevance_level="chunk",
    )
    retriever = CallableRetriever(
        "scored",
        lambda query, top_k: [
            RetrievedItem("z", score=1.0, rank=1),
            RetrievedItem("a", score=1.0, rank=2),
        ],
    )

    result = evaluate_retrievers(
        dataset=dataset,
        retrievers={"scored": retriever},
        top_k=[1, 2],
    )

    query = result.query_results["scored"][0]
    assert query.retrieved_ids == ("a", "z")
    assert result.metrics["scored"].recall_at(1) == 1.0


def test_score_magnitude_does_not_change_ranking_run_identity() -> None:
    def evaluate(score: float):
        retriever = CallableRetriever(
            "scored",
            lambda query, top_k: [
                RetrievedItem("a", score=score, rank=1),
                RetrievedItem("b", score=score - 1.0, rank=2),
            ],
        )
        return evaluate_retrievers(
            dataset=_chunk_dataset(),
            retrievers={"scored": retriever},
            top_k=[1, 2],
        )

    first = evaluate(2.0)
    second = evaluate(200.0)

    assert first.run_id == second.run_id
    assert (
        first.manifest["retrieved_rankings_hash"]
        == second.manifest["retrieved_rankings_hash"]
    )


def test_evaluate_retrievers_stops_clock_before_result_validation() -> None:
    now_ns = 0

    class DelayedSequence(Sequence[RetrievedItem]):
        def __iter__(self) -> Iterator[RetrievedItem]:
            nonlocal now_ns
            now_ns = 100_000_000
            yield RetrievedItem("chunk-a")

        def __getitem__(self, index: int) -> RetrievedItem:
            if index != 0:
                raise IndexError(index)
            return RetrievedItem("chunk-a")

        def __len__(self) -> int:
            return 1

    class RetrieverWithDelayedResults:
        name = "delayed"

        def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedItem]:
            return DelayedSequence()

    result = evaluate_retrievers(
        dataset=_chunk_dataset(),
        retrievers={"delayed": RetrieverWithDelayedResults()},  # type: ignore[dict-item]
        top_k=[1],
        clock=lambda: now_ns,
    )

    assert result.query_results["delayed"][0].search_latency_ms == 0.0


def test_evaluate_retrievers_matches_precomputed_metrics() -> None:
    dataset = _document_dataset()

    def search(query: str, *, top_k: int) -> list[RetrievedItem]:
        if query == "first":
            return [RetrievedItem("doc-b"), RetrievedItem("doc-a")]
        return []

    callable_result = evaluate_retrievers(
        dataset=dataset,
        retrievers={"system": CallableRetriever("system", search)},
        top_k=[1, 2],
        clock=iter([0, 1, 2, 3]).__next__,
    )
    precomputed_result = evaluate_results(
        dataset=dataset,
        retrieved_results=[
            RetrievedQueryResult("q-1", ["doc-b", "doc-a"]),
            RetrievedQueryResult("q-2", []),
        ],
        top_k=[1, 2],
        name="system",
    )
    assert callable_result.metrics["system"] == precomputed_result.metrics["system"]


def test_evaluate_retrievers_matches_builtin_metrics_for_the_same_ranking() -> None:
    query = EvaluationQuery(
        id="q",
        query="apple",
        relevant_document_ids={"doc-a"},
    )
    built_in = EvaluationRunner.quick_evaluate(
        documents=[
            Document(id="doc-a", text="apple"),
            Document(id="doc-b", text="banana"),
        ],
        queries=[query],
        strategies=["keyword"],
        top_k=[1, 2],
    )
    ranking = built_in.query_results["keyword"][0].retrieved_ids
    external = evaluate_retrievers(
        dataset=EvaluationDataset(queries=[query], relevance_level="document"),
        retrievers={
            "keyword": CallableRetriever(
                "keyword",
                lambda query, top_k: [
                    RetrievedItem(identifier) for identifier in ranking
                ],
            )
        },
        top_k=[1, 2],
    )
    assert external.metrics["keyword"] == built_in.metrics["keyword"]


def test_run_id_ignores_clock_and_runtime_timestamps() -> None:
    dataset = _chunk_dataset()
    retriever = CallableRetriever(
        "system", lambda query, top_k: [RetrievedItem("chunk-a")]
    )
    first = evaluate_retrievers(
        dataset=dataset,
        retrievers={"system": retriever},
        top_k=[1],
        clock=iter([0, 1_000_000]).__next__,
    )
    second = evaluate_retrievers(
        dataset=dataset,
        retrievers={"system": retriever},
        top_k=[1],
        clock=iter([100_000_000, 200_000_000]).__next__,
    )
    assert first.run_id == second.run_id
    assert (
        first.query_results["system"][0].search_latency_ms
        != second.query_results["system"][0].search_latency_ms
    )


def test_evaluate_retrievers_validates_mapping_identity_and_dataset() -> None:
    with pytest.raises(ConfigurationError, match="mapping"):
        evaluate_retrievers(
            dataset=_chunk_dataset(),
            retrievers=cast(dict[str, Retriever], {}),
            top_k=[1],
        )
    with pytest.raises(ConfigurationError, match="does not match"):
        evaluate_retrievers(
            dataset=_chunk_dataset(),
            retrievers={"wrong": CallableRetriever("right", lambda query, top_k: [])},
            top_k=[1],
        )
    with pytest.raises(DatasetValidationError, match="EvaluationDataset"):
        evaluate_retrievers(
            dataset=cast(EvaluationDataset, object()),
            retrievers={"x": CallableRetriever("x", lambda query, top_k: [])},
            top_k=[1],
        )


def test_evaluate_retrievers_validates_protocol_without_leaking_property_errors() -> (
    None
):
    cause = RuntimeError("dynamic property failed")

    class BrokenNameRetriever:
        @property
        def name(self) -> str:
            raise cause

        def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedItem]:
            return []

    with pytest.raises(ConfigurationError, match="Retriever protocol") as raised:
        evaluate_retrievers(
            dataset=_chunk_dataset(),
            retrievers={"broken": BrokenNameRetriever()},  # type: ignore[dict-item]
            top_k=[1],
        )
    assert raised.value.__cause__ is cause

    class MissingRetrieve:
        name = "missing"

    with pytest.raises(ConfigurationError, match="Retriever protocol"):
        evaluate_retrievers(
            dataset=_chunk_dataset(),
            retrievers={"missing": MissingRetrieve()},  # type: ignore[dict-item]
            top_k=[1],
        )


def test_public_imports_expose_callable_api() -> None:
    assert CallableRetriever.__module__ == "retrieval_lab.retrievers.callable"
    assert RetrievedItem.__module__ == "retrieval_lab.retrievers.callable"
    assert Retriever.__module__ == "retrieval_lab.retrievers.callable"
    assert evaluate_retrievers.__module__ == "retrieval_lab.retrievers.callable"


def test_manifest_does_not_embed_retrieved_paths_or_payload_text() -> None:
    secret_path = "/absolute/private/token.txt"
    dataset = EvaluationDataset(
        queries=[
            EvaluationQuery(
                id="q",
                query="private query",
                relevant_chunk_ids={secret_path},
            )
        ],
        relevance_level="chunk",
    )
    result = evaluate_retrievers(
        dataset=dataset,
        retrievers={
            "system": CallableRetriever(
                "system", lambda query, top_k: [RetrievedItem(secret_path)]
            )
        },
        top_k=[1],
    )
    assert secret_path not in json.dumps(result.to_dict()["run"]["manifest"])
