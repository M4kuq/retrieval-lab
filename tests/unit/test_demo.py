import math
from collections.abc import Sequence

import pytest

from retrieval_lab import (
    BaseRetriever,
    Chunk,
    DemoComparison,
    SearchResult,
    compare_retrievers_for_query,
    retrieval_metric_explanations,
)
from retrieval_lab.exceptions import EvaluationError, RetrieverContractError


class StubRetriever(BaseRetriever):
    def __init__(
        self,
        name: str,
        results: list[SearchResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._name = name
        self._results = [] if results is None else results
        self._error = error

    @property
    def name(self) -> str:
        return self._name

    def index(self, chunks: Sequence[Chunk]) -> None:
        return None

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        if self._error is not None:
            raise self._error
        return self._results


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _result(
    chunk_id: str = "chunk-1",
    *,
    rank: int = 1,
    document_id: str = "doc-1",
    score: float = 1.0,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        text=f"text for {chunk_id}",
        score=score,
        rank=rank,
        metadata={"category": "demo"},
    )


def test_compare_retrievers_for_query_preserves_order_and_latency() -> None:
    comparison = compare_retrievers_for_query(
        [StubRetriever("first", [_result()]), StubRetriever("second", [])],
        "query",
        top_k=3,
        clock=SequenceClock(10.0, 10.002, 20.0, 20.005),
    )

    assert isinstance(comparison, DemoComparison)
    assert comparison.query == "query"
    assert comparison.top_k == 3
    assert [view.retriever for view in comparison.views] == ["first", "second"]
    assert comparison.views[0].latency_ms == pytest.approx(2.0)
    assert comparison.views[1].latency_ms == pytest.approx(5.0)
    assert comparison.views[0].results[0].metadata == {"category": "demo"}
    payload = comparison.to_dict()
    assert payload["query"] == "query"
    assert payload["views"][0]["results"][0]["chunk_id"] == "chunk-1"


def test_demo_hit_metadata_is_defensively_copied() -> None:
    metadata = {"category": "before"}
    source = SearchResult("chunk-1", "doc-1", "text", 1.0, 1, metadata)
    comparison = compare_retrievers_for_query(
        [StubRetriever("stub", [source])],
        "query",
        clock=SequenceClock(1.0, 1.0),
    )
    metadata["category"] = "after"

    assert comparison.views[0].results[0].metadata["category"] == "before"
    with pytest.raises(TypeError):
        comparison.views[0].results[0].metadata["category"] = "mutated"  # type: ignore[index]


@pytest.mark.parametrize("query", ["", "   ", None, 1])
def test_compare_rejects_invalid_query(query: object) -> None:
    with pytest.raises(EvaluationError, match="query"):
        compare_retrievers_for_query(
            [StubRetriever("stub")], query  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_compare_rejects_invalid_top_k(top_k: object) -> None:
    with pytest.raises(EvaluationError, match="top_k"):
        compare_retrievers_for_query(
            [StubRetriever("stub")], "query", top_k=top_k  # type: ignore[arg-type]
        )


def test_compare_requires_nonempty_retriever_sequence() -> None:
    with pytest.raises(EvaluationError, match="must not be empty"):
        compare_retrievers_for_query([], "query")


def test_compare_rejects_non_retriever_item() -> None:
    with pytest.raises(EvaluationError, match="BaseRetriever"):
        compare_retrievers_for_query([object()], "query")  # type: ignore[list-item]


def test_compare_rejects_duplicate_retriever_names() -> None:
    with pytest.raises(EvaluationError, match="unique"):
        compare_retrievers_for_query(
            [StubRetriever("same"), StubRetriever("same")], "query"
        )


def test_compare_rejects_empty_retriever_name() -> None:
    with pytest.raises(RetrieverContractError, match="name"):
        compare_retrievers_for_query([StubRetriever("")], "query")


def test_compare_rejects_non_callable_clock() -> None:
    with pytest.raises(EvaluationError, match="clock"):
        compare_retrievers_for_query(
            [StubRetriever("stub")], "query", clock=1  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [math.inf, math.nan, "bad"])
def test_compare_rejects_invalid_clock_value(value: object) -> None:
    with pytest.raises(EvaluationError, match="finite"):
        compare_retrievers_for_query(
            [StubRetriever("stub")],
            "query",
            clock=lambda: value,  # type: ignore[return-value]
        )


def test_compare_rejects_clock_moving_backwards() -> None:
    with pytest.raises(EvaluationError, match="backwards"):
        compare_retrievers_for_query(
            [StubRetriever("stub")],
            "query",
            clock=SequenceClock(2.0, 1.0),
        )


def test_compare_wraps_unexpected_retriever_error() -> None:
    with pytest.raises(RetrieverContractError, match="boom") as captured:
        compare_retrievers_for_query(
            [StubRetriever("stub", error=RuntimeError("boom"))],
            "query",
            clock=SequenceClock(1.0),
        )

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_compare_preserves_retrieval_lab_contract_error() -> None:
    error = RetrieverContractError("not indexed")
    with pytest.raises(RetrieverContractError, match="not indexed") as captured:
        compare_retrievers_for_query(
            [StubRetriever("stub", error=error)],
            "query",
            clock=SequenceClock(1.0),
        )

    assert captured.value is error


def test_compare_requires_list_results() -> None:
    retriever = StubRetriever("stub")
    retriever._results = ()  # type: ignore[assignment]

    with pytest.raises(RetrieverContractError, match="list"):
        compare_retrievers_for_query(
            [retriever], "query", clock=SequenceClock(1.0, 1.0)
        )


def test_compare_rejects_more_than_top_k_results() -> None:
    with pytest.raises(RetrieverContractError, match="more than top_k"):
        compare_retrievers_for_query(
            [StubRetriever("stub", [_result("a", rank=1), _result("b", rank=2)])],
            "query",
            top_k=1,
            clock=SequenceClock(1.0, 1.0),
        )


def test_compare_rejects_non_search_result() -> None:
    retriever = StubRetriever("stub")
    retriever._results = [object()]  # type: ignore[list-item]

    with pytest.raises(RetrieverContractError, match="SearchResult"):
        compare_retrievers_for_query(
            [retriever], "query", clock=SequenceClock(1.0, 1.0)
        )


def test_compare_rejects_noncontiguous_rank() -> None:
    with pytest.raises(RetrieverContractError, match="contiguous"):
        compare_retrievers_for_query(
            [StubRetriever("stub", [_result(rank=2)])],
            "query",
            clock=SequenceClock(1.0, 1.0),
        )


def test_compare_rejects_duplicate_chunk_ids() -> None:
    with pytest.raises(RetrieverContractError, match="duplicate chunk ID"):
        compare_retrievers_for_query(
            [
                StubRetriever(
                    "stub",
                    [_result("same", rank=1), _result("same", rank=2)],
                )
            ],
            "query",
            top_k=2,
            clock=SequenceClock(1.0, 1.0),
        )


def test_metric_explanations_cover_public_retrieval_metrics() -> None:
    explanations = retrieval_metric_explanations()

    assert set(explanations) == {
        "hit_rate",
        "map",
        "mrr",
        "ndcg",
        "precision",
        "recall",
    }
    assert all(explanations.values())
    with pytest.raises(TypeError):
        explanations["new"] = "value"  # type: ignore[index]
