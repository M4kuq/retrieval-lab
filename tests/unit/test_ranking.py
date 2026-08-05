from __future__ import annotations

from dataclasses import fields

import pytest

from retrieval_lab.domain import SearchResult
from retrieval_lab.evaluation.ranking import (
    collapse_to_documents,
    stable_rank_results,
    validate_search_results,
)
from retrieval_lab.exceptions import RetrieverContractError


def _result(
    chunk_id: str,
    document_id: str,
    score: float,
    rank: int,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        text=f"text for {chunk_id}",
        score=score,
        rank=rank,
    )


def _unsafe_replace(result: SearchResult, **changes: object) -> SearchResult:
    """Create an invalid frozen record to exercise the defensive boundary."""

    invalid = object.__new__(SearchResult)
    for field in fields(SearchResult):
        object.__setattr__(
            invalid,
            field.name,
            changes.get(field.name, getattr(result, field.name)),
        )
    return invalid


def test_validate_search_results_accepts_empty_and_valid_rankings() -> None:
    assert validate_search_results((), top_k=3) == ()
    results = (_result("a", "d1", 1.0, 1), _result("b", "d2", 0.5, 2))
    assert validate_search_results(results, top_k=2) == results


def test_stable_ranking_sorts_by_score_then_chunk_id_and_recomputes_rank() -> None:
    results = (
        _result("chunk-b", "doc-1", 1.0, 1),
        _result("chunk-a", "doc-2", 1.0, 2),
        _result("chunk-c", "doc-3", 2.0, 3),
    )
    ranked = stable_rank_results(results, top_k=3)
    assert tuple(result.chunk_id for result in ranked) == (
        "chunk-c",
        "chunk-a",
        "chunk-b",
    )
    assert tuple(result.rank for result in ranked) == (1, 2, 3)


def test_stable_ranking_is_deterministic_for_repeated_calls() -> None:
    results = (
        _result("b", "doc-b", 1.0, 1),
        _result("a", "doc-a", 1.0, 2),
    )
    assert stable_rank_results(results, top_k=2) == stable_rank_results(
        results, top_k=2
    )


def test_collapse_keeps_first_chunk_per_document_and_recomputes_ranks() -> None:
    results = (
        _result("a-1", "doc-a", 3.0, 1),
        _result("a-2", "doc-a", 2.0, 2),
        _result("b-1", "doc-b", 1.0, 3),
    )
    collapsed = collapse_to_documents(results)
    assert tuple(result.chunk_id for result in collapsed) == ("a-1", "b-1")
    assert tuple(result.rank for result in collapsed) == (1, 2)
    assert collapse_to_documents(()) == ()


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_validate_rejects_invalid_top_k(top_k: object) -> None:
    with pytest.raises(RetrieverContractError, match="positive integer"):
        validate_search_results((), top_k=top_k)  # type: ignore[arg-type]


def test_validate_rejects_too_many_results() -> None:
    results = (_result("a", "d1", 1.0, 1), _result("b", "d2", 0.5, 2))
    with pytest.raises(RetrieverContractError, match="returned 2 results"):
        validate_search_results(results, top_k=1)


def test_validate_rejects_duplicate_chunks() -> None:
    results = (_result("a", "d1", 1.0, 1), _result("a", "d1", 0.5, 2))
    with pytest.raises(RetrieverContractError, match="duplicate"):
        validate_search_results(results, top_k=2)


@pytest.mark.parametrize(
    "score", [float("nan"), float("inf"), -float("inf"), True, "1.0"]
)
def test_validate_rejects_non_finite_or_non_numeric_scores(score: object) -> None:
    valid = _result("a", "d1", 1.0, 1)
    with pytest.raises(RetrieverContractError, match="finite"):
        validate_search_results((_unsafe_replace(valid, score=score),), top_k=1)


@pytest.mark.parametrize("rank", [0, 2, True, 1.0, "1"])
def test_validate_rejects_invalid_or_non_contiguous_rank(rank: object) -> None:
    valid = _result("a", "d1", 1.0, 1)
    with pytest.raises(RetrieverContractError, match="contiguous"):
        validate_search_results((_unsafe_replace(valid, rank=rank),), top_k=1)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"chunk_id": ""}, "chunk_id"),
        ({"chunk_id": 1}, "chunk_id"),
        ({"document_id": ""}, "document_id"),
        ({"document_id": 1}, "document_id"),
    ],
)
def test_validate_rejects_empty_identifiers(
    changes: dict[str, object], message: str
) -> None:
    valid = _result("a", "d1", 1.0, 1)
    with pytest.raises(RetrieverContractError, match=message):
        validate_search_results((_unsafe_replace(valid, **changes),), top_k=1)
