"""Tests for the deterministic keyword baseline."""

from collections.abc import Sequence
from typing import cast

import pytest

from retrieval_lab.exceptions import RetrieverContractError
from retrieval_lab.models import Chunk
from retrieval_lab.retrievers import KeywordRetriever


def _chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id or f"doc-{chunk_id}",
        text=text,
        start_offset=0,
        end_offset=len(text),
        metadata={"chunk": chunk_id},
    )


def test_japanese_terms_rank_by_distinct_substring_matches() -> None:
    retriever = KeywordRetriever()
    assert retriever.name == "keyword"
    retriever.index(
        [
            _chunk("chunk-b", "RAGでは検索品質の評価が重要です。"),
            _chunk("chunk-a", "検索システムを作ります。"),
            _chunk("chunk-c", "評価だけを説明します。"),
        ]
    )

    results = retriever.search("検索 検索 品質", top_k=3)

    assert [(result.chunk_id, result.score, result.rank) for result in results] == [
        ("chunk-b", 2.0, 1),
        ("chunk-a", 1.0, 2),
    ]


def test_normalization_uses_unicode_nfc_and_casefold() -> None:
    retriever = KeywordRetriever()
    retriever.index(
        [
            _chunk("strasse", "Die STRASSE ist lang."),
            _chunk("accent", "Caf\u00e9"),
        ]
    )

    casefolded = retriever.search("Stra\u00dfe", top_k=2)
    composed = retriever.search("Cafe\u0301", top_k=2)

    assert [result.chunk_id for result in casefolded] == ["strasse"]
    assert [result.chunk_id for result in composed] == ["accent"]


def test_equal_scores_break_ties_by_chunk_id_and_recompute_rank() -> None:
    retriever = KeywordRetriever()
    retriever.index([_chunk("z", "term"), _chunk("a", "term")])

    results = retriever.search("term", top_k=2)

    assert [(result.chunk_id, result.rank) for result in results] == [
        ("a", 1),
        ("z", 2),
    ]


def test_no_matches_and_whitespace_query_return_empty_results() -> None:
    retriever = KeywordRetriever()
    retriever.index([_chunk("chunk", "retrieval")])

    assert retriever.search("generation", top_k=5) == []
    assert retriever.search(" \t\n", top_k=5) == []


def test_search_requires_index_and_positive_integer_top_k() -> None:
    retriever = KeywordRetriever()

    with pytest.raises(RetrieverContractError, match="not indexed"):
        retriever.search("query", top_k=1)

    retriever.index([])
    for invalid_top_k in (0, -1, True):
        with pytest.raises(RetrieverContractError, match="top_k"):
            retriever.search("query", top_k=invalid_top_k)

    with pytest.raises(RetrieverContractError, match="query"):
        retriever.search(cast(str, object()), top_k=1)


def test_index_rejects_invalid_chunk_inputs_with_library_error() -> None:
    retriever = KeywordRetriever()

    with pytest.raises(RetrieverContractError, match="sequence"):
        retriever.index(cast(Sequence[Chunk], "not chunks"))
    with pytest.raises(RetrieverContractError, match=r"chunks\[0\]"):
        retriever.index(cast(Sequence[Chunk], [object()]))


def test_duplicate_chunk_ids_are_rejected_without_replacing_index() -> None:
    retriever = KeywordRetriever()
    original = _chunk("original", "old term")
    retriever.index([original])

    with pytest.raises(RetrieverContractError, match="duplicate"):
        retriever.index([_chunk("duplicate", "new term"), _chunk("duplicate", "other")])

    assert [result.chunk_id for result in retriever.search("old", top_k=1)] == [
        "original"
    ]


def test_reindex_replaces_previous_state_and_preserves_result_fields() -> None:
    retriever = KeywordRetriever()
    retriever.index([_chunk("old", "shared")])
    replacement = _chunk("new", "shared", document_id="new-document")

    retriever.index([replacement])
    results = retriever.search("shared", top_k=10)

    assert len(results) == 1
    result = results[0]
    assert result.chunk_id == "new"
    assert result.document_id == "new-document"
    assert result.text == "shared"
    assert result.metadata == {"chunk": "new"}
    assert result.score == 1.0
    assert result.rank == 1
