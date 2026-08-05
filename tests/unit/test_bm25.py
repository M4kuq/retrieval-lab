"""Unit tests for the deterministic BM25 retriever."""

import math
from collections.abc import Sequence
from typing import cast

import pytest

from retrieval_lab.exceptions import RetrieverContractError
from retrieval_lab.models import Chunk
from retrieval_lab.retrievers.bm25 import BM25Retriever


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


def _term_score(
    *,
    document_count: int,
    document_frequency: int,
    term_frequency: int,
    query_frequency: int,
    document_length: int,
    average_document_length: float,
    k1: float,
    b: float,
) -> float:
    idf = math.log(
        1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
    )
    length_normalization = 1.0 - b + b * (document_length / average_document_length)
    saturated_frequency = (term_frequency * (k1 + 1.0)) / (
        term_frequency + k1 * length_normalization
    )
    return query_frequency * idf * saturated_frequency


def test_scores_match_hand_calculated_bm25_with_query_frequency() -> None:
    k1 = 1.5
    b = 0.75
    retriever = BM25Retriever(
        k1=k1,
        b=b,
        tokenizer=lambda value: value.split(),
    )
    retriever.index(
        [
            _chunk("one", "alpha alpha beta"),
            _chunk("two", "alpha gamma"),
        ]
    )

    results = retriever.search("alpha beta beta", top_k=2)

    expected_one = _term_score(
        document_count=2,
        document_frequency=2,
        term_frequency=2,
        query_frequency=1,
        document_length=3,
        average_document_length=2.5,
        k1=k1,
        b=b,
    ) + _term_score(
        document_count=2,
        document_frequency=1,
        term_frequency=1,
        query_frequency=2,
        document_length=3,
        average_document_length=2.5,
        k1=k1,
        b=b,
    )
    expected_two = _term_score(
        document_count=2,
        document_frequency=2,
        term_frequency=1,
        query_frequency=1,
        document_length=2,
        average_document_length=2.5,
        k1=k1,
        b=b,
    )

    assert retriever.name == "bm25"
    assert [result.chunk_id for result in results] == ["one", "two"]
    assert results[0].score == pytest.approx(expected_one)
    assert results[1].score == pytest.approx(expected_two)


def test_document_term_frequency_saturates_and_improves_score() -> None:
    retriever = BM25Retriever(tokenizer=lambda value: value.split())
    retriever.index(
        [
            _chunk("frequent", "term term filler"),
            _chunk("single", "term filler filler"),
        ]
    )

    results = retriever.search("term", top_k=2)

    assert [result.chunk_id for result in results] == ["frequent", "single"]
    assert results[0].score > results[1].score > 0.0


def test_repeated_query_terms_scale_score() -> None:
    retriever = BM25Retriever(tokenizer=lambda value: value.split())
    retriever.index([_chunk("only", "term")])

    once = retriever.search("term", top_k=1)[0].score
    twice = retriever.search("term term", top_k=1)[0].score

    assert twice == pytest.approx(2.0 * once)


def test_length_normalization_favors_shorter_document_at_equal_frequency() -> None:
    retriever = BM25Retriever(b=0.75, tokenizer=lambda value: value.split())
    retriever.index(
        [
            _chunk("long", "term filler filler filler"),
            _chunk("short", "term"),
        ]
    )

    results = retriever.search("term", top_k=2)

    assert [result.chunk_id for result in results] == ["short", "long"]
    assert results[0].score > results[1].score


def test_equal_scores_use_chunk_identifier_tie_break_and_contiguous_ranks() -> None:
    retriever = BM25Retriever(tokenizer=lambda value: value.split())
    retriever.index([_chunk("z", "same"), _chunk("a", "same")])

    results = retriever.search("same", top_k=2)

    assert [(result.chunk_id, result.rank) for result in results] == [
        ("a", 1),
        ("z", 2),
    ]


def test_default_tokenizer_handles_unsegmented_japanese_text() -> None:
    retriever = BM25Retriever()
    retriever.index(
        [
            _chunk("retrieval", "検索品質を評価します"),
            _chunk("generation", "回答生成を評価します"),
        ]
    )

    results = retriever.search("検索品質", top_k=2)

    assert [result.chunk_id for result in results] == ["retrieval"]


def test_default_tokenizer_applies_nfc_and_casefold() -> None:
    retriever = BM25Retriever()
    retriever.index([_chunk("normalized", "Die STRASSE und Caf\u00e9")])

    results = retriever.search("Stra\u00dfe Cafe\u0301", top_k=1)

    assert [result.chunk_id for result in results] == ["normalized"]


def test_custom_tokenizer_controls_boundaries_and_empty_tokens_are_ignored() -> None:
    retriever = BM25Retriever(tokenizer=lambda value: [*value.split("|"), "   "])
    retriever.index(
        [
            _chunk("exact", "Alpha||Beta"),
            _chunk("different-case", "alpha|beta"),
        ]
    )

    results = retriever.search("Alpha|", top_k=2)

    assert [result.chunk_id for result in results] == ["exact"]


@pytest.mark.parametrize("invalid", [0.0, -1.0, True, math.nan, math.inf, "1.5"])
def test_constructor_rejects_invalid_k1(invalid: object) -> None:
    with pytest.raises(RetrieverContractError, match="k1"):
        BM25Retriever(k1=cast(float, invalid))


@pytest.mark.parametrize("invalid", [-0.01, 1.01, True, math.nan, -math.inf, "0.75"])
def test_constructor_rejects_invalid_b(invalid: object) -> None:
    with pytest.raises(RetrieverContractError, match="b"):
        BM25Retriever(b=cast(float, invalid))


def test_constructor_rejects_non_callable_tokenizer() -> None:
    with pytest.raises(RetrieverContractError, match="tokenizer"):
        BM25Retriever(tokenizer=cast("object", "not callable"))


def test_search_requires_index_and_valid_arguments() -> None:
    retriever = BM25Retriever()

    with pytest.raises(RetrieverContractError, match="not indexed"):
        retriever.search("query", top_k=1)

    retriever.index([])
    for invalid_top_k in (0, -1, True):
        with pytest.raises(RetrieverContractError, match="top_k"):
            retriever.search("query", top_k=invalid_top_k)

    with pytest.raises(RetrieverContractError, match="query"):
        retriever.search(cast(str, object()), top_k=1)


def test_empty_index_empty_query_and_no_match_return_no_results() -> None:
    retriever = BM25Retriever()
    retriever.index([])
    assert retriever.search("query", top_k=5) == []

    retriever.index([_chunk("chunk", "retrieval")])
    assert retriever.search("", top_k=5) == []
    assert retriever.search("generation", top_k=5) == []


def test_documents_with_no_tokens_do_not_divide_by_zero() -> None:
    retriever = BM25Retriever()
    retriever.index([_chunk("punctuation", "!!!")])

    assert retriever.search("query", top_k=1) == []


def test_index_rejects_invalid_inputs() -> None:
    retriever = BM25Retriever()

    with pytest.raises(RetrieverContractError, match="sequence"):
        retriever.index(cast(Sequence[Chunk], "not chunks"))
    with pytest.raises(RetrieverContractError, match=r"chunks\[0\]"):
        retriever.index(cast(Sequence[Chunk], [object()]))


def test_duplicate_reindex_failure_is_atomic() -> None:
    retriever = BM25Retriever(tokenizer=lambda value: value.split())
    retriever.index([_chunk("original", "old term")])

    with pytest.raises(RetrieverContractError, match="duplicate"):
        retriever.index([_chunk("same", "new"), _chunk("same", "other")])

    assert [result.chunk_id for result in retriever.search("old", top_k=1)] == [
        "original"
    ]


def test_tokenizer_exception_during_reindex_is_chained_and_atomic() -> None:
    def tokenizer(value: str) -> Sequence[str]:
        if value == "explode":
            raise RuntimeError("tokenization failed")
        return value.split()

    retriever = BM25Retriever(tokenizer=tokenizer)
    retriever.index([_chunk("original", "old term")])

    with pytest.raises(RetrieverContractError, match="chunk 'bad'") as raised:
        retriever.index([_chunk("bad", "explode")])

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert [result.chunk_id for result in retriever.search("old", top_k=1)] == [
        "original"
    ]


@pytest.mark.parametrize(
    "invalid_tokens",
    [cast(Sequence[str], "one token"), cast(Sequence[str], ["valid", 1])],
)
def test_invalid_tokenizer_results_are_chained_to_contract_error(
    invalid_tokens: Sequence[str],
) -> None:
    retriever = BM25Retriever(tokenizer=lambda _value: invalid_tokens)

    with pytest.raises(RetrieverContractError, match="tokenizer failed") as raised:
        retriever.index([_chunk("chunk", "text")])

    assert isinstance(raised.value.__cause__, TypeError)


def test_query_tokenizer_exception_is_chained_to_contract_error() -> None:
    def tokenizer(value: str) -> Sequence[str]:
        if value == "bad query":
            raise ValueError("query rejected")
        return value.split()

    retriever = BM25Retriever(tokenizer=tokenizer)
    retriever.index([_chunk("chunk", "searchable")])

    with pytest.raises(RetrieverContractError, match="query") as raised:
        retriever.search("bad query", top_k=1)

    assert isinstance(raised.value.__cause__, ValueError)


def test_reindex_replaces_state_and_preserves_result_fields_without_mutation() -> None:
    retriever = BM25Retriever(tokenizer=lambda value: value.split())
    old = _chunk("old", "shared")
    replacement = _chunk("new", "shared", document_id="replacement-document")
    chunks = [replacement]
    retriever.index([old])

    retriever.index(chunks)
    results = retriever.search("shared", top_k=10)

    assert chunks == [replacement]
    assert replacement.text == "shared"
    assert len(results) == 1
    result = results[0]
    assert result.chunk_id == "new"
    assert result.document_id == "replacement-document"
    assert result.text == "shared"
    assert result.metadata == {"chunk": "new"}
    assert result.score > 0.0
    assert result.rank == 1
