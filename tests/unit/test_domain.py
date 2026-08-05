from dataclasses import FrozenInstanceError

import pytest

from retrieval_lab.domain import Chunk, Document, EvaluationQuery, SearchResult
from retrieval_lab.exceptions import (
    CorpusValidationError,
    DatasetValidationError,
    RetrieverContractError,
)


def test_document_accepts_json_metadata_and_is_frozen() -> None:
    metadata = {"category": "日本語", "tags": ["rag"], "count": 2}

    document = Document("doc-1", "本文", metadata, source="manual.md")
    metadata["category"] = "changed"

    assert document.metadata["category"] == "日本語"
    with pytest.raises(FrozenInstanceError):
        document.id = "replacement"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [("id", ""), ("id", "  "), ("text", ""), ("text", "\n")],
)
def test_document_rejects_empty_identifiers_and_text(field: str, value: str) -> None:
    arguments = {"id": "doc-1", "text": "text", field: value}

    with pytest.raises(CorpusValidationError, match=field):
        Document(**arguments)


@pytest.mark.parametrize(
    "metadata",
    [
        {1: "not-a-string-key"},
        {"bad": object()},
        {"bad": float("nan")},
        {"bad": float("inf")},
    ],
)
def test_document_rejects_non_json_metadata(metadata: object) -> None:
    with pytest.raises(CorpusValidationError, match="metadata"):
        Document("doc-1", "text", metadata)  # type: ignore[arg-type]


def test_document_validates_optional_source() -> None:
    with pytest.raises(CorpusValidationError, match="source"):
        Document("doc-1", "text", source=" ")


def test_chunk_keeps_parent_offsets_and_metadata() -> None:
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        text="本文",
        start_offset=4,
        end_offset=6,
        metadata={"section": 1},
    )

    assert (chunk.start_offset, chunk.end_offset) == (4, 6)
    assert chunk.metadata == {"section": 1}


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 1), (0, 0), (2, 2), (3, 2), (True, 2), (0, False)],
)
def test_chunk_rejects_invalid_offsets(start: object, end: object) -> None:
    with pytest.raises(CorpusValidationError, match="offset"):
        Chunk(
            id="chunk-1",
            document_id="doc-1",
            text="text",
            start_offset=start,  # type: ignore[arg-type]
            end_offset=end,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field", ["id", "document_id", "text"])
def test_chunk_rejects_empty_required_strings(field: str) -> None:
    arguments = {
        "id": "chunk-1",
        "document_id": "doc-1",
        "text": "text",
        "start_offset": 0,
        "end_offset": 1,
        field: " ",
    }

    with pytest.raises(CorpusValidationError, match=field):
        Chunk(**arguments)


def test_evaluation_query_normalizes_relevant_sets() -> None:
    documents = {"doc-1"}
    chunks = {"chunk-1"}

    query = EvaluationQuery(
        id="q-1",
        query="検索クエリ",
        relevant_document_ids=documents,
        relevant_chunk_ids=chunks,
        metadata={"category": "how-to"},
    )
    documents.add("doc-2")

    assert query.relevant_document_ids == frozenset({"doc-1"})
    assert query.relevant_chunk_ids == frozenset({"chunk-1"})
    assert isinstance(query.relevant_document_ids, frozenset)


def test_evaluation_query_accepts_only_chunk_relevance() -> None:
    query = EvaluationQuery("q-1", "query", relevant_chunk_ids={"chunk-1"})

    assert not query.relevant_document_ids


def test_evaluation_query_requires_positive_relevance() -> None:
    with pytest.raises(DatasetValidationError, match="at least one"):
        EvaluationQuery("q-1", "query")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", ""),
        ("query", " "),
        ("relevant_document_ids", ["doc-1"]),
        ("relevant_document_ids", {""}),
    ],
)
def test_evaluation_query_rejects_invalid_fields(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "id": "q-1",
        "query": "query",
        "relevant_document_ids": {"doc-1"},
        field: value,
    }

    with pytest.raises(DatasetValidationError):
        EvaluationQuery(**arguments)  # type: ignore[arg-type]


def test_search_result_normalizes_numeric_score() -> None:
    result = SearchResult("chunk-1", "doc-1", "text", 1, 1)

    assert result.score == 1.0
    assert isinstance(result.score, float)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), True])
def test_search_result_requires_finite_score(score: object) -> None:
    with pytest.raises(RetrieverContractError, match="score"):
        SearchResult(
            "chunk-1",
            "doc-1",
            "text",
            score,  # type: ignore[arg-type]
            1,
        )


@pytest.mark.parametrize("rank", [0, -1, True, 1.5])
def test_search_result_requires_one_based_rank(rank: object) -> None:
    with pytest.raises(RetrieverContractError, match="rank"):
        SearchResult(
            "chunk-1",
            "doc-1",
            "text",
            1.0,
            rank,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field", ["chunk_id", "document_id", "text"])
def test_search_result_rejects_empty_required_strings(field: str) -> None:
    arguments = {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "text": "text",
        "score": 1.0,
        "rank": 1,
        field: " ",
    }

    with pytest.raises(RetrieverContractError, match=field):
        SearchResult(**arguments)
