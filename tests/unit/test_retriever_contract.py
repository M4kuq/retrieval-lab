"""Shared behavioral contract for built-in synchronous retrievers."""

from collections.abc import Callable, Sequence
from typing import cast

import pytest

from retrieval_lab.exceptions import RetrieverContractError
from retrieval_lab.models import Chunk
from retrieval_lab.retrievers.base import BaseRetriever
from retrieval_lab.retrievers.bm25 import BM25Retriever
from retrieval_lab.retrievers.dense import (
    DenseRetriever,
    EmbeddingModelMetadata,
)
from retrieval_lab.retrievers.keyword import KeywordRetriever

RetrieverFactory = Callable[[], BaseRetriever]


class _ContractEmbeddingBackend:
    metadata = EmbeddingModelMetadata(model_id="contract-fake")

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] for _ in texts]


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=f"document-{chunk_id}",
        text=text,
        start_offset=0,
        end_offset=len(text),
        metadata={"id": chunk_id},
    )


@pytest.fixture(
    params=[
        KeywordRetriever,
        BM25Retriever,
        lambda: DenseRetriever(backend=_ContractEmbeddingBackend()),
    ],
    ids=["keyword", "bm25", "dense"],
)
def retriever_factory(request: pytest.FixtureRequest) -> RetrieverFactory:
    return cast(RetrieverFactory, request.param)


def test_built_in_retrievers_require_index_before_search(
    retriever_factory: RetrieverFactory,
) -> None:
    with pytest.raises(RetrieverContractError, match="not indexed"):
        retriever_factory().search("term", top_k=1)


def test_built_in_retrievers_reject_invalid_index_records(
    retriever_factory: RetrieverFactory,
) -> None:
    retriever = retriever_factory()

    with pytest.raises(RetrieverContractError, match="sequence"):
        retriever.index(cast(Sequence[Chunk], "invalid"))
    with pytest.raises(RetrieverContractError, match=r"chunks\[0\]"):
        retriever.index(cast(Sequence[Chunk], [object()]))


def test_built_in_retrievers_return_stable_complete_rankings(
    retriever_factory: RetrieverFactory,
) -> None:
    retriever = retriever_factory()
    retriever.index([_chunk("z", "shared term"), _chunk("a", "shared term")])

    first = retriever.search("shared", top_k=2)
    second = retriever.search("shared", top_k=2)

    assert first == second
    assert [result.chunk_id for result in first] == ["a", "z"]
    assert [result.rank for result in first] == [1, 2]
    assert all(result.score > 0.0 for result in first)
    assert [result.document_id for result in first] == ["document-a", "document-z"]
    assert [result.text for result in first] == ["shared term", "shared term"]
    assert [result.metadata for result in first] == [{"id": "a"}, {"id": "z"}]


def test_built_in_retrievers_atomically_reject_duplicate_ids(
    retriever_factory: RetrieverFactory,
) -> None:
    retriever = retriever_factory()
    retriever.index([_chunk("original", "old term")])

    with pytest.raises(RetrieverContractError, match="duplicate"):
        retriever.index([_chunk("same", "new term"), _chunk("same", "other term")])

    assert [result.chunk_id for result in retriever.search("old", top_k=1)] == [
        "original"
    ]
