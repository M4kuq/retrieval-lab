"""Transparent deterministic keyword retrieval baseline."""

import unicodedata
from collections.abc import Sequence

from retrieval_lab.exceptions import RetrieverContractError
from retrieval_lab.models import Chunk, SearchResult
from retrieval_lab.retrievers.base import (
    BaseRetriever,
    _chunk_payload,
    _serialized_index_size_bytes,
)


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


class KeywordRetriever(BaseRetriever):
    """Rank chunks by distinct normalized query-term substring matches."""

    def __init__(self) -> None:
        """Create an unindexed keyword retriever."""
        self._chunks: tuple[Chunk, ...] | None = None

    @property
    def name(self) -> str:
        """Return the strategy name used by the evaluation runner."""
        return "keyword"

    @property
    def index_size_bytes(self) -> int | None:
        """Return the deterministic logical size of the indexed chunks.

        Keyword retrieval keeps the shared chunks and normalizes them at query
        time, so its index footprint is represented by the serialized chunk
        payload rather than by a process-specific Python object size.
        """

        if self._chunks is None:
            return None
        if not self._chunks:
            return 0
        return _serialized_index_size_bytes(
            [_chunk_payload(chunk) for chunk in self._chunks]
        )

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Replace the in-memory index, rejecting duplicate chunk identifiers."""
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise RetrieverContractError("chunks must be a sequence of Chunk records")
        indexed = tuple(chunks)
        seen: set[str] = set()
        duplicates: set[str] = set()
        for position, chunk in enumerate(indexed):
            if not isinstance(chunk, Chunk):
                raise RetrieverContractError(
                    f"chunks[{position}] must be a Chunk record"
                )
            if chunk.id in seen:
                duplicates.add(chunk.id)
            seen.add(chunk.id)

        if duplicates:
            duplicate_list = ", ".join(sorted(duplicates))
            raise RetrieverContractError(
                "keyword index received duplicate chunk identifiers: "
                f"{duplicate_list}; provide one chunk per identifier"
            )

        self._chunks = indexed

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Search the current index using normalized substring term matching."""
        if self._chunks is None:
            raise RetrieverContractError(
                "keyword retriever is not indexed; call index(chunks) before search"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RetrieverContractError(
                "top_k must be a positive integer; for example, top_k=5"
            )
        if not isinstance(query, str):
            raise RetrieverContractError("query must be a string")

        terms = set(_normalize(query).split())
        if not terms:
            return []

        scored: list[tuple[int, Chunk]] = []
        for chunk in self._chunks:
            normalized_text = _normalize(chunk.text)
            score = sum(term in normalized_text for term in terms)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].id))
        selected = scored[:top_k]
        return [
            SearchResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=float(score),
                rank=rank,
                metadata=dict(chunk.metadata),
            )
            for rank, (score, chunk) in enumerate(selected, start=1)
        ]
