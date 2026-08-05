"""Core corpus, query, and search result records."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field

from retrieval_lab.exceptions import (
    CorpusValidationError,
    DatasetValidationError,
    RetrieverContractError,
)

from ._validation import (
    normalize_json_mapping,
    require_finite_float,
    require_non_empty_string,
    require_non_negative_int,
    require_positive_int,
)
from .json_types import JSONValue


@dataclass(frozen=True)
class Document:
    """A source document to index and evaluate."""

    id: str
    text: str
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "id",
            require_non_empty_string(
                self.id,
                field_name="Document.id",
                error_type=CorpusValidationError,
            ),
        )
        object.__setattr__(
            self,
            "text",
            require_non_empty_string(
                self.text,
                field_name=f"Document[{self.id!r}].text",
                error_type=CorpusValidationError,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            normalize_json_mapping(
                self.metadata,
                field_name=f"Document[{self.id!r}].metadata",
                error_type=CorpusValidationError,
            ),
        )
        if self.source is not None:
            object.__setattr__(
                self,
                "source",
                require_non_empty_string(
                    self.source,
                    field_name=f"Document[{self.id!r}].source",
                    error_type=CorpusValidationError,
                ),
            )


@dataclass(frozen=True)
class Chunk:
    """A contiguous, zero-based slice of a parent document."""

    id: str
    document_id: str
    text: str
    start_offset: int
    end_offset: int
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("id", "document_id", "text"):
            object.__setattr__(
                self,
                field_name,
                require_non_empty_string(
                    getattr(self, field_name),
                    field_name=f"Chunk.{field_name}",
                    error_type=CorpusValidationError,
                ),
            )
        start = require_non_negative_int(
            self.start_offset,
            field_name=f"Chunk[{self.id!r}].start_offset",
            error_type=CorpusValidationError,
        )
        end = require_positive_int(
            self.end_offset,
            field_name=f"Chunk[{self.id!r}].end_offset",
            error_type=CorpusValidationError,
        )
        if start >= end:
            raise CorpusValidationError(
                f"Chunk[{self.id!r}] offsets must satisfy 0 <= start_offset "
                "< end_offset"
            )
        object.__setattr__(self, "start_offset", start)
        object.__setattr__(self, "end_offset", end)
        object.__setattr__(
            self,
            "metadata",
            normalize_json_mapping(
                self.metadata,
                field_name=f"Chunk[{self.id!r}].metadata",
                error_type=CorpusValidationError,
            ),
        )


@dataclass(frozen=True, init=False)
class EvaluationQuery:
    """An evaluation query with explicit document and chunk relevance."""

    id: str
    query: str
    relevant_document_ids: frozenset[str]
    relevant_chunk_ids: frozenset[str]
    metadata: Mapping[str, JSONValue]

    def __init__(
        self,
        id: str,
        query: str,
        relevant_document_ids: AbstractSet[str] = frozenset(),
        relevant_chunk_ids: AbstractSet[str] = frozenset(),
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> None:
        """Create a query and normalize mutable relevance sets."""

        normalized_id = require_non_empty_string(
            id,
            field_name="EvaluationQuery.id",
            error_type=DatasetValidationError,
        )
        normalized_query = require_non_empty_string(
            query,
            field_name=f"EvaluationQuery[{normalized_id!r}].query",
            error_type=DatasetValidationError,
        )
        document_ids = _normalize_relevant_ids(
            relevant_document_ids,
            field_name="relevant_document_ids",
            query_id=normalized_id,
        )
        chunk_ids = _normalize_relevant_ids(
            relevant_chunk_ids,
            field_name="relevant_chunk_ids",
            query_id=normalized_id,
        )
        if not document_ids and not chunk_ids:
            raise DatasetValidationError(
                f"EvaluationQuery[{normalized_id!r}] requires at least one "
                "relevant document or chunk ID"
            )
        normalized_metadata = normalize_json_mapping(
            {} if metadata is None else metadata,
            field_name=f"EvaluationQuery[{normalized_id!r}].metadata",
            error_type=DatasetValidationError,
        )

        object.__setattr__(self, "id", normalized_id)
        object.__setattr__(self, "query", normalized_query)
        object.__setattr__(self, "relevant_document_ids", document_ids)
        object.__setattr__(self, "relevant_chunk_ids", chunk_ids)
        object.__setattr__(self, "metadata", normalized_metadata)


def _normalize_relevant_ids(
    values: AbstractSet[str],
    *,
    field_name: str,
    query_id: str,
) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, AbstractSet):
        raise DatasetValidationError(
            f"EvaluationQuery[{query_id!r}].{field_name} must be a set of IDs"
        )
    normalized: set[str] = set()
    for value in values:
        normalized.add(
            require_non_empty_string(
                value,
                field_name=f"EvaluationQuery[{query_id!r}].{field_name} item",
                error_type=DatasetValidationError,
            )
        )
    return frozenset(normalized)


@dataclass(frozen=True)
class SearchResult:
    """One ranked chunk returned by a retriever."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("chunk_id", "document_id", "text"):
            object.__setattr__(
                self,
                field_name,
                require_non_empty_string(
                    getattr(self, field_name),
                    field_name=f"SearchResult.{field_name}",
                    error_type=RetrieverContractError,
                ),
            )
        object.__setattr__(
            self,
            "score",
            require_finite_float(
                self.score,
                field_name=f"SearchResult[{self.chunk_id!r}].score",
                error_type=RetrieverContractError,
            ),
        )
        object.__setattr__(
            self,
            "rank",
            require_positive_int(
                self.rank,
                field_name=f"SearchResult[{self.chunk_id!r}].rank",
                error_type=RetrieverContractError,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            normalize_json_mapping(
                self.metadata,
                field_name=f"SearchResult[{self.chunk_id!r}].metadata",
                error_type=RetrieverContractError,
            ),
        )


__all__ = ["Chunk", "Document", "EvaluationQuery", "SearchResult"]
