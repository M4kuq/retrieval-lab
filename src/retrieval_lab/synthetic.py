"""Provider-independent synthetic evaluation dataset generation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from retrieval_lab.dataset_authoring import (
    DatasetDraft,
    DatasetProvenance,
    DraftQuery,
)
from retrieval_lab.domain import Document
from retrieval_lab.exceptions import DatasetValidationError


class SyntheticQueryGenerator(Protocol):
    """Contract for a caller-controlled synthetic query generator."""

    name: str

    def generate(
        self,
        documents: Sequence[Document],
        *,
        count: int,
        seed: int,
    ) -> Sequence[DraftQuery]:
        """Generate candidate queries with positive document judgments."""
        ...


SyntheticGenerateCallable = Callable[
    [Sequence[Document], int, int],
    Sequence[DraftQuery],
]


@dataclass(frozen=True)
class CallableSyntheticGenerator:
    """Adapt a synchronous Python callable to ``SyntheticQueryGenerator``."""

    name: str
    callback: SyntheticGenerateCallable

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DatasetValidationError(
                "CallableSyntheticGenerator.name must be a non-empty string"
            )
        if not callable(self.callback):
            raise DatasetValidationError(
                "CallableSyntheticGenerator.callback must be callable"
            )

    def generate(
        self,
        documents: Sequence[Document],
        *,
        count: int,
        seed: int,
    ) -> Sequence[DraftQuery]:
        """Invoke the wrapped callable without adding network behavior."""

        return self.callback(documents, count, seed)


def generate_synthetic_draft(
    documents: Sequence[Document],
    *,
    generator: SyntheticQueryGenerator,
    count: int,
    seed: int = 42,
) -> DatasetDraft:
    """Generate an explicitly experimental, unreviewed document-level draft."""

    normalized_documents = _validate_documents(documents)
    normalized_count = _require_positive_int(count, "count")
    normalized_seed = _require_int(seed, "seed")
    generator_name = _generator_name(generator)

    try:
        generated = generator.generate(
            normalized_documents,
            count=normalized_count,
            seed=normalized_seed,
        )
    except DatasetValidationError:
        raise
    except Exception as exc:
        raise DatasetValidationError(
            f"synthetic generator {generator_name!r} failed: {exc}"
        ) from exc

    queries = _validate_generated_queries(
        generated,
        count=normalized_count,
        document_ids=frozenset(document.id for document in normalized_documents),
    )
    return DatasetDraft(
        queries,
        relevance_level="document",
        provenance=DatasetProvenance(
            origin="synthetic",
            review_status="unreviewed",
            generator=generator_name,
            notes=f"generated with seed={normalized_seed}",
        ),
    )


def mark_draft_in_review(
    draft: DatasetDraft,
    *,
    notes: str | None = None,
) -> DatasetDraft:
    """Mark a dataset draft as undergoing human review."""

    normalized = _require_draft(draft)
    return normalized.with_provenance(
        replace(
            normalized.provenance,
            review_status="in_review",
            notes=_select_notes(normalized, notes),
        )
    )


def mark_draft_reviewed(
    draft: DatasetDraft,
    *,
    notes: str | None = None,
) -> DatasetDraft:
    """Mark human review complete without changing the dataset's origin."""

    normalized = _require_draft(draft)
    return normalized.with_provenance(
        replace(
            normalized.provenance,
            review_status="reviewed",
            notes=_select_notes(normalized, notes),
        )
    )


def _validate_documents(documents: Sequence[Document]) -> tuple[Document, ...]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise DatasetValidationError("documents must be a sequence of Document values")
    normalized = tuple(documents)
    if not normalized:
        raise DatasetValidationError("documents must not be empty")
    seen: set[str] = set()
    for position, document in enumerate(normalized):
        if not isinstance(document, Document):
            raise DatasetValidationError(f"documents[{position}] must be a Document")
        if document.id in seen:
            raise DatasetValidationError(
                f"document IDs must be unique; duplicate {document.id!r}"
            )
        seen.add(document.id)
    return normalized


def _validate_generated_queries(
    values: object,
    *,
    count: int,
    document_ids: frozenset[str],
) -> tuple[DraftQuery, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise DatasetValidationError(
            "synthetic generator output must be a sequence of DraftQuery values"
        )
    queries = tuple(values)
    if len(queries) != count:
        raise DatasetValidationError(
            f"synthetic generator returned {len(queries)} queries; expected {count}"
        )
    seen: set[str] = set()
    for position, query in enumerate(queries):
        if not isinstance(query, DraftQuery):
            raise DatasetValidationError(
                f"synthetic generator output[{position}] must be a DraftQuery"
            )
        if query.id in seen:
            raise DatasetValidationError(
                f"synthetic query IDs must be unique; duplicate {query.id!r}"
            )
        seen.add(query.id)
        if not query.complete:
            raise DatasetValidationError(
                f"synthetic query {query.id!r} requires positive relevance"
            )
        unknown = sorted(set(query.relevance) - document_ids)
        if unknown:
            raise DatasetValidationError(
                f"synthetic query {query.id!r} references unknown documents: {unknown}"
            )
    return queries


def _generator_name(generator: SyntheticQueryGenerator) -> str:
    try:
        name = generator.name
        method = generator.generate
    except AttributeError as exc:
        raise DatasetValidationError(
            "generator must provide non-empty name and generate()"
        ) from exc
    if not isinstance(name, str) or not name.strip() or not callable(method):
        raise DatasetValidationError(
            "generator must provide non-empty name and callable generate()"
        )
    return name


def _require_positive_int(value: object, name: str) -> int:
    normalized = _require_int(value, name)
    if normalized <= 0:
        raise DatasetValidationError(f"{name} must be a positive integer")
    return normalized


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetValidationError(f"{name} must be an integer")
    return value


def _require_draft(draft: DatasetDraft) -> DatasetDraft:
    if not isinstance(draft, DatasetDraft):
        raise DatasetValidationError("draft must be a DatasetDraft")
    return draft


def _select_notes(draft: DatasetDraft, notes: str | None) -> str | None:
    if notes is None:
        return draft.provenance.notes
    if not isinstance(notes, str) or not notes.strip():
        raise DatasetValidationError("review notes must be non-empty or None")
    return notes


__all__ = [
    "CallableSyntheticGenerator",
    "SyntheticGenerateCallable",
    "SyntheticQueryGenerator",
    "generate_synthetic_draft",
    "mark_draft_in_review",
    "mark_draft_reviewed",
]
