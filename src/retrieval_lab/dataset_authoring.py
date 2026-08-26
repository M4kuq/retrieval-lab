"""Typed authoring models for building trustworthy evaluation datasets."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

from retrieval_lab.datasets import EvaluationDataset, RelevanceLevel
from retrieval_lab.domain import EvaluationQuery
from retrieval_lab.domain._validation import normalize_json_mapping
from retrieval_lab.domain.json_types import JSONValue
from retrieval_lab.exceptions import DatasetValidationError

DatasetOrigin = Literal["unknown", "human", "synthetic", "mixed"]
DatasetReviewStatus = Literal["unreviewed", "in_review", "reviewed"]


@dataclass(frozen=True)
class DatasetProvenance:
    """Dataset-level origin and human-review state."""

    origin: DatasetOrigin = "unknown"
    review_status: DatasetReviewStatus = "unreviewed"
    generator: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.origin not in {"unknown", "human", "synthetic", "mixed"}:
            raise DatasetValidationError(
                "DatasetProvenance.origin must be unknown, human, synthetic, or mixed"
            )
        if self.review_status not in {"unreviewed", "in_review", "reviewed"}:
            raise DatasetValidationError(
                "DatasetProvenance.review_status must be unreviewed, in_review, "
                "or reviewed"
            )
        for name in ("generator", "notes"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise DatasetValidationError(
                        f"DatasetProvenance.{name} must be a non-empty string or None"
                    )
                object.__setattr__(self, name, unicodedata.normalize("NFC", value))

    @property
    def human_reviewed(self) -> bool:
        """Return whether the dataset has completed human review."""

        return self.review_status == "reviewed"

    @property
    def reliability(self) -> str:
        """Return the conservative reliability label used by reports and demos."""

        if self.origin in {"synthetic", "mixed"} and not self.human_reviewed:
            return "experimental"
        if self.origin == "unknown" or not self.human_reviewed:
            return "unverified"
        return "reviewed"

    def to_dict(self) -> dict[str, str | bool | None]:
        """Return a JSON-compatible provenance record."""

        return {
            "origin": self.origin,
            "review_status": self.review_status,
            "human_reviewed": self.human_reviewed,
            "reliability": self.reliability,
            "generator": self.generator,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DraftQuery:
    """One editable query whose relevance judgments may still be incomplete."""

    id: str
    query: str
    relevance: Mapping[str, int] = field(default_factory=dict)
    reference_answer: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("id", "query"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DatasetValidationError(f"DraftQuery.{name} must be non-empty")
            object.__setattr__(self, name, unicodedata.normalize("NFC", value))

        if not isinstance(self.relevance, Mapping):
            raise DatasetValidationError("DraftQuery.relevance must be a mapping")
        normalized_relevance: dict[str, int] = {}
        for relevance_id, grade in self.relevance.items():
            if not isinstance(relevance_id, str) or not relevance_id.strip():
                raise DatasetValidationError(
                    "DraftQuery.relevance keys must be non-empty strings"
                )
            if isinstance(grade, bool) or not isinstance(grade, int) or grade < 1:
                raise DatasetValidationError(
                    "DraftQuery relevance grades must be integers >= 1"
                )
            normalized_relevance[unicodedata.normalize("NFC", relevance_id)] = grade
        object.__setattr__(self, "relevance", MappingProxyType(normalized_relevance))

        if self.reference_answer is not None:
            if (
                not isinstance(self.reference_answer, str)
                or not self.reference_answer.strip()
            ):
                raise DatasetValidationError(
                    "DraftQuery.reference_answer must be non-empty or None"
                )
            object.__setattr__(
                self,
                "reference_answer",
                unicodedata.normalize("NFC", self.reference_answer),
            )
        object.__setattr__(
            self,
            "metadata",
            normalize_json_mapping(
                self.metadata,
                field_name=f"DraftQuery[{self.id!r}].metadata",
                error_type=DatasetValidationError,
            ),
        )

    @property
    def complete(self) -> bool:
        """Return whether the query has at least one positive judgment."""

        return bool(self.relevance)

    def with_relevance(self, relevance_id: str, grade: int = 1) -> Self:
        """Return a copy with one positive relevance judgment added or replaced."""

        updated = dict(self.relevance)
        updated[relevance_id] = grade
        return replace(self, relevance=updated)

    def without_relevance(self, relevance_id: str) -> Self:
        """Return a copy with one relevance judgment removed if present."""

        updated = dict(self.relevance)
        updated.pop(relevance_id, None)
        return replace(self, relevance=updated)


@dataclass(frozen=True, init=False)
class DatasetDraft:
    """Immutable authoring state that cannot be evaluated until finalized."""

    queries: tuple[DraftQuery, ...]
    relevance_level: RelevanceLevel
    provenance: DatasetProvenance

    def __init__(
        self,
        queries: Sequence[DraftQuery] = (),
        *,
        relevance_level: RelevanceLevel = "document",
        provenance: DatasetProvenance | None = None,
    ) -> None:
        if relevance_level not in {"document", "chunk"}:
            raise DatasetValidationError(
                "DatasetDraft.relevance_level must be document or chunk"
            )
        if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
            raise DatasetValidationError("DatasetDraft.queries must be a sequence")
        normalized = tuple(queries)
        seen: set[str] = set()
        for position, query in enumerate(normalized):
            if not isinstance(query, DraftQuery):
                raise DatasetValidationError(
                    f"DatasetDraft.queries[{position}] must be a DraftQuery"
                )
            if query.id in seen:
                raise DatasetValidationError(
                    f"DatasetDraft query IDs must be unique; duplicate {query.id!r}"
                )
            seen.add(query.id)
        normalized_provenance = provenance or DatasetProvenance()
        if not isinstance(normalized_provenance, DatasetProvenance):
            raise DatasetValidationError(
                "DatasetDraft.provenance must be a DatasetProvenance"
            )
        object.__setattr__(self, "queries", normalized)
        object.__setattr__(self, "relevance_level", relevance_level)
        object.__setattr__(self, "provenance", normalized_provenance)

    @property
    def complete(self) -> bool:
        """Return whether all draft queries have positive relevance judgments."""

        return bool(self.queries) and all(query.complete for query in self.queries)

    @property
    def pending_query_ids(self) -> tuple[str, ...]:
        """Return query IDs that still require relevance judgments."""

        return tuple(query.id for query in self.queries if not query.complete)

    def add_query(self, query: DraftQuery) -> Self:
        """Return a copy with one new query appended."""

        return DatasetDraft(
            (*self.queries, query),
            relevance_level=self.relevance_level,
            provenance=self.provenance,
        )

    def replace_query(self, query: DraftQuery) -> Self:
        """Return a copy replacing an existing query by identifier."""

        replaced = False
        values: list[DraftQuery] = []
        for existing in self.queries:
            if existing.id == query.id:
                values.append(query)
                replaced = True
            else:
                values.append(existing)
        if not replaced:
            raise DatasetValidationError(f"unknown draft query ID {query.id!r}")
        return DatasetDraft(
            values,
            relevance_level=self.relevance_level,
            provenance=self.provenance,
        )

    def with_provenance(self, provenance: DatasetProvenance) -> Self:
        """Return a copy with updated dataset provenance."""

        return DatasetDraft(
            self.queries,
            relevance_level=self.relevance_level,
            provenance=provenance,
        )

    def finalize(self) -> EvaluationDataset:
        """Build a validated EvaluationDataset when every query is complete."""

        if not self.queries:
            raise DatasetValidationError("cannot finalize an empty DatasetDraft")
        if not self.complete:
            pending = ", ".join(self.pending_query_ids)
            raise DatasetValidationError(
                f"cannot finalize draft with pending relevance judgments: {pending}"
            )
        queries: list[EvaluationQuery] = []
        grades: dict[str, Mapping[str, int]] = {}
        answers: dict[str, str] = {}
        for draft_query in self.queries:
            relevance_ids = frozenset(draft_query.relevance)
            queries.append(
                EvaluationQuery(
                    id=draft_query.id,
                    query=draft_query.query,
                    relevant_document_ids=(
                        relevance_ids
                        if self.relevance_level == "document"
                        else frozenset()
                    ),
                    relevant_chunk_ids=(
                        relevance_ids
                        if self.relevance_level == "chunk"
                        else frozenset()
                    ),
                    metadata=draft_query.metadata,
                )
            )
            grades[draft_query.id] = draft_query.relevance
            if draft_query.reference_answer is not None:
                answers[draft_query.id] = draft_query.reference_answer
        return EvaluationDataset(
            queries,
            relevance_level=self.relevance_level,
            relevance_grades_by_query=grades,
            reference_answers_by_query=answers,
        )

    def to_jsonl(self) -> str:
        """Serialize draft queries in deterministic canonical evaluation JSONL."""

        records: list[str] = []
        for query in self.queries:
            record: dict[str, object] = {
                "query_id": query.id,
                "query": query.query,
                "relevant": [
                    {"id": relevance_id, "relevance": grade}
                    for relevance_id, grade in sorted(query.relevance.items())
                ],
            }
            if query.reference_answer is not None:
                record["reference_answer"] = query.reference_answer
            if query.metadata:
                record["metadata"] = dict(query.metadata)
            records.append(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return "" if not records else "\n".join(records) + "\n"

    def save_bundle(self, directory: str | Path) -> tuple[Path, Path]:
        """Persist JSONL plus a sidecar manifest containing trust metadata."""

        target = Path(directory)
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DatasetValidationError(
                f"cannot create dataset bundle directory {target}: {exc}"
            ) from exc
        dataset_path = target / "evaluation.jsonl"
        manifest_path = target / "dataset-manifest.json"
        manifest = {
            "schema_version": 1,
            "relevance_level": self.relevance_level,
            "provenance": self.provenance.to_dict(),
            "query_count": len(self.queries),
            "pending_query_ids": list(self.pending_query_ids),
        }
        _write_text(dataset_path, self.to_jsonl())
        _write_text(
            manifest_path,
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
        )
        return dataset_path, manifest_path

    @classmethod
    def from_dataset(
        cls,
        dataset: EvaluationDataset,
        *,
        provenance: DatasetProvenance | None = None,
    ) -> Self:
        """Create editable authoring state from an existing validated dataset."""

        if not isinstance(dataset, EvaluationDataset):
            raise DatasetValidationError("dataset must be an EvaluationDataset")
        queries = [
            DraftQuery(
                id=query.id,
                query=query.query,
                relevance=dataset.relevance_grades_by_query[query.id],
                reference_answer=dataset.reference_answers_by_query.get(query.id),
                metadata=query.metadata,
            )
            for query in dataset.queries
        ]
        return cls(
            queries,
            relevance_level=dataset.relevance_level,
            provenance=provenance,
        )


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    except OSError as exc:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise DatasetValidationError(
            f"cannot write dataset artifact {path}: {exc}"
        ) from exc


__all__ = [
    "DatasetDraft",
    "DatasetOrigin",
    "DatasetProvenance",
    "DatasetReviewStatus",
    "DraftQuery",
]
