"""Validated evaluation datasets for Retrieval Lab."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self, TypeVar, cast

from retrieval_lab.domain import Chunk, Document, EvaluationQuery
from retrieval_lab.domain._validation import normalize_json_mapping
from retrieval_lab.exceptions import CorpusValidationError, DatasetValidationError

RelevanceLevel = Literal["document", "chunk"]
_RecordT = TypeVar("_RecordT", Document, Chunk)


@dataclass(frozen=True, init=False)
class EvaluationDataset:
    """An immutable set of queries and their graded relevance judgments."""

    queries: tuple[EvaluationQuery, ...]
    relevance_level: RelevanceLevel
    relevance_grades_by_query: Mapping[str, Mapping[str, int]]
    reference_answers_by_query: Mapping[str, str]

    def __init__(
        self,
        queries: Sequence[EvaluationQuery],
        *,
        relevance_level: RelevanceLevel = "document",
        relevance_grades_by_query: Mapping[str, Mapping[str, int]] | None = None,
        reference_answers_by_query: Mapping[str, str] | None = None,
    ) -> None:
        """Validate and defensively copy all dataset inputs."""

        normalized_queries = _validate_queries(queries)
        normalized_level = _validate_relevance_level(relevance_level)
        selected_ids = {
            query.id: (
                query.relevant_document_ids
                if normalized_level == "document"
                else query.relevant_chunk_ids
            )
            for query in normalized_queries
        }
        for query_id, relevant_ids in selected_ids.items():
            if not relevant_ids:
                raise DatasetValidationError(
                    f"query {query_id!r} has no positive {normalized_level} "
                    "relevance and cannot be evaluated"
                )

        grades = _normalize_relevance_grades(
            selected_ids,
            relevance_grades_by_query,
        )
        answers = _normalize_reference_answers(
            frozenset(selected_ids),
            reference_answers_by_query,
        )

        object.__setattr__(self, "queries", normalized_queries)
        object.__setattr__(self, "relevance_level", normalized_level)
        object.__setattr__(self, "relevance_grades_by_query", grades)
        object.__setattr__(self, "reference_answers_by_query", answers)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        relevance_level: RelevanceLevel = "document",
    ) -> Self:
        """Load the canonical native evaluation JSONL format from ``path``."""

        source = _coerce_path(path)
        level = _validate_relevance_level(relevance_level)
        text = _read_utf8(source)
        if not text:
            raise _dataset_error(source, 1, "dataset must not be empty")

        queries: list[EvaluationQuery] = []
        grades_by_query: dict[str, Mapping[str, int]] = {}
        answers_by_query: dict[str, str] = {}
        query_lines: dict[str, int] = {}

        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise _dataset_error(
                    source,
                    line_number,
                    "blank lines are not valid JSONL records",
                )
            record = _parse_json_object(line, path=source, line_number=line_number)
            query, grades, reference_answer = _parse_dataset_record(
                record,
                relevance_level=level,
                path=source,
                line_number=line_number,
            )
            if query.id in query_lines:
                raise _dataset_error(
                    source,
                    line_number,
                    f"duplicate query_id {query.id!r}; first seen on line "
                    f"{query_lines[query.id]}",
                )
            query_lines[query.id] = line_number
            queries.append(query)
            grades_by_query[query.id] = grades
            if reference_answer is not None:
                answers_by_query[query.id] = reference_answer

        if not queries:
            raise _dataset_error(source, 1, "dataset must not be empty")
        return cls(
            queries,
            relevance_level=level,
            relevance_grades_by_query=grades_by_query,
            reference_answers_by_query=answers_by_query,
        )

    def relevance_for(self, query_id: str) -> Mapping[str, int]:
        """Return the positive graded judgments for one query identifier."""

        try:
            return self.relevance_grades_by_query[query_id]
        except KeyError as exc:
            raise DatasetValidationError(
                f"unknown query_id {query_id!r}; use an ID present in this dataset"
            ) from exc

    def reference_answer_for(self, query_id: str) -> str | None:
        """Return a query's optional reference answer."""

        if query_id not in {query.id for query in self.queries}:
            raise DatasetValidationError(
                f"unknown query_id {query_id!r}; use an ID present in this dataset"
            )
        return self.reference_answers_by_query.get(query_id)

    def validate(
        self,
        *,
        documents: Sequence[Document] | None = None,
        chunks: Sequence[Chunk] | None = None,
    ) -> None:
        """Validate that every positive judgment exists in the supplied corpus."""

        validate_dataset(self, documents=documents, chunks=chunks)


def validate_dataset(
    dataset: EvaluationDataset,
    *,
    documents: Sequence[Document] | None = None,
    chunks: Sequence[Chunk] | None = None,
) -> None:
    """Validate dataset-to-corpus identifiers at the explicit relevance level."""

    if not isinstance(dataset, EvaluationDataset):
        raise DatasetValidationError("dataset must be an EvaluationDataset")

    if dataset.relevance_level == "document":
        available = _validate_corpus_records(
            documents,
            record_type=Document,
            record_name="document",
        )
    else:
        available = _validate_corpus_records(
            chunks,
            record_type=Chunk,
            record_name="chunk",
        )

    missing_by_query: dict[str, tuple[str, ...]] = {}
    for query in dataset.queries:
        missing = tuple(
            sorted(set(dataset.relevance_grades_by_query[query.id]) - available)
        )
        if missing:
            missing_by_query[query.id] = missing

    if missing_by_query:
        details = "; ".join(
            f"{query_id}: {', '.join(missing_ids)}"
            for query_id, missing_ids in sorted(missing_by_query.items())
        )
        raise DatasetValidationError(
            f"gold {dataset.relevance_level} IDs are missing from the corpus "
            f"({details}); add those records or correct the evaluation JSONL"
        )


def _validate_queries(
    queries: Sequence[EvaluationQuery],
) -> tuple[EvaluationQuery, ...]:
    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
        raise DatasetValidationError(
            "queries must be a sequence of EvaluationQuery values"
        )
    normalized = tuple(queries)
    if not normalized:
        raise DatasetValidationError("queries must not be empty")
    query_lines: dict[str, int] = {}
    for position, query in enumerate(normalized):
        if not isinstance(query, EvaluationQuery):
            raise DatasetValidationError(
                f"queries[{position}] must be an EvaluationQuery"
            )
        if query.id in query_lines:
            raise DatasetValidationError(
                f"query IDs must be unique; duplicate {query.id!r} at positions "
                f"{query_lines[query.id]} and {position}"
            )
        query_lines[query.id] = position
    return normalized


def _validate_relevance_level(value: object) -> RelevanceLevel:
    if value not in ("document", "chunk"):
        raise DatasetValidationError(
            "relevance_level must be either 'document' or 'chunk'"
        )
    return value


def _normalize_relevance_grades(
    selected_ids: Mapping[str, frozenset[str]],
    supplied: Mapping[str, Mapping[str, int]] | None,
) -> Mapping[str, Mapping[str, int]]:
    if supplied is None:
        return MappingProxyType(
            {
                query_id: MappingProxyType(
                    {identifier: 1 for identifier in sorted(relevant_ids)}
                )
                for query_id, relevant_ids in selected_ids.items()
            }
        )
    if not isinstance(supplied, Mapping):
        raise DatasetValidationError(
            "relevance_grades_by_query must be a mapping keyed by query ID"
        )
    supplied_query_ids = set(supplied)
    expected_query_ids = set(selected_ids)
    if supplied_query_ids != expected_query_ids:
        missing = sorted(expected_query_ids - supplied_query_ids)
        unknown = sorted(supplied_query_ids - expected_query_ids)
        raise DatasetValidationError(
            "relevance_grades_by_query keys must exactly match query IDs; "
            f"missing={missing}, unknown={unknown}"
        )

    normalized: dict[str, Mapping[str, int]] = {}
    for query_id, expected_ids in selected_ids.items():
        values = supplied[query_id]
        if not isinstance(values, Mapping):
            raise DatasetValidationError(
                f"relevance grades for query {query_id!r} must be a mapping"
            )
        if set(values) != set(expected_ids):
            raise DatasetValidationError(
                f"relevance grade IDs for query {query_id!r} must exactly match "
                "its positive relevance IDs"
            )
        grades: dict[str, int] = {}
        for identifier, grade in values.items():
            if not isinstance(identifier, str) or not identifier.strip():
                raise DatasetValidationError(
                    f"relevance ID for query {query_id!r} must be non-empty"
                )
            if isinstance(grade, bool) or not isinstance(grade, int) or grade < 1:
                raise DatasetValidationError(
                    f"relevance for query {query_id!r}, ID {identifier!r} must "
                    "be an integer >= 1 (booleans are not valid integers)"
                )
            grades[identifier] = grade
        normalized[query_id] = MappingProxyType(grades)
    return MappingProxyType(normalized)


def _normalize_reference_answers(
    query_ids: frozenset[str],
    supplied: Mapping[str, str] | None,
) -> Mapping[str, str]:
    if supplied is None:
        return MappingProxyType({})
    if not isinstance(supplied, Mapping):
        raise DatasetValidationError(
            "reference_answers_by_query must be a mapping keyed by query ID"
        )
    unknown = sorted(set(supplied) - query_ids)
    if unknown:
        raise DatasetValidationError(
            f"reference answers contain unknown query IDs: {unknown}"
        )
    normalized: dict[str, str] = {}
    for query_id, answer in supplied.items():
        if not isinstance(answer, str) or not answer.strip():
            raise DatasetValidationError(
                f"reference answer for query {query_id!r} must be non-empty"
            )
        normalized[query_id] = unicodedata.normalize("NFC", answer)
    return MappingProxyType(normalized)


def _validate_corpus_records(
    records: Sequence[_RecordT] | None,
    *,
    record_type: type[_RecordT],
    record_name: str,
) -> frozenset[str]:
    if records is None:
        raise CorpusValidationError(
            f"{record_name}s are required for {record_name}-level validation"
        )
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise CorpusValidationError(
            f"{record_name}s must be a sequence of {record_type.__name__} values"
        )
    if not records:
        raise CorpusValidationError(f"{record_name}s must not be empty")
    identifiers: set[str] = set()
    for position, record in enumerate(records):
        if not isinstance(record, record_type):
            raise CorpusValidationError(
                f"{record_name}s[{position}] must be a {record_type.__name__}"
            )
        if record.id in identifiers:
            raise CorpusValidationError(
                f"duplicate {record_name} ID {record.id!r} at position {position}"
            )
        identifiers.add(record.id)
    return frozenset(identifiers)


def _coerce_path(path: str | Path) -> Path:
    try:
        source = Path(path)
    except TypeError as exc:
        raise DatasetValidationError("dataset path must be a string or Path") from exc
    try:
        is_file = source.is_file()
    except OSError as exc:
        raise DatasetValidationError(
            f"cannot inspect dataset path {source}: {exc}"
        ) from exc
    if not is_file:
        raise DatasetValidationError(f"dataset path is not a file: {source}")
    return source


def _read_utf8(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DatasetValidationError(f"cannot read dataset {path}: {exc}") from exc
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        line_number = data[: exc.start].count(b"\n") + 1
        raise _dataset_error(
            path,
            line_number,
            "dataset must be valid UTF-8",
        ) from exc


def _parse_json_object(
    line: str,
    *,
    path: Path,
    line_number: int,
) -> dict[str, object]:
    try:
        value = cast(
            object,
            json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_non_json_constant,
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise _dataset_error(path, line_number, f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _dataset_error(path, line_number, "each record must be a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _parse_dataset_record(
    record: Mapping[str, object],
    *,
    relevance_level: RelevanceLevel,
    path: Path,
    line_number: int,
) -> tuple[EvaluationQuery, Mapping[str, int], str | None]:
    required = {"query_id", "query", "relevant"}
    allowed = required | {"reference_answer", "metadata"}
    missing = sorted(required - set(record))
    unknown = sorted(set(record) - allowed)
    if missing or unknown:
        raise _dataset_error(
            path,
            line_number,
            f"record fields are invalid; missing={missing}, unknown={unknown}",
        )

    query_id = _record_string(record["query_id"], "query_id", path, line_number)
    query_text = _record_string(record["query"], "query", path, line_number)
    relevant = record["relevant"]
    if not isinstance(relevant, list) or not relevant:
        raise _dataset_error(
            path,
            line_number,
            "relevant must be a non-empty array",
        )

    grades: dict[str, int] = {}
    for position, item in enumerate(relevant):
        if not isinstance(item, dict):
            raise _dataset_error(
                path,
                line_number,
                f"relevant[{position}] must be an object",
            )
        expected_fields = {"id", "relevance"}
        missing_fields = sorted(expected_fields - set(item))
        unknown_fields = sorted(set(item) - expected_fields)
        if missing_fields or unknown_fields:
            raise _dataset_error(
                path,
                line_number,
                f"relevant[{position}] fields are invalid; "
                f"missing={missing_fields}, unknown={unknown_fields}",
            )
        identifier = _record_string(
            item["id"],
            f"relevant[{position}].id",
            path,
            line_number,
        )
        grade = item["relevance"]
        if isinstance(grade, bool) or not isinstance(grade, int) or grade < 1:
            raise _dataset_error(
                path,
                line_number,
                f"relevant[{position}].relevance must be an integer >= 1 "
                "(booleans are not valid integers)",
            )
        if identifier in grades:
            raise _dataset_error(
                path,
                line_number,
                f"duplicate relevant ID {identifier!r}",
            )
        grades[identifier] = grade

    metadata_value = record.get("metadata", {})
    try:
        metadata = normalize_json_mapping(
            metadata_value,
            field_name=f"query {query_id!r} metadata",
            error_type=DatasetValidationError,
        )
    except DatasetValidationError as exc:
        raise _dataset_error(path, line_number, str(exc)) from exc

    reference_answer: str | None = None
    if "reference_answer" in record:
        reference_answer = _record_string(
            record["reference_answer"],
            "reference_answer",
            path,
            line_number,
        )

    try:
        query = EvaluationQuery(
            id=query_id,
            query=query_text,
            relevant_document_ids=(
                frozenset(grades) if relevance_level == "document" else frozenset()
            ),
            relevant_chunk_ids=(
                frozenset(grades) if relevance_level == "chunk" else frozenset()
            ),
            metadata=metadata,
        )
    except DatasetValidationError as exc:
        raise _dataset_error(path, line_number, str(exc)) from exc
    return query, MappingProxyType(grades), reference_answer


def _record_string(
    value: object,
    field_name: str,
    path: Path,
    line_number: int,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _dataset_error(
            path,
            line_number,
            f"{field_name} must be a non-empty string",
        )
    return unicodedata.normalize("NFC", value)


def _dataset_error(
    path: Path,
    line_number: int,
    message: str,
) -> DatasetValidationError:
    return DatasetValidationError(f"{path}:{line_number}: {message}")


__all__ = ["EvaluationDataset", "RelevanceLevel", "validate_dataset"]
