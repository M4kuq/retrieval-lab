"""Application services for persisted evaluation-dataset authoring workflows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from retrieval_lab.dataset_authoring import (
    DatasetDraft,
    DatasetOrigin,
    DatasetProvenance,
    DatasetReviewStatus,
    DraftQuery,
)
from retrieval_lab.datasets import RelevanceLevel
from retrieval_lab.domain._validation import normalize_json_mapping
from retrieval_lab.domain.json_types import JSONValue
from retrieval_lab.exceptions import DatasetValidationError
from retrieval_lab.loaders import load_documents
from retrieval_lab.synthetic import mark_draft_reviewed

_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_DRAFT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class DatasetDraftStatus:
    """Safe, deterministic status for a persisted dataset draft."""

    query_count: int
    complete_query_count: int
    pending_query_ids: tuple[str, ...]
    relevance_level: RelevanceLevel
    origin: str
    review_status: str
    reliability: str

    @classmethod
    def from_draft(cls, draft: DatasetDraft) -> DatasetDraftStatus:
        """Build status from validated in-memory authoring state."""

        pending = draft.pending_query_ids
        return cls(
            query_count=len(draft.queries),
            complete_query_count=len(draft.queries) - len(pending),
            pending_query_ids=pending,
            relevance_level=draft.relevance_level,
            origin=draft.provenance.origin,
            review_status=draft.provenance.review_status,
            reliability=draft.provenance.reliability,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return deterministic JSON-compatible status data."""

        return {
            "complete_query_count": self.complete_query_count,
            "origin": self.origin,
            "pending_query_ids": list(self.pending_query_ids),
            "query_count": self.query_count,
            "relevance_level": self.relevance_level,
            "reliability": self.reliability,
            "review_status": self.review_status,
        }

    def to_json(self) -> str:
        """Return strict deterministic JSON with a trailing newline."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )


def load_dataset_draft(bundle: str | Path) -> DatasetDraft:
    """Load and validate a bundle produced by ``DatasetDraft.save_bundle``."""

    root = _bundle_directory(bundle)
    manifest_path = root / "dataset-manifest.json"
    dataset_path = root / "evaluation.jsonl"
    manifest = _read_json_object(
        manifest_path,
        max_bytes=_MAX_MANIFEST_BYTES,
        label="dataset manifest",
    )
    provenance, relevance_level, query_count, expected_pending = _parse_manifest(
        manifest
    )
    queries = _read_draft_queries(dataset_path)
    draft = DatasetDraft(
        queries,
        relevance_level=relevance_level,
        provenance=provenance,
    )
    if len(draft.queries) != query_count:
        raise DatasetValidationError(
            "dataset manifest query_count does not match evaluation.jsonl"
        )
    if draft.pending_query_ids != expected_pending:
        raise DatasetValidationError(
            "dataset manifest pending_query_ids do not match evaluation.jsonl"
        )
    return draft


def dataset_draft_status(bundle: str | Path) -> DatasetDraftStatus:
    """Load a bundle and return authoring/review status."""

    return DatasetDraftStatus.from_draft(load_dataset_draft(bundle))


def review_dataset_query(
    bundle: str | Path,
    *,
    query_id: str,
    relevance: Mapping[str, int],
    corpus: str | Path,
    complete_review: bool = False,
    notes: str | None = None,
) -> DatasetDraftStatus:
    """Replace one query's judgments, validate them against the corpus, and save."""

    draft = load_dataset_draft(bundle)
    if draft.relevance_level != "document":
        raise DatasetValidationError(
            "interactive review currently supports document-level drafts only"
        )
    documents = load_documents(corpus)
    available_ids = frozenset(document.id for document in documents)
    normalized_relevance = _validate_relevance_mapping(relevance)
    unknown = sorted(set(normalized_relevance) - available_ids)
    if unknown:
        raise DatasetValidationError(
            f"review relevance references unknown documents: {unknown}"
        )

    target = next((query for query in draft.queries if query.id == query_id), None)
    if target is None:
        raise DatasetValidationError(f"unknown draft query ID {query_id!r}")
    updated_query = DraftQuery(
        id=target.id,
        query=target.query,
        relevance=normalized_relevance,
        reference_answer=target.reference_answer,
        metadata=target.metadata,
    )
    updated = draft.replace_query(updated_query)
    if complete_review:
        if not updated.complete:
            raise DatasetValidationError(
                "cannot complete review while relevance judgments are pending"
            )
        updated = mark_draft_reviewed(updated, notes=notes)
    elif notes is not None:
        updated = updated.with_provenance(
            DatasetProvenance(
                origin=updated.provenance.origin,
                review_status=updated.provenance.review_status,
                generator=updated.provenance.generator,
                notes=notes,
            )
        )
    updated.save_bundle(_bundle_directory(bundle))
    return DatasetDraftStatus.from_draft(updated)


def finalize_dataset_bundle(bundle: str | Path) -> DatasetDraftStatus:
    """Validate that a persisted draft is ready for EvaluationDataset loading."""

    draft = load_dataset_draft(bundle)
    draft.finalize()
    return DatasetDraftStatus.from_draft(draft)


def _bundle_directory(value: str | Path) -> Path:
    try:
        path = Path(value)
    except TypeError as exc:
        raise DatasetValidationError(
            "dataset bundle path must be a string or Path"
        ) from exc
    try:
        if not path.is_dir():
            raise DatasetValidationError(
                f"dataset bundle path is not a directory: {path}"
            )
    except DatasetValidationError:
        raise
    except OSError as exc:
        raise DatasetValidationError(
            f"cannot inspect dataset bundle path {path}: {exc}"
        ) from exc
    return path


def _read_draft_queries(path: Path) -> tuple[DraftQuery, ...]:
    text = _read_text(path, max_bytes=_MAX_DRAFT_BYTES, label="dataset draft")
    if not text:
        return ()
    queries: list[DraftQuery] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise _line_error(path, line_number, "blank lines are not valid JSONL")
        record = _loads_object(line, path=path, line_number=line_number)
        query = _parse_draft_record(record, path=path, line_number=line_number)
        if query.id in seen:
            raise _line_error(path, line_number, f"duplicate query_id {query.id!r}")
        seen.add(query.id)
        queries.append(query)
    return tuple(queries)


def _parse_draft_record(
    record: Mapping[str, object],
    *,
    path: Path,
    line_number: int,
) -> DraftQuery:
    required = {"query_id", "query", "relevant"}
    allowed = required | {"reference_answer", "metadata"}
    missing = sorted(required - set(record))
    unknown = sorted(set(record) - allowed)
    if missing or unknown:
        raise _line_error(
            path,
            line_number,
            f"record fields are invalid; missing={missing}, unknown={unknown}",
        )
    query_id = _string(record["query_id"], "query_id", path, line_number)
    query_text = _string(record["query"], "query", path, line_number)
    relevant = record["relevant"]
    if not isinstance(relevant, list):
        raise _line_error(path, line_number, "relevant must be an array")
    relevance: dict[str, int] = {}
    for index, item in enumerate(relevant):
        if not isinstance(item, dict) or set(item) != {"id", "relevance"}:
            raise _line_error(
                path,
                line_number,
                f"relevant[{index}] must contain only id and relevance",
            )
        identifier = _string(
            item["id"],
            f"relevant[{index}].id",
            path,
            line_number,
        )
        grade = item["relevance"]
        if isinstance(grade, bool) or not isinstance(grade, int) or grade < 1:
            raise _line_error(
                path,
                line_number,
                f"relevant[{index}].relevance must be an integer >= 1",
            )
        if identifier in relevance:
            raise _line_error(
                path,
                line_number,
                f"duplicate relevant ID {identifier!r}",
            )
        relevance[identifier] = grade
    reference_answer = None
    if "reference_answer" in record:
        reference_answer = _string(
            record["reference_answer"],
            "reference_answer",
            path,
            line_number,
        )
    try:
        metadata = normalize_json_mapping(
            record.get("metadata", {}),
            field_name=f"DraftQuery[{query_id!r}].metadata",
            error_type=DatasetValidationError,
        )
        return DraftQuery(
            query_id,
            query_text,
            relevance=relevance,
            reference_answer=reference_answer,
            metadata=metadata,
        )
    except DatasetValidationError as exc:
        raise _line_error(path, line_number, str(exc)) from exc


def _parse_manifest(
    manifest: Mapping[str, object],
) -> tuple[DatasetProvenance, RelevanceLevel, int, tuple[str, ...]]:
    expected = {
        "schema_version",
        "relevance_level",
        "provenance",
        "query_count",
        "pending_query_ids",
    }
    if set(manifest) != expected:
        raise DatasetValidationError(
            "dataset manifest fields are invalid; expected schema_version, "
            "relevance_level, provenance, query_count, and pending_query_ids"
        )
    if manifest["schema_version"] != 1:
        raise DatasetValidationError("unsupported dataset manifest schema_version")
    relevance_level = manifest["relevance_level"]
    if relevance_level not in {"document", "chunk"}:
        raise DatasetValidationError(
            "dataset manifest relevance_level must be document or chunk"
        )
    query_count = manifest["query_count"]
    if (
        isinstance(query_count, bool)
        or not isinstance(query_count, int)
        or query_count < 0
    ):
        raise DatasetValidationError(
            "dataset manifest query_count must be an integer >= 0"
        )
    pending = manifest["pending_query_ids"]
    if not isinstance(pending, list) or any(
        not isinstance(item, str) or not item.strip() for item in pending
    ):
        raise DatasetValidationError(
            "dataset manifest pending_query_ids must be an array of non-empty strings"
        )
    if len(set(pending)) != len(pending):
        raise DatasetValidationError(
            "dataset manifest pending_query_ids must not contain duplicates"
        )
    provenance_value = manifest["provenance"]
    if not isinstance(provenance_value, dict):
        raise DatasetValidationError("dataset manifest provenance must be an object")
    provenance = _parse_provenance(provenance_value)
    return (
        provenance,
        cast(RelevanceLevel, relevance_level),
        query_count,
        tuple(pending),
    )


def _parse_provenance(value: Mapping[str, object]) -> DatasetProvenance:
    expected = {
        "origin",
        "review_status",
        "human_reviewed",
        "reliability",
        "generator",
        "notes",
    }
    if set(value) != expected:
        raise DatasetValidationError("dataset manifest provenance fields are invalid")
    origin = value["origin"]
    review_status = value["review_status"]
    if not isinstance(origin, str) or not isinstance(review_status, str):
        raise DatasetValidationError(
            "dataset manifest provenance origin and review_status must be strings"
        )
    generator = value["generator"]
    notes = value["notes"]
    if generator is not None and not isinstance(generator, str):
        raise DatasetValidationError(
            "dataset manifest provenance generator must be a string or null"
        )
    if notes is not None and not isinstance(notes, str):
        raise DatasetValidationError(
            "dataset manifest provenance notes must be a string or null"
        )
    provenance = DatasetProvenance(
        origin=cast(DatasetOrigin, origin),
        review_status=cast(DatasetReviewStatus, review_status),
        generator=generator,
        notes=notes,
    )
    if value["human_reviewed"] is not provenance.human_reviewed:
        raise DatasetValidationError(
            "dataset manifest human_reviewed conflicts with review_status"
        )
    if value["reliability"] != provenance.reliability:
        raise DatasetValidationError(
            "dataset manifest reliability conflicts with provenance"
        )
    return provenance


def _validate_relevance_mapping(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise DatasetValidationError("review relevance must be a non-empty mapping")
    normalized: dict[str, int] = {}
    for identifier, grade in value.items():
        if not isinstance(identifier, str) or not identifier.strip():
            raise DatasetValidationError("review relevance IDs must be non-empty")
        if isinstance(grade, bool) or not isinstance(grade, int) or grade < 1:
            raise DatasetValidationError(
                "review relevance grades must be integers >= 1"
            )
        normalized[identifier] = grade
    return normalized


def _read_json_object(path: Path, *, max_bytes: int, label: str) -> dict[str, object]:
    text = _read_text(path, max_bytes=max_bytes, label=label)
    try:
        value = cast(
            object,
            json.loads(text, object_pairs_hook=_unique_object),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise DatasetValidationError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{label} must be a JSON object")
    return value


def _read_text(path: Path, *, max_bytes: int, label: str) -> str:
    try:
        if not path.is_file():
            raise DatasetValidationError(f"{label} is not a regular file: {path}")
        size = path.stat().st_size
        if size > max_bytes:
            raise DatasetValidationError(
                f"{label} exceeds the {max_bytes}-byte safety limit"
            )
        data = path.read_bytes()
    except DatasetValidationError:
        raise
    except OSError as exc:
        raise DatasetValidationError(f"cannot read {label}: {exc}") from exc
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(f"{label} must be valid UTF-8") from exc


def _loads_object(line: str, *, path: Path, line_number: int) -> dict[str, object]:
    try:
        value = cast(
            object,
            json.loads(line, object_pairs_hook=_unique_object),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise _line_error(path, line_number, f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _line_error(path, line_number, "record must be a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _string(value: object, field: str, path: Path, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _line_error(path, line_number, f"{field} must be a non-empty string")
    return value


def _line_error(path: Path, line_number: int, message: str) -> DatasetValidationError:
    return DatasetValidationError(f"{path}:{line_number}: {message}")


__all__ = [
    "DatasetDraftStatus",
    "dataset_draft_status",
    "finalize_dataset_bundle",
    "load_dataset_draft",
    "review_dataset_query",
]
