"""Loaders for standard retrieval benchmarks in their published layouts.

The loaders are offline: they read a benchmark that is already on
disk (downloading stays with the caller, like every other external
dependency in this library) and convert it into the native
:class:`~retrieval_lab.Document` and
:class:`~retrieval_lab.EvaluationDataset` records, so a standard
benchmark runs through the same evaluation pipeline as a local
corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from retrieval_lab.datasets import EvaluationDataset
from retrieval_lab.domain import Document, EvaluationQuery
from retrieval_lab.exceptions import DatasetValidationError

__all__ = [
    "BenchmarkDataset",
    "load_beir_dataset",
]


@dataclass(frozen=True)
class BenchmarkDataset:
    """A loaded benchmark: its corpus documents and evaluation dataset."""

    documents: tuple[Document, ...]
    dataset: EvaluationDataset
    excluded_query_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.documents:
            raise DatasetValidationError("benchmark documents must not be empty")
        for position, document in enumerate(self.documents):
            if not isinstance(document, Document):
                raise DatasetValidationError(
                    f"documents[{position}] must be a Document"
                )
        if not isinstance(self.dataset, EvaluationDataset):
            raise DatasetValidationError("dataset must be an EvaluationDataset")
        seen: set[str] = set()
        for query_id in self.excluded_query_ids:
            if not isinstance(query_id, str) or not query_id:
                raise DatasetValidationError(
                    "excluded query IDs must be non-empty strings"
                )
            if query_id in seen:
                raise DatasetValidationError("excluded query IDs must be unique")
            seen.add(query_id)


def load_beir_dataset(path: str | Path, *, split: str = "test") -> BenchmarkDataset:
    """Load a BEIR-format dataset directory from disk.

    The directory must contain ``corpus.jsonl`` and ``queries.jsonl``
    (records with ``_id`` and ``text``; corpus records may carry a
    ``title``, which is prepended to the document text) and
    ``qrels/<split>.tsv`` with ``query-id``, ``corpus-id``, and
    ``score`` columns. Positive scores become graded relevance
    judgments. Queries with no positive judgment in the split cannot
    be evaluated and are reported in ``excluded_query_ids`` instead of
    being dropped silently.
    """

    if not isinstance(split, str) or not split.strip():
        raise DatasetValidationError("split must be a non-empty string")
    if "/" in split or "\\" in split or split in {".", ".."}:
        raise DatasetValidationError("split must be a plain file name")
    root = Path(path)
    if not root.is_dir():
        raise DatasetValidationError(f"benchmark directory not found: {root}")
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels" / f"{split}.tsv"
    for required in (corpus_path, queries_path, qrels_path):
        if not required.is_file():
            raise DatasetValidationError(f"benchmark file not found: {required}")

    documents = _load_beir_corpus(corpus_path)
    queries = _load_beir_queries(queries_path)
    grades_by_query = _load_beir_qrels(
        qrels_path,
        query_ids=frozenset(queries),
        document_ids=frozenset(document.id for document in documents),
    )
    evaluated: list[EvaluationQuery] = []
    excluded: list[str] = []
    for query_id, query in queries.items():
        grades = grades_by_query.get(query_id, {})
        if not grades:
            excluded.append(query_id)
            continue
        evaluated.append(
            EvaluationQuery(
                query_id,
                query,
                relevant_document_ids=frozenset(grades),
            )
        )
    if not evaluated:
        raise DatasetValidationError(
            f"no queries in {queries_path} have positive judgments in {qrels_path}"
        )
    dataset = EvaluationDataset(
        evaluated,
        relevance_grades_by_query=grades_by_query,
    )
    return BenchmarkDataset(
        documents=documents,
        dataset=dataset,
        excluded_query_ids=tuple(excluded),
    )


def _load_beir_corpus(path: Path) -> tuple[Document, ...]:
    documents: list[Document] = []
    seen: dict[str, int] = {}
    for line_number, record in _jsonl_records(path):
        document_id = _record_id(record, path=path, line_number=line_number)
        if document_id in seen:
            raise _error(
                path,
                line_number,
                f"duplicate corpus _id {document_id!r}; first seen on line "
                f"{seen[document_id]}",
            )
        seen[document_id] = line_number
        title = record.get("title", "")
        if not isinstance(title, str):
            raise _error(path, line_number, "corpus title must be a string")
        text = record.get("text")
        if not isinstance(text, str):
            raise _error(path, line_number, "corpus text must be a string")
        combined = f"{title}\n{text}" if title.strip() else text
        if not combined.strip():
            raise _error(
                path,
                line_number,
                f"corpus document {document_id!r} has no title or text",
            )
        documents.append(Document(document_id, combined))
    if not documents:
        raise _error(path, 1, "corpus must not be empty")
    return tuple(documents)


def _load_beir_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    seen: dict[str, int] = {}
    for line_number, record in _jsonl_records(path):
        query_id = _record_id(record, path=path, line_number=line_number)
        if query_id in seen:
            raise _error(
                path,
                line_number,
                f"duplicate query _id {query_id!r}; first seen on line "
                f"{seen[query_id]}",
            )
        seen[query_id] = line_number
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise _error(path, line_number, "query text must be a non-empty string")
        queries[query_id] = text
    if not queries:
        raise _error(path, 1, "queries must not be empty")
    return queries


def _load_beir_qrels(
    path: Path, *, query_ids: frozenset[str], document_ids: frozenset[str]
) -> dict[str, dict[str, int]]:
    grades: dict[str, dict[str, int]] = {}
    for line_number, raw_line in _iter_lines(path):
        if not raw_line.strip():
            continue
        columns = raw_line.split("\t")
        stripped = tuple(column.strip() for column in columns)
        if line_number == 1 and stripped == ("query-id", "corpus-id", "score"):
            continue
        if len(columns) != 3:
            raise _error(
                path, line_number, "qrels rows must have exactly three TSV columns"
            )
        query_id, document_id, score_text = stripped
        if query_id not in query_ids:
            raise _error(
                path, line_number, f"qrels reference unknown query {query_id!r}"
            )
        if document_id not in document_ids:
            raise _error(
                path, line_number, f"qrels reference unknown corpus ID {document_id!r}"
            )
        try:
            score = int(score_text)
        except ValueError:
            raise _error(
                path, line_number, f"qrels score must be an integer, got {score_text!r}"
            ) from None
        if score < 0:
            raise _error(path, line_number, "qrels score must not be negative")
        if score > 0:
            grades.setdefault(query_id, {})[document_id] = score
        elif query_id in grades:
            # A later zero judgment overrides an earlier positive one.
            grades[query_id].pop(document_id, None)
    return {query_id: items for query_id, items in grades.items() if items}


def _jsonl_records(path: Path) -> list[tuple[int, dict[str, object]]]:
    records: list[tuple[int, dict[str, object]]] = []
    for line_number, line in _iter_lines(path):
        if not line.strip():
            raise _error(path, line_number, "blank lines are not valid JSONL records")
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            raise _error(path, line_number, f"invalid JSON: {error.msg}") from error
        if not isinstance(parsed, dict):
            raise _error(path, line_number, "JSONL records must be JSON objects")
        records.append((line_number, parsed))
    return records


def _iter_lines(path: Path) -> list[tuple[int, str]]:
    pieces = path.read_bytes().split(b"\n")
    lines: list[tuple[int, str]] = []
    for line_number, piece in enumerate(pieces, start=1):
        if line_number == len(pieces) and not piece:
            break
        try:
            text = piece.decode("utf-8")
        except UnicodeDecodeError:
            raise _error(path, line_number, "file is not valid UTF-8") from None
        lines.append((line_number, text.removesuffix("\r")))
    return lines


def _record_id(record: dict[str, object], *, path: Path, line_number: int) -> str:
    identifier = record.get("_id")
    if not isinstance(identifier, str) or not identifier.strip():
        raise _error(path, line_number, "record _id must be a non-empty string")
    return identifier


def _error(path: Path, line_number: int, message: str) -> DatasetValidationError:
    return DatasetValidationError(f"{path}:{line_number}: {message}")
