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
    "load_msmarco_dataset",
    "load_natural_questions_dataset",
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

    _validate_split(split)
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
    return _finish_benchmark(
        documents,
        queries,
        grades_by_query,
        no_positive_message=(
            f"no queries in {queries_path} have positive judgments in {qrels_path}"
        ),
    )


def load_msmarco_dataset(path: str | Path, *, split: str = "train") -> BenchmarkDataset:
    """Load an MS MARCO passage-ranking dataset directory from disk.

    The directory must contain ``collection.tsv`` (``pid`` and passage
    text per line), ``queries.<split>.tsv`` (``qid`` and query text),
    and ``qrels.<split>.tsv`` in TREC format (``qid``, an ignored
    iteration column, ``pid``, and an integer relevance). Positive
    relevance values become graded judgments. Queries with no positive
    judgment cannot be evaluated and are reported in
    ``excluded_query_ids`` instead of being dropped silently.
    """

    _validate_split(split)
    root = Path(path)
    if not root.is_dir():
        raise DatasetValidationError(f"benchmark directory not found: {root}")
    collection_path = root / "collection.tsv"
    queries_path = root / f"queries.{split}.tsv"
    qrels_path = root / f"qrels.{split}.tsv"
    for required in (collection_path, queries_path, qrels_path):
        if not required.is_file():
            raise DatasetValidationError(f"benchmark file not found: {required}")

    documents = tuple(
        Document(document_id, text)
        for document_id, text in _load_tsv_pairs(collection_path, kind="collection")
    )
    queries = dict(_load_tsv_pairs(queries_path, kind="queries"))
    grades_by_query = _load_msmarco_qrels(
        qrels_path,
        query_ids=frozenset(queries),
        document_ids=frozenset(document.id for document in documents),
    )
    return _finish_benchmark(
        documents,
        queries,
        grades_by_query,
        no_positive_message=(
            f"no queries in {queries_path} have positive judgments in {qrels_path}"
        ),
    )


def load_natural_questions_dataset(path: str | Path) -> BenchmarkDataset:
    """Load a simplified Natural Questions JSONL file from disk.

    Each record is one example of the official simplified format: an
    ``example_id``, a ``question_text``, the page's ``document_text``
    (space-separated tokens), and ``annotations`` whose ``long_answer``
    spans mark the answer-bearing page. Every example's page becomes a
    corpus document keyed by the string form of its ``example_id``;
    an example is a relevant judgment (grade 1) for its own question
    when at least one annotation carries a long answer
    (``candidate_index`` >= 0 with a valid token span). Examples with
    no long answer cannot be evaluated and are reported in
    ``excluded_query_ids`` instead of being dropped silently.
    """

    source = Path(path)
    if not source.is_file():
        raise DatasetValidationError(f"benchmark file not found: {source}")
    documents: list[Document] = []
    queries: dict[str, str] = {}
    grades_by_query: dict[str, dict[str, int]] = {}
    seen: dict[str, int] = {}
    for line_number, record in _jsonl_records(source):
        example_id = _nq_example_id(record, path=source, line_number=line_number)
        if example_id in seen:
            raise _error(
                source,
                line_number,
                f"duplicate example_id {example_id!r}; first seen on line "
                f"{seen[example_id]}",
            )
        seen[example_id] = line_number
        question = record.get("question_text")
        if not isinstance(question, str) or not question.strip():
            raise _error(
                source, line_number, "question_text must be a non-empty string"
            )
        document_text = record.get("document_text")
        if not isinstance(document_text, str) or not document_text.strip():
            raise _error(
                source, line_number, "document_text must be a non-empty string"
            )
        documents.append(Document(example_id, document_text))
        queries[example_id] = question
        if _nq_has_long_answer(
            record, path=source, line_number=line_number, document_text=document_text
        ):
            grades_by_query[example_id] = {example_id: 1}
    if not documents:
        raise _error(source, 1, "examples must not be empty")
    return _finish_benchmark(
        tuple(documents),
        queries,
        grades_by_query,
        no_positive_message=f"no examples in {source} have a long answer",
    )


def _validate_split(split: str) -> None:
    if not isinstance(split, str) or not split.strip():
        raise DatasetValidationError("split must be a non-empty string")
    if "/" in split or "\\" in split or split in {".", ".."}:
        raise DatasetValidationError("split must be a plain file name")


def _finish_benchmark(
    documents: tuple[Document, ...],
    queries: dict[str, str],
    grades_by_query: dict[str, dict[str, int]],
    *,
    no_positive_message: str,
) -> BenchmarkDataset:
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
        raise DatasetValidationError(no_positive_message)
    dataset = EvaluationDataset(
        evaluated,
        relevance_grades_by_query=grades_by_query,
    )
    return BenchmarkDataset(
        documents=documents,
        dataset=dataset,
        excluded_query_ids=tuple(excluded),
    )


def _load_tsv_pairs(path: Path, *, kind: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    for line_number, raw_line in _iter_lines(path):
        if not raw_line.strip():
            continue
        columns = raw_line.split("\t", 1)
        if len(columns) != 2:
            raise _error(
                path, line_number, f"{kind} rows must have exactly two TSV columns"
            )
        identifier, text = columns[0].strip(), columns[1]
        if not identifier:
            raise _error(path, line_number, f"{kind} ID must be a non-empty string")
        if identifier in seen:
            raise _error(
                path,
                line_number,
                f"duplicate {kind} ID {identifier!r}; first seen on line "
                f"{seen[identifier]}",
            )
        seen[identifier] = line_number
        if not text.strip():
            raise _error(path, line_number, f"{kind} text must be a non-empty string")
        pairs.append((identifier, text))
    if not pairs:
        raise _error(path, 1, f"{kind} must not be empty")
    return pairs


def _load_msmarco_qrels(
    path: Path, *, query_ids: frozenset[str], document_ids: frozenset[str]
) -> dict[str, dict[str, int]]:
    grades: dict[str, dict[str, int]] = {}
    for line_number, raw_line in _iter_lines(path):
        if not raw_line.strip():
            continue
        columns = raw_line.split("\t")
        if len(columns) != 4:
            raise _error(
                path, line_number, "qrels rows must have exactly four TSV columns"
            )
        query_id, _, document_id, score_text = (column.strip() for column in columns)
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
            grades[query_id].pop(document_id, None)
    return {query_id: items for query_id, items in grades.items() if items}


def _nq_example_id(record: dict[str, object], *, path: Path, line_number: int) -> str:
    identifier = record.get("example_id")
    if isinstance(identifier, bool):
        raise _error(
            path, line_number, "example_id must be a non-empty string or integer"
        )
    if isinstance(identifier, int):
        return str(identifier)
    if isinstance(identifier, str) and identifier.strip():
        return identifier
    raise _error(path, line_number, "example_id must be a non-empty string or integer")


def _nq_has_long_answer(
    record: dict[str, object],
    *,
    path: Path,
    line_number: int,
    document_text: str,
) -> bool:
    annotations = record.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise _error(path, line_number, "annotations must be a non-empty list")
    token_count = len(document_text.split(" "))
    found = False
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise _error(path, line_number, "annotations must contain objects")
        long_answer = annotation.get("long_answer")
        if not isinstance(long_answer, dict):
            raise _error(
                path, line_number, "annotations must contain a long_answer object"
            )
        candidate_index = long_answer.get("candidate_index")
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise _error(
                path, line_number, "long_answer candidate_index must be an integer"
            )
        start = long_answer.get("start_token")
        end = long_answer.get("end_token")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise _error(
                path,
                line_number,
                "long_answer token span must use integer start_token and end_token",
            )
        if candidate_index < 0:
            if candidate_index == start == end == -1:
                continue
            raise _error(
                path,
                line_number,
                "long_answer without an answer must use -1 for candidate_index, "
                "start_token and end_token",
            )
        if not 0 <= start < end <= token_count:
            raise _error(
                path,
                line_number,
                "long_answer token span is outside the document tokens",
            )
        found = True
    return found


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
