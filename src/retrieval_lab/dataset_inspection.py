"""Dataset quality diagnostics for v0.2 authoring workflows."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from retrieval_lab.datasets import EvaluationDataset
from retrieval_lab.domain import Chunk, Document
from retrieval_lab.exceptions import DatasetValidationError

DatasetInspectionCode = Literal[
    "duplicate_query_text",
    "relevance_concentration",
    "verbatim_query_in_relevant_text",
]


@dataclass(frozen=True)
class DatasetInspectionIssue:
    """One deterministic dataset-quality finding that requires human review."""

    code: DatasetInspectionCode
    message: str
    query_ids: tuple[str, ...] = ()
    relevance_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetInspectionReport:
    """Summary of non-destructive quality diagnostics for one dataset."""

    query_count: int
    unique_relevance_id_count: int
    max_relevance_share: float
    issues: tuple[DatasetInspectionIssue, ...]

    @property
    def has_issues(self) -> bool:
        """Return whether any diagnostic requires review."""
        return bool(self.issues)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation with deterministic ordering."""
        return {
            "query_count": self.query_count,
            "unique_relevance_id_count": self.unique_relevance_id_count,
            "max_relevance_share": self.max_relevance_share,
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "query_ids": list(issue.query_ids),
                    "relevance_ids": list(issue.relevance_ids),
                }
                for issue in self.issues
            ],
        }


def inspect_dataset(
    dataset: EvaluationDataset,
    *,
    documents: Sequence[Document] | None = None,
    chunks: Sequence[Chunk] | None = None,
    relevance_concentration_threshold: float = 0.5,
    min_queries_for_concentration: int = 5,
    min_verbatim_query_characters: int = 12,
) -> DatasetInspectionReport:
    """Return conservative, deterministic candidates for human dataset review."""
    if not isinstance(dataset, EvaluationDataset):
        raise DatasetValidationError("dataset must be an EvaluationDataset")
    threshold = _validate_probability(relevance_concentration_threshold)
    min_queries = _validate_integer(min_queries_for_concentration, minimum=2)
    min_chars = _validate_integer(min_verbatim_query_characters, minimum=1)

    issues = list(_duplicate_query_issues(dataset))
    counts: Counter[str] = Counter()
    queries_by_id: dict[str, list[str]] = defaultdict(list)
    for query in dataset.queries:
        for relevance_id in sorted(dataset.relevance_grades_by_query[query.id]):
            counts[relevance_id] += 1
            queries_by_id[relevance_id].append(query.id)

    query_count = len(dataset.queries)
    max_share = max(
        (count / query_count for count in counts.values()),
        default=0.0,
    )
    if query_count >= min_queries:
        ranked_counts = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        concentrated = tuple(
            relevance_id
            for relevance_id, count in ranked_counts
            if count / query_count > threshold
        )
        if concentrated:
            query_ids = tuple(
                sorted(
                    {
                        query_id
                        for relevance_id in concentrated
                        for query_id in queries_by_id[relevance_id]
                    }
                )
            )
            issues.append(
                DatasetInspectionIssue(
                    code="relevance_concentration",
                    message=(
                        "positive relevance IDs exceed the configured query-share "
                        "threshold; review whether judgments are over-concentrated."
                    ),
                    query_ids=query_ids,
                    relevance_ids=concentrated,
                )
            )

    corpus_text = _corpus_text(dataset, documents=documents, chunks=chunks)
    if corpus_text is not None:
        issues.extend(_verbatim_issues(dataset, corpus_text, min_chars=min_chars))

    return DatasetInspectionReport(
        query_count=query_count,
        unique_relevance_id_count=len(counts),
        max_relevance_share=max_share,
        issues=tuple(issues),
    )


def _duplicate_query_issues(
    dataset: EvaluationDataset,
) -> tuple[DatasetInspectionIssue, ...]:
    by_text: dict[str, list[str]] = defaultdict(list)
    for query in dataset.queries:
        by_text[_normalize(query.query)].append(query.id)
    groups = sorted(tuple(sorted(ids)) for ids in by_text.values() if len(ids) > 1)
    return tuple(
        DatasetInspectionIssue(
            code="duplicate_query_text",
            message=(
                "queries normalize to identical text; review whether they are "
                "duplicates."
            ),
            query_ids=group,
        )
        for group in groups
    )


def _corpus_text(
    dataset: EvaluationDataset,
    *,
    documents: Sequence[Document] | None,
    chunks: Sequence[Chunk] | None,
) -> dict[str, str] | None:
    if dataset.relevance_level == "document":
        if documents is None:
            return None
        dataset.validate(documents=documents)
        return {item.id: item.text for item in documents}
    if chunks is None:
        return None
    dataset.validate(chunks=chunks)
    return {item.id: item.text for item in chunks}


def _verbatim_issues(
    dataset: EvaluationDataset,
    corpus_text: dict[str, str],
    *,
    min_chars: int,
) -> tuple[DatasetInspectionIssue, ...]:
    normalized_corpus = {
        key: _normalize(value) for key, value in corpus_text.items()
    }
    findings: list[DatasetInspectionIssue] = []
    for query in dataset.queries:
        normalized_query = _normalize(query.query)
        if len(normalized_query) < min_chars:
            continue
        matches = tuple(
            relevance_id
            for relevance_id in sorted(dataset.relevance_grades_by_query[query.id])
            if normalized_query in normalized_corpus[relevance_id]
        )
        if matches:
            findings.append(
                DatasetInspectionIssue(
                    code="verbatim_query_in_relevant_text",
                    message=(
                        "the normalized query appears verbatim in positive corpus "
                        "text; review it as a possible low-difficulty or "
                        "generation-leakage candidate."
                    ),
                    query_ids=(query.id,),
                    relevance_ids=matches,
                )
            )
    return tuple(findings)


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _validate_probability(value: object) -> float:
    message = "relevance_concentration_threshold must be a finite number in (0, 1]"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetValidationError(message)
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0 or normalized > 1.0:
        raise DatasetValidationError(message)
    return normalized


def _validate_integer(value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DatasetValidationError(
            f"inspection limit must be an integer >= {minimum}"
        )
    return value


__all__ = [
    "DatasetInspectionCode",
    "DatasetInspectionIssue",
    "DatasetInspectionReport",
    "inspect_dataset",
]
