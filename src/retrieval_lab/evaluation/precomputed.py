"""Evaluate pre-ranked document identifiers without running retrieval."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from retrieval_lab.datasets import EvaluationDataset
from retrieval_lab.domain import (
    EvaluationQuery,
    EvaluationResult,
    JSONValue,
    QueryEvaluation,
)
from retrieval_lab.evaluation.engine import (
    aggregate_metrics,
    content_hash,
    dataset_payload,
    evaluate_ranking,
    normalize_top_k,
)
from retrieval_lab.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    RetrieverContractError,
)


@dataclass(frozen=True, init=False)
class RetrievedQueryResult:
    """An immutable document ranking produced by an existing search system.

    An empty ranking is valid and receives zero for every quality metric. Document
    identifiers must be unique, non-empty strings in best-to-worst rank order.
    """

    query_id: str
    retrieved_document_ids: tuple[str, ...]

    def __init__(
        self,
        query_id: str,
        retrieved_document_ids: Sequence[str],
    ) -> None:
        """Validate and defensively copy a precomputed ranking."""

        if not isinstance(query_id, str) or not query_id.strip():
            raise DatasetValidationError(
                "RetrievedQueryResult.query_id must be a non-empty string"
            )
        if isinstance(retrieved_document_ids, (str, bytes)) or not isinstance(
            retrieved_document_ids, Sequence
        ):
            raise RetrieverContractError(
                "RetrievedQueryResult.retrieved_document_ids must be a sequence "
                "of IDs, not a string"
            )

        normalized = tuple(retrieved_document_ids)
        if any(
            not isinstance(identifier, str) or not identifier.strip()
            for identifier in normalized
        ):
            raise RetrieverContractError(
                "RetrievedQueryResult.retrieved_document_ids must contain only "
                "non-empty strings"
            )
        if len(set(normalized)) != len(normalized):
            raise RetrieverContractError(
                "RetrievedQueryResult.retrieved_document_ids must be unique"
            )

        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "retrieved_document_ids", normalized)


def evaluate_results(
    *,
    dataset: EvaluationDataset,
    retrieved_results: Sequence[RetrievedQueryResult],
    top_k: Sequence[int] = (1, 3, 5, 10),
    name: str = "precomputed",
) -> EvaluationResult:
    """Evaluate existing document rankings using Retrieval Lab's core metrics.

    The result set must contain exactly one ranking for every dataset query. This
    entry point performs no indexing, retrieval, I/O, or network access.
    """

    normalized_name = _validate_name(name)
    normalized_top_k = normalize_top_k(top_k)
    queries, grades_by_query = _validate_dataset(dataset)
    rankings_by_query = _validate_results(retrieved_results, queries=queries)

    evaluations: list[QueryEvaluation] = []
    for query in queries:
        retrieved_ids = rankings_by_query[query.id]
        evaluations.append(
            evaluate_ranking(
                query_id=query.id,
                retrieved_ids=retrieved_ids,
                relevance_grades=grades_by_query[query.id],
                top_k=normalized_top_k,
            )
        )

    query_evaluations = tuple(evaluations)
    aggregate = aggregate_metrics(query_evaluations, normalized_top_k)
    manifest, run_id = _build_manifest(
        name=normalized_name,
        queries=queries,
        grades_by_query=grades_by_query,
        rankings_by_query=rankings_by_query,
        top_k=normalized_top_k,
    )
    return EvaluationResult(
        run_id=run_id,
        metrics={normalized_name: aggregate},
        query_results={normalized_name: query_evaluations},
        manifest=manifest,
    )


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("name must be a non-empty string")
    return name


def _validate_dataset(
    dataset: EvaluationDataset,
) -> tuple[tuple[EvaluationQuery, ...], dict[str, dict[str, int]]]:
    if not isinstance(dataset, EvaluationDataset):
        raise DatasetValidationError("dataset must be an EvaluationDataset")
    if dataset.relevance_level != "document":
        raise DatasetValidationError(
            "evaluate_results currently supports only document relevance"
        )
    return dataset.queries, {
        query.id: dict(dataset.relevance_grades_by_query[query.id])
        for query in dataset.queries
    }


def _validate_results(
    retrieved_results: Sequence[RetrievedQueryResult],
    *,
    queries: Sequence[EvaluationQuery],
) -> dict[str, tuple[str, ...]]:
    if isinstance(retrieved_results, (str, bytes)) or not isinstance(
        retrieved_results, Sequence
    ):
        raise DatasetValidationError(
            "retrieved_results must be a sequence of RetrievedQueryResult values"
        )
    normalized = tuple(retrieved_results)
    if not all(isinstance(result, RetrievedQueryResult) for result in normalized):
        raise DatasetValidationError(
            "retrieved_results must contain only RetrievedQueryResult values"
        )

    result_ids = [result.query_id for result in normalized]
    duplicate_ids = sorted(
        identifier for identifier, count in Counter(result_ids).items() if count > 1
    )
    if duplicate_ids:
        raise DatasetValidationError(
            f"retrieved_results contain duplicate query IDs: {duplicate_ids!r}"
        )

    expected_ids = {query.id for query in queries}
    actual_ids = set(result_ids)
    missing_ids = sorted(expected_ids - actual_ids)
    unknown_ids = sorted(actual_ids - expected_ids)
    if missing_ids or unknown_ids:
        raise DatasetValidationError(
            "retrieved result query IDs must exactly match dataset queries; "
            f"missing={missing_ids!r}, unknown={unknown_ids!r}"
        )
    return {result.query_id: result.retrieved_document_ids for result in normalized}


def _build_manifest(
    *,
    name: str,
    queries: Sequence[EvaluationQuery],
    grades_by_query: Mapping[str, Mapping[str, int]],
    rankings_by_query: Mapping[str, Sequence[str]],
    top_k: Sequence[int],
) -> tuple[dict[str, JSONValue], str]:
    rankings_payload: list[JSONValue] = []
    for query in queries:
        rankings_payload.append(
            {
                "query_id": query.id,
                "retrieved_document_ids": list(rankings_by_query[query.id]),
            }
        )

    dataset_hash = content_hash(dataset_payload(queries, grades_by_query))
    retrieved_results_hash = content_hash(rankings_payload)
    run_payload: dict[str, JSONValue] = {
        "dataset_hash": dataset_hash,
        "evaluation_mode": "precomputed",
        "metric_version": 1,
        "name": name,
        "relevance_level": "document",
        "retrieved_results_hash": retrieved_results_hash,
        "top_k": list(top_k),
    }
    run_id = content_hash(run_payload)
    manifest: dict[str, JSONValue] = {
        **run_payload,
        "query_count": len(queries),
        "query_ids": [query.id for query in queries],
    }
    return manifest, run_id


__all__ = ["RetrievedQueryResult", "evaluate_results"]
