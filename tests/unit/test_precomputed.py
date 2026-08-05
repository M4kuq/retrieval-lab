from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from retrieval_lab.datasets import EvaluationDataset
from retrieval_lab.domain import EvaluationQuery
from retrieval_lab.evaluation.precomputed import (
    RetrievedQueryResult,
    evaluate_results,
)
from retrieval_lab.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    EvaluationError,
    RetrieverContractError,
)


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        queries=[
            EvaluationQuery(
                id="q-1",
                query="first query",
                relevant_document_ids={"doc-a", "doc-b"},
            ),
            EvaluationQuery(
                id="q-2",
                query="second query",
                relevant_document_ids={"doc-c"},
            ),
        ],
        relevance_grades_by_query={
            "q-1": {"doc-a": 3, "doc-b": 1},
            "q-2": {"doc-c": 1},
        },
    )


def _rankings() -> list[RetrievedQueryResult]:
    return [
        RetrievedQueryResult(
            query_id="q-1",
            retrieved_document_ids=["doc-b", "doc-x", "doc-a"],
        ),
        RetrievedQueryResult(query_id="q-2", retrieved_document_ids=[]),
    ]


def test_evaluate_results_computes_hand_calculated_metrics_and_macro() -> None:
    result = evaluate_results(
        dataset=_dataset(),
        retrieved_results=_rankings(),
        top_k=[3, 1],
    )

    first = result.query_results["precomputed"][0]
    empty = result.query_results["precomputed"][1]
    aggregate = result.metrics["precomputed"].metrics_by_cutoff

    assert tuple(aggregate) == (1, 3)
    assert first.metrics_by_cutoff[1] == {
        "ap": 1.0,
        "hit_rate": 1.0,
        "mrr": 1.0,
        "ndcg": pytest.approx(1.0 / 7.0),
        "precision": 1.0,
        "recall": 0.5,
    }
    assert first.metrics_by_cutoff[3]["precision"] == pytest.approx(2.0 / 3.0)
    assert first.metrics_by_cutoff[3]["recall"] == 1.0
    assert first.metrics_by_cutoff[3]["ap"] == pytest.approx((1.0 + 2 / 3) / 2)

    # Observed gains are [1, 0, 7]; ideal gains are [7, 1].
    expected_ndcg_at_3 = (1.0 + 7.0 / math.log2(4)) / (7.0 + 1.0 / math.log2(3))
    assert first.metrics_by_cutoff[3]["ndcg"] == pytest.approx(expected_ndcg_at_3)
    assert set(empty.metrics_by_cutoff[1].values()) == {0.0}

    # The empty retrieval remains in the two-query macro denominator.
    assert aggregate[1]["hit_rate"] == 0.5
    assert aggregate[1]["recall"] == 0.25
    assert aggregate[1]["precision"] == 0.5
    assert aggregate[3]["recall"] == 0.5
    assert aggregate[3]["ndcg"] == pytest.approx(expected_ndcg_at_3 / 2)


def test_retrieved_query_result_defensively_copies_mutable_input() -> None:
    identifiers = ["doc-a"]

    result = RetrievedQueryResult("q-1", identifiers)
    identifiers.append("doc-b")

    assert result.retrieved_document_ids == ("doc-a",)


@pytest.mark.parametrize(
    ("query_id", "identifiers", "error_type", "message"),
    [
        ("", [], DatasetValidationError, "query_id"),
        ("q-1", "doc-a", RetrieverContractError, "not a string"),
        ("q-1", [""], RetrieverContractError, "non-empty"),
        ("q-1", ["doc-a", "doc-a"], RetrieverContractError, "unique"),
    ],
)
def test_retrieved_query_result_rejects_invalid_identifiers(
    query_id: str,
    identifiers: Sequence[str],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        RetrievedQueryResult(query_id, identifiers)


@pytest.mark.parametrize("top_k", [[], [0], [True], [1, 1], "1"])
def test_evaluate_results_rejects_invalid_top_k(top_k: object) -> None:
    with pytest.raises(ConfigurationError):
        evaluate_results(
            dataset=_dataset(),
            retrieved_results=_rankings(),
            top_k=top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("rankings", "message"),
    [
        ([RetrievedQueryResult("q-1", [])], "missing"),
        (
            [
                RetrievedQueryResult("q-1", []),
                RetrievedQueryResult("unknown", []),
            ],
            "unknown",
        ),
        (
            [
                RetrievedQueryResult("q-1", []),
                RetrievedQueryResult("q-1", []),
                RetrievedQueryResult("q-2", []),
            ],
            "duplicate",
        ),
    ],
)
def test_evaluate_results_requires_exactly_one_result_per_query(
    rankings: list[RetrievedQueryResult],
    message: str,
) -> None:
    with pytest.raises(DatasetValidationError, match=message):
        evaluate_results(dataset=_dataset(), retrieved_results=rankings)


def test_evaluate_results_rejects_chunk_relevance_explicitly() -> None:
    dataset = EvaluationDataset(
        queries=[
            EvaluationQuery(
                id="q",
                query="query",
                relevant_chunk_ids={"chunk"},
            )
        ],
        relevance_level="chunk",
    )
    with pytest.raises(DatasetValidationError, match="only document"):
        evaluate_results(
            dataset=dataset,
            retrieved_results=[RetrievedQueryResult("q", [])],
        )


def test_evaluate_results_is_deterministic_and_uses_custom_name() -> None:
    first = evaluate_results(
        dataset=_dataset(),
        retrieved_results=_rankings(),
        top_k=[1, 3],
        name="production",
    )
    second = evaluate_results(
        dataset=_dataset(),
        retrieved_results=list(reversed(_rankings())),
        top_k=[3, 1],
        name="production",
    )

    assert first.to_json() == second.to_json()
    assert len(first.run_id) == 64
    assert tuple(first.metrics) == ("production",)
    assert first.manifest["evaluation_mode"] == "precomputed"
    assert first.manifest["relevance_level"] == "document"
    assert first.manifest["query_count"] == 2


def test_evaluate_results_rejects_empty_name() -> None:
    with pytest.raises(ConfigurationError, match="name"):
        evaluate_results(
            dataset=_dataset(),
            retrieved_results=_rankings(),
            name="  ",
        )


def test_evaluate_results_rejects_non_dataset_object() -> None:
    with pytest.raises(DatasetValidationError, match="EvaluationDataset"):
        evaluate_results(
            dataset=object(),  # type: ignore[arg-type]
            retrieved_results=_rankings(),
        )


@pytest.mark.parametrize("rankings", ["q-1", [object()]])
def test_evaluate_results_rejects_invalid_result_collection(rankings: object) -> None:
    with pytest.raises(DatasetValidationError, match="retrieved_results"):
        evaluate_results(
            dataset=_dataset(),
            retrieved_results=rankings,  # type: ignore[arg-type]
        )


def test_evaluate_results_wraps_metric_overflow() -> None:
    query = EvaluationQuery(
        id="q",
        query="query",
        relevant_document_ids={"doc"},
    )
    dataset = EvaluationDataset(
        queries=[query],
        relevance_grades_by_query={"q": {"doc": 1024}},
    )

    with pytest.raises(EvaluationError, match="metrics could not be computed"):
        evaluate_results(
            dataset=dataset,
            retrieved_results=[RetrievedQueryResult("q", ["doc"])],
            top_k=[1],
        )
