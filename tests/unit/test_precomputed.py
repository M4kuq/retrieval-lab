from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from retrieval_lab.comparison import check_comparability, compare_runs
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
    IncomparableRunError,
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


def _chunk_dataset() -> EvaluationDataset:
    return EvaluationDataset(
        queries=[
            EvaluationQuery(
                id="q-1",
                query="first query",
                relevant_chunk_ids={"chunk-a", "chunk-b"},
            ),
            EvaluationQuery(
                id="q-2",
                query="second query",
                relevant_chunk_ids={"chunk-c"},
            ),
        ],
        relevance_level="chunk",
        relevance_grades_by_query={
            "q-1": {"chunk-a": 3, "chunk-b": 1},
            "q-2": {"chunk-c": 1},
        },
    )


def _chunk_rankings() -> list[RetrievedQueryResult]:
    return [
        RetrievedQueryResult(
            query_id="q-1",
            retrieved_document_ids=["chunk-b", "chunk-x", "chunk-a"],
        ),
        RetrievedQueryResult(query_id="q-2", retrieved_document_ids=[]),
    ]


def test_evaluate_results_computes_hand_calculated_chunk_metrics() -> None:
    result = evaluate_results(
        dataset=_chunk_dataset(),
        retrieved_results=_chunk_rankings(),
        top_k=[3, 1],
    )

    first = result.query_results["precomputed"][0]
    aggregate = result.metrics["precomputed"].metrics_by_cutoff

    # Chunk IDs match chunk-level grades directly; the numbers mirror the
    # document-level hand-calculated example with renamed identifiers.
    assert first.metrics_by_cutoff[1] == {
        "ap": 1.0,
        "hit_rate": 1.0,
        "mrr": 1.0,
        "ndcg": pytest.approx(1.0 / 7.0),
        "precision": 1.0,
        "recall": 0.5,
    }
    assert first.metrics_by_cutoff[3]["recall"] == 1.0
    assert first.metrics_by_cutoff[3]["ap"] == pytest.approx((1.0 + 2 / 3) / 2)
    expected_ndcg_at_3 = (1.0 + 7.0 / math.log2(4)) / (7.0 + 1.0 / math.log2(3))
    assert first.metrics_by_cutoff[3]["ndcg"] == pytest.approx(expected_ndcg_at_3)
    assert aggregate[1]["recall"] == 0.25
    assert result.manifest["relevance_level"] == "chunk"


def test_relevance_level_is_part_of_the_precomputed_run_identity() -> None:
    same_ids_at_chunk_level = EvaluationDataset(
        queries=[
            EvaluationQuery(
                id="q-1",
                query="first query",
                relevant_chunk_ids={"doc-a", "doc-b"},
            ),
            EvaluationQuery(
                id="q-2",
                query="second query",
                relevant_chunk_ids={"doc-c"},
            ),
        ],
        relevance_level="chunk",
        relevance_grades_by_query={
            "q-1": {"doc-a": 3, "doc-b": 1},
            "q-2": {"doc-c": 1},
        },
    )
    document = evaluate_results(
        dataset=_dataset(),
        retrieved_results=_rankings(),
        top_k=[1, 3],
    )
    chunk = evaluate_results(
        dataset=same_ids_at_chunk_level,
        retrieved_results=_rankings(),
        top_k=[1, 3],
    )

    assert document.manifest["relevance_level"] == "document"
    assert chunk.manifest["relevance_level"] == "chunk"
    assert document.run_id != chunk.run_id
    assert document.metrics == chunk.metrics


def test_chunk_manifest_records_a_deterministic_derived_chunk_hash() -> None:
    result = evaluate_results(
        dataset=_chunk_dataset(),
        retrieved_results=_chunk_rankings(),
        top_k=[1, 3],
    )
    rerun = evaluate_results(
        dataset=_chunk_dataset(),
        retrieved_results=_chunk_rankings(),
        top_k=[1, 3],
    )

    chunk_hash = result.manifest["chunk_hash"]
    assert isinstance(chunk_hash, str)
    assert len(chunk_hash) == 64
    assert rerun.manifest["chunk_hash"] == chunk_hash
    assert rerun.run_id == result.run_id


def test_derived_chunk_hash_tracks_the_chunk_set_not_the_grades() -> None:
    regraded = EvaluationDataset(
        queries=[
            EvaluationQuery(
                id="q-1",
                query="first query",
                relevant_chunk_ids={"chunk-a", "chunk-b"},
            ),
            EvaluationQuery(
                id="q-2",
                query="second query",
                relevant_chunk_ids={"chunk-c"},
            ),
        ],
        relevance_level="chunk",
        relevance_grades_by_query={
            "q-1": {"chunk-a": 1, "chunk-b": 2},
            "q-2": {"chunk-c": 3},
        },
    )
    baseline = evaluate_results(
        dataset=_chunk_dataset(),
        retrieved_results=_chunk_rankings(),
        top_k=[1, 3],
    )
    candidate = evaluate_results(
        dataset=regraded,
        retrieved_results=_chunk_rankings(),
        top_k=[1, 3],
    )

    assert candidate.manifest["chunk_hash"] == baseline.manifest["chunk_hash"]
    assert candidate.manifest["dataset_hash"] != baseline.manifest["dataset_hash"]


def test_chunk_runs_compare_with_themselves_and_each_other() -> None:
    baseline = evaluate_results(
        dataset=_chunk_dataset(),
        retrieved_results=_chunk_rankings(),
        top_k=[1, 3],
    )
    candidate = evaluate_results(
        dataset=_chunk_dataset(),
        retrieved_results=[
            RetrievedQueryResult("q-1", ["chunk-a", "chunk-b"]),
            RetrievedQueryResult("q-2", ["chunk-c"]),
        ],
        top_k=[1, 3],
    )

    assert check_comparability(baseline, baseline).comparable is True
    comparison = compare_runs(baseline, candidate)
    assert comparison.comparability.comparable is True
    assert set(comparison.comparisons) == {"precomputed"}


def test_chunk_runs_with_different_chunk_sets_stay_incomparable() -> None:
    other_dataset = EvaluationDataset(
        queries=[
            EvaluationQuery(
                id="q-1",
                query="first query",
                relevant_chunk_ids={"chunk-a", "chunk-z"},
            ),
            EvaluationQuery(
                id="q-2",
                query="second query",
                relevant_chunk_ids={"chunk-c"},
            ),
        ],
        relevance_level="chunk",
        relevance_grades_by_query={
            "q-1": {"chunk-a": 3, "chunk-z": 1},
            "q-2": {"chunk-c": 1},
        },
    )
    baseline = evaluate_results(
        dataset=_chunk_dataset(),
        retrieved_results=_chunk_rankings(),
        top_k=[1, 3],
    )
    candidate = evaluate_results(
        dataset=other_dataset,
        retrieved_results=_chunk_rankings(),
        top_k=[1, 3],
    )

    assert candidate.manifest["chunk_hash"] != baseline.manifest["chunk_hash"]
    report = check_comparability(baseline, candidate)
    assert report.comparable is False
    assert any(issue.field == "manifest.chunk_hash" for issue in report.issues)
    with pytest.raises(IncomparableRunError):
        compare_runs(baseline, candidate)


def test_explicit_chunk_hash_overrides_the_derived_identity() -> None:
    first = evaluate_results(
        dataset=_chunk_dataset(),
        retrieved_results=_chunk_rankings(),
        top_k=[1, 3],
        chunk_hash="a" * 64,
    )
    matching = evaluate_results(
        dataset=_chunk_dataset(),
        retrieved_results=_chunk_rankings(),
        top_k=[1, 3],
        chunk_hash="a" * 64,
    )
    other = evaluate_results(
        dataset=_chunk_dataset(),
        retrieved_results=_chunk_rankings(),
        top_k=[1, 3],
        chunk_hash="b" * 64,
    )

    assert first.manifest["chunk_hash"] == "a" * 64
    assert first.run_id != other.run_id
    assert check_comparability(first, matching).comparable is True
    report = check_comparability(first, other)
    assert report.comparable is False
    assert any(issue.field == "manifest.chunk_hash" for issue in report.issues)


def test_document_manifest_omits_chunk_hash_unless_one_is_supplied() -> None:
    default = evaluate_results(dataset=_dataset(), retrieved_results=_rankings())
    supplied = evaluate_results(
        dataset=_dataset(),
        retrieved_results=_rankings(),
        chunk_hash="c" * 64,
    )

    assert "chunk_hash" not in default.manifest
    assert supplied.manifest["chunk_hash"] == "c" * 64
    assert supplied.run_id != default.run_id


@pytest.mark.parametrize("chunk_hash", ["", "   ", 42])
def test_evaluate_results_rejects_an_invalid_chunk_hash(chunk_hash: object) -> None:
    with pytest.raises(ConfigurationError, match="chunk_hash"):
        evaluate_results(
            dataset=_chunk_dataset(),
            retrieved_results=_chunk_rankings(),
            chunk_hash=chunk_hash,  # type: ignore[arg-type]
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
