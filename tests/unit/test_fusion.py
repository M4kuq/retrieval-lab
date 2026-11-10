from __future__ import annotations

import pytest

from retrieval_lab import evaluate_fused_results, fuse_rankings, fuse_results
from retrieval_lab.comparison import compare_runs
from retrieval_lab.datasets import EvaluationDataset
from retrieval_lab.domain import EvaluationQuery
from retrieval_lab.evaluation.precomputed import RetrievedQueryResult
from retrieval_lab.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    RetrieverContractError,
)


def test_fuse_rankings_hand_calculated_with_ties() -> None:
    fused = fuse_rankings(
        {
            "bm25": ["doc-1", "doc-2", "doc-3"],
            "dense": ["doc-2", "doc-1", "doc-4"],
        }
    )

    # doc-1 and doc-2 both score 1/61 + 1/62; doc-3 and doc-4 both score
    # 1/63. Ties break by identifier.
    assert fused == ("doc-1", "doc-2", "doc-3", "doc-4")


def test_fuse_rankings_is_independent_of_source_order() -> None:
    rankings = {
        "bm25": ["doc-1", "doc-2", "doc-3"],
        "dense": ["doc-2", "doc-1", "doc-4"],
    }

    assert fuse_rankings(rankings) == fuse_rankings(
        dict(reversed(list(rankings.items())))
    )


def test_fuse_rankings_hand_calculated_small_rrf_k() -> None:
    fused = fuse_rankings(
        {"a": ["doc-1", "doc-2"], "b": ["doc-3", "doc-1"]},
        rrf_k=1,
    )

    # doc-1: 1/2 + 1/3, doc-3: 1/2, doc-2: 1/3.
    assert fused == ("doc-1", "doc-3", "doc-2")


def test_fuse_rankings_rejects_invalid_inputs() -> None:
    with pytest.raises(DatasetValidationError, match="at least two"):
        fuse_rankings({"only": ["doc-1"]})
    with pytest.raises(DatasetValidationError, match="mapping"):
        fuse_rankings([["doc-1"], ["doc-2"]])  # type: ignore[arg-type]
    with pytest.raises(DatasetValidationError, match="non-empty"):
        fuse_rankings({"": ["doc-1"], "b": ["doc-2"]})
    with pytest.raises(RetrieverContractError, match="not a string"):
        fuse_rankings({"a": "doc-1", "b": ["doc-2"]})  # type: ignore[arg-type]
    with pytest.raises(RetrieverContractError, match="non-empty strings"):
        fuse_rankings({"a": ["doc-1", ""], "b": ["doc-2"]})
    with pytest.raises(RetrieverContractError, match="unique"):
        fuse_rankings({"a": ["doc-1", "doc-1"], "b": ["doc-2"]})
    with pytest.raises(ConfigurationError, match="rrf_k"):
        fuse_rankings({"a": ["doc-1"], "b": ["doc-2"]}, rrf_k=0)
    with pytest.raises(ConfigurationError, match="rrf_k"):
        fuse_rankings({"a": ["doc-1"], "b": ["doc-2"]}, rrf_k=True)


def _source_results() -> dict[str, list[RetrievedQueryResult]]:
    return {
        "bm25": [
            RetrievedQueryResult("q-2", ["doc-y"]),
            RetrievedQueryResult("q-1", ["doc-a", "doc-x"]),
        ],
        "dense": [
            RetrievedQueryResult("q-1", ["doc-x", "doc-b"]),
            RetrievedQueryResult("q-2", ["doc-c"]),
        ],
    }


def test_fuse_results_fuses_per_query_in_query_order() -> None:
    fused = fuse_results(_source_results())

    assert [result.query_id for result in fused] == ["q-1", "q-2"]
    # q-1: doc-x scores 1/62 + 1/61, doc-a 1/61, doc-b 1/62.
    assert fused[0].retrieved_document_ids == ("doc-x", "doc-a", "doc-b")
    # q-2: doc-y and doc-c both score 1/61; the tie breaks by identifier.
    assert fused[1].retrieved_document_ids == ("doc-c", "doc-y")


def test_fuse_results_rejects_invalid_inputs() -> None:
    with pytest.raises(DatasetValidationError, match="at least two"):
        fuse_results({"only": [RetrievedQueryResult("q-1", ["doc-1"])]})
    with pytest.raises(DatasetValidationError, match="duplicate query IDs"):
        fuse_results(
            {
                "a": [
                    RetrievedQueryResult("q-1", ["doc-1"]),
                    RetrievedQueryResult("q-1", ["doc-2"]),
                ],
                "b": [RetrievedQueryResult("q-1", ["doc-3"])],
            }
        )
    with pytest.raises(DatasetValidationError, match="same query IDs"):
        fuse_results(
            {
                "a": [RetrievedQueryResult("q-1", ["doc-1"])],
                "b": [RetrievedQueryResult("q-2", ["doc-3"])],
            }
        )
    with pytest.raises(DatasetValidationError, match="RetrievedQueryResult"):
        fuse_results(
            {
                "a": [RetrievedQueryResult("q-1", ["doc-1"])],
                "b": ["q-1"],  # type: ignore[list-item]
            }
        )
    with pytest.raises(DatasetValidationError, match="mapping"):
        fuse_results([RetrievedQueryResult("q-1", ["doc-1"])])  # type: ignore[arg-type]
    with pytest.raises(DatasetValidationError, match="non-empty"):
        fuse_results(
            {
                "": [RetrievedQueryResult("q-1", ["doc-1"])],
                "b": [RetrievedQueryResult("q-1", ["doc-2"])],
            }
        )
    with pytest.raises(DatasetValidationError, match="sequence"):
        fuse_results(
            {
                "a": [RetrievedQueryResult("q-1", ["doc-1"])],
                "b": "q-1",  # type: ignore[dict-item]
            }
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


def test_evaluate_fused_results_produces_a_comparable_run() -> None:
    dataset = _dataset()
    sources = _source_results()

    fused = evaluate_fused_results(dataset=dataset, results=sources, top_k=(1, 3))

    assert fused.query_results["hybrid"][0].retrieved_ids == (
        "doc-x",
        "doc-a",
        "doc-b",
    )
    assert fused.metrics["hybrid"].metrics_by_cutoff[1]["recall"] == 0.5
    assert fused.metrics["hybrid"].metrics_by_cutoff[3]["recall"] == 1.0
    assert fused.manifest["name"] == "hybrid"
    assert fused.manifest["evaluation_mode"] == "precomputed"

    variant_sources = _source_results()
    variant_sources["dense"] = [
        RetrievedQueryResult("q-1", ["doc-b", "doc-a"]),
        RetrievedQueryResult("q-2", ["doc-c"]),
    ]
    variant = evaluate_fused_results(
        dataset=dataset, results=variant_sources, top_k=(1, 3)
    )

    comparison = compare_runs(fused, variant)
    assert comparison.comparability.comparable
    recall_at_1 = next(
        item
        for item in comparison.metrics["hybrid"]
        if item.metric == "recall" and item.cutoff == 1
    )
    # Variant q-1 fuses to doc-a first (recall@1 0.5 instead of 0.0), so the
    # macro recall@1 moves from 0.5 to 0.75.
    assert recall_at_1.aggregate.absolute_delta == 0.25


def test_evaluate_fused_results_uses_the_requested_name() -> None:
    fused = evaluate_fused_results(
        dataset=_dataset(),
        results=_source_results(),
        top_k=(1,),
        name="bm25+dense",
    )

    assert set(fused.metrics) == {"bm25+dense"}
    assert fused.manifest["name"] == "bm25+dense"
