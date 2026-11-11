from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from retrieval_lab import (
    Chunk,
    HybridRetriever,
    SearchResult,
    evaluate_fused_results,
    fuse_rankings,
    fuse_results,
)
from retrieval_lab.comparison import compare_runs
from retrieval_lab.datasets import EvaluationDataset
from retrieval_lab.domain import EvaluationQuery, EvaluationResult
from retrieval_lab.evaluation.engine import content_hash
from retrieval_lab.evaluation.precomputed import RetrievedQueryResult, evaluate_results
from retrieval_lab.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    RetrieverContractError,
)
from retrieval_lab.retrievers import BaseRetriever


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
    with pytest.raises(RetrieverContractError, match="must be a sequence"):
        fuse_rankings({"a": {"doc-1"}, "b": ["doc-2"]})  # type: ignore[arg-type]
    with pytest.raises(RetrieverContractError, match="non-empty strings"):
        fuse_rankings({"a": ["doc-1", ""], "b": ["doc-2"]})
    with pytest.raises(RetrieverContractError, match="non-empty strings"):
        fuse_rankings({"a": ["doc-1", 1], "b": ["doc-2"]})  # type: ignore[list-item]
    with pytest.raises(RetrieverContractError, match="unique"):
        fuse_rankings({"a": ["doc-1", "doc-1"], "b": ["doc-2"]})
    with pytest.raises(RetrieverContractError, match="rrf_k"):
        fuse_rankings({"a": ["doc-1"], "b": ["doc-2"]}, rrf_k=0)
    with pytest.raises(RetrieverContractError, match="rrf_k"):
        fuse_rankings({"a": ["doc-1"], "b": ["doc-2"]}, rrf_k=True)


def test_fuse_rankings_three_sources_exact_tie() -> None:
    fused = fuse_rankings(
        {
            "a": ["doc-1", "doc-2", "doc-3"],
            "b": ["doc-2", "doc-3", "doc-1"],
            "c": ["doc-3", "doc-1", "doc-2"],
        }
    )

    # Every document scores exactly 1/61 + 1/62 + 1/63, summed in a different
    # order per document; exact arithmetic keeps the tie, so identifiers win.
    assert fused == ("doc-1", "doc-2", "doc-3")


def test_fuse_rankings_allows_empty_rankings() -> None:
    assert fuse_rankings({"a": [], "b": []}) == ()
    assert fuse_rankings({"a": [], "b": ["doc-1"]}) == ("doc-1",)


def test_fuse_rankings_matches_hybrid_retriever() -> None:
    rankings = {
        "bm25": ["doc-1", "doc-2", "doc-3"],
        "dense": ["doc-2", "doc-1", "doc-4"],
    }
    chunks = [_chunk(identifier) for identifier in ("doc-1", "doc-2", "doc-3", "doc-4")]
    sources = [_StubRetriever(name, rankings[name]) for name in ("bm25", "dense")]
    hybrid = HybridRetriever(sources, rrf_k=60, candidate_k=100)
    hybrid.index(chunks)

    fused_order = [result.chunk_id for result in hybrid.search("query", 4)]

    assert tuple(fused_order) == fuse_rankings(rankings)


class _StubRetriever(BaseRetriever):
    def __init__(self, strategy_name: str, ranking: list[str]) -> None:
        self._name = strategy_name
        self._ranking = tuple(ranking)

    @property
    def name(self) -> str:
        return self._name

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = chunks

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id=identifier,
                document_id=f"doc-{identifier}",
                text=f"text-{identifier}",
                score=1.0,
                rank=rank,
            )
            for rank, identifier in enumerate(self._ranking[:top_k], start=1)
        ]


def _chunk(identifier: str) -> Chunk:
    text = f"text-{identifier}"
    return Chunk(
        id=identifier,
        document_id=f"doc-{identifier}",
        text=text,
        start_offset=0,
        end_offset=len(text),
    )


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
    assert recall_at_1.aggregate.absolute_delta == pytest.approx(0.25)


def test_evaluate_fused_results_uses_the_requested_name() -> None:
    fused = evaluate_fused_results(
        dataset=_dataset(),
        results=_source_results(),
        top_k=(1,),
        name="bm25+dense",
    )

    assert set(fused.metrics) == {"bm25+dense"}
    assert fused.manifest["name"] == "bm25+dense"


def test_evaluate_fused_results_records_fusion_provenance() -> None:
    dataset = _dataset()
    sources = _source_results()

    fused = evaluate_fused_results(dataset=dataset, results=sources, top_k=(1, 3))

    assert fused.manifest["fusion"] == {"rrf_k": 60, "sources": ["bm25", "dense"]}

    same_rankings_other_k = evaluate_fused_results(
        dataset=dataset, results=sources, rrf_k=1, top_k=(1, 3)
    )
    assert same_rankings_other_k.run_id != fused.run_id

    plain = evaluate_results(
        dataset=dataset,
        retrieved_results=fuse_results(sources),
        top_k=(1, 3),
        name="hybrid",
    )
    assert plain.run_id != fused.run_id


def test_evaluate_fused_results_distinguishes_source_names() -> None:
    dataset = _dataset()
    sources = _source_results()
    renamed = {"sparse": sources["bm25"], "vector": sources["dense"]}

    fused = evaluate_fused_results(dataset=dataset, results=sources, top_k=(1, 3))
    other = evaluate_fused_results(dataset=dataset, results=renamed, top_k=(1, 3))

    assert other.manifest["fusion"] == {"rrf_k": 60, "sources": ["sparse", "vector"]}
    assert other.run_id != fused.run_id


def test_evaluate_fused_results_run_id_pins_the_payload_keys() -> None:
    fused = evaluate_fused_results(
        dataset=_dataset(), results=_source_results(), top_k=(1, 3)
    )

    payload = {
        key: value
        for key, value in fused.manifest.items()
        if key not in ("query_count", "query_ids")
    }
    assert set(payload) == {
        "dataset_hash",
        "evaluation_mode",
        "fusion",
        "metric_version",
        "name",
        "relevance_level",
        "retrieved_results_hash",
        "top_k",
    }
    assert fused.run_id == content_hash(payload)


def test_evaluate_fused_results_survives_a_json_round_trip(tmp_path: Path) -> None:
    fused = evaluate_fused_results(
        dataset=_dataset(), results=_source_results(), top_k=(1, 3)
    )
    path = tmp_path / "fused.json"

    fused.save_json(path)
    loaded = EvaluationResult.load_json(path)

    assert loaded.run_id == fused.run_id
    assert loaded.manifest["fusion"] == {"rrf_k": 60, "sources": ["bm25", "dense"]}
    assert loaded.to_dict() == fused.to_dict()


def test_evaluate_fused_results_rejects_bad_parameters() -> None:
    with pytest.raises(RetrieverContractError, match="rrf_k"):
        evaluate_fused_results(dataset=_dataset(), results=_source_results(), rrf_k=0)
    with pytest.raises(ConfigurationError, match="name"):
        evaluate_fused_results(dataset=_dataset(), results=_source_results(), name="")


def test_evaluate_fused_results_rejects_queries_outside_the_dataset() -> None:
    sources = _source_results()
    for name in sources:
        sources[name] = [
            *sources[name],
            RetrievedQueryResult("q-3", ["doc-a"]),
        ]

    with pytest.raises(DatasetValidationError, match="exactly match"):
        evaluate_fused_results(dataset=_dataset(), results=sources)


def test_evaluate_fused_results_scores_empty_rankings_as_zero() -> None:
    sources = {
        "bm25": [
            RetrievedQueryResult("q-1", []),
            RetrievedQueryResult("q-2", []),
        ],
        "dense": [
            RetrievedQueryResult("q-1", []),
            RetrievedQueryResult("q-2", []),
        ],
    }

    fused = evaluate_fused_results(dataset=_dataset(), results=sources, top_k=(1,))

    assert fused.metrics["hybrid"].metrics_by_cutoff[1]["recall"] == 0.0
