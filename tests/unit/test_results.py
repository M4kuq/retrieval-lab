import json
from pathlib import Path

import pytest

from retrieval_lab.domain import (
    EvaluationResult,
    LatencyStats,
    QueryEvaluation,
    RetrieverMetrics,
)
from retrieval_lab.exceptions import EvaluationError


def _query_evaluation(query_id: str = "q-1") -> QueryEvaluation:
    return QueryEvaluation(
        query_id=query_id,
        retrieved_ids=("doc-2", "doc-1"),
        metrics_by_cutoff={
            1: {"recall": 0.0, "hit_rate": 0.0},
            2: {"recall": 1.0, "hit_rate": 1.0},
        },
    )


def _retriever_metrics() -> RetrieverMetrics:
    return RetrieverMetrics(
        metrics_by_cutoff={
            1: {"recall": 0.5, "hit_rate": 0.5},
            2: {"recall": 1.0, "hit_rate": 1.0},
        }
    )


def _result() -> EvaluationResult:
    return EvaluationResult(
        run_id="run-日本語",
        metrics={"keyword": _retriever_metrics()},
        query_results={"keyword": [_query_evaluation()]},
        manifest={"dataset": "日本語データ", "seed": 42},
    )


def test_query_evaluation_normalizes_sequences_and_metrics() -> None:
    retrieved = ["doc-1"]
    values = {3: {"recall": 1}}

    result = QueryEvaluation("q-1", retrieved, values)  # type: ignore[arg-type]
    retrieved.append("doc-2")
    values[3]["recall"] = 0

    assert result.retrieved_ids == ("doc-1",)
    assert result.metrics_by_cutoff[3]["recall"] == 1.0
    assert result.to_dict() == {
        "metrics": {"recall@3": 1.0},
        "query_id": "q-1",
        "retrieved_ids": ["doc-1"],
    }


def test_query_evaluation_translates_latency_float_overflow() -> None:
    with pytest.raises(EvaluationError):
        QueryEvaluation(
            query_id="q-1",
            retrieved_ids=(),
            metrics_by_cutoff={1: {"recall": 0.0}},
            search_latency_ms=10**10000,
        )


def test_result_rejects_mixed_per_query_latency_with_aggregate() -> None:
    timed_base = _query_evaluation("q-timed")
    timed = QueryEvaluation(
        query_id=timed_base.query_id,
        retrieved_ids=timed_base.retrieved_ids,
        metrics_by_cutoff=timed_base.metrics_by_cutoff,
        search_latency_ms=1.0,
    )
    untimed = QueryEvaluation(
        query_id="q-untimed",
        retrieved_ids=("doc-2",),
        metrics_by_cutoff=timed.metrics_by_cutoff,
    )
    with pytest.raises(EvaluationError, match="partial per-query latency"):
        EvaluationResult(
            run_id="run",
            metrics={"keyword": RetrieverMetrics({1: {"recall": 0.5}})},
            query_results={"keyword": (timed, untimed)},
            latency={
                "keyword": LatencyStats(
                    mean_ms=1.0,
                    p50_ms=1.0,
                    p95_ms=1.0,
                    max_ms=1.0,
                    sample_count=1,
                    failure_count=1,
                )
            },
        )


@pytest.mark.parametrize(
    ("retrieved_ids", "error"),
    [
        (("doc-1", "doc-1"), "unique"),
        (("",), "non-empty"),
        ("doc-1", "sequence"),
    ],
)
def test_query_evaluation_rejects_invalid_retrieved_ids(
    retrieved_ids: object, error: str
) -> None:
    with pytest.raises(EvaluationError, match=error):
        QueryEvaluation(
            "q-1",
            retrieved_ids,  # type: ignore[arg-type]
            {1: {"recall": 1.0}},
        )


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {0: {"recall": 1.0}},
        {1: {}},
        {1: {"": 1.0}},
        {1: {"recall": float("nan")}},
        {1: {"recall": float("inf")}},
    ],
)
def test_retriever_metrics_rejects_invalid_metric_mappings(
    metrics: object,
) -> None:
    with pytest.raises(EvaluationError):
        RetrieverMetrics(metrics)  # type: ignore[arg-type]


def test_retriever_metrics_returns_macro_recall() -> None:
    metrics = _retriever_metrics()

    assert metrics.recall_at(1) == 0.5
    assert metrics.to_dict() == {
        "hit_rate@1": 0.5,
        "hit_rate@2": 1.0,
        "recall@1": 0.5,
        "recall@2": 1.0,
    }


def test_recall_at_raises_key_error_for_unevaluated_cutoff() -> None:
    with pytest.raises(KeyError, match="Recall@5"):
        _retriever_metrics().recall_at(5)


def test_recall_at_raises_key_error_when_recall_metric_is_absent() -> None:
    metrics = RetrieverMetrics({1: {"hit_rate": 1.0}})

    with pytest.raises(KeyError, match="Recall@1"):
        metrics.recall_at(1)


def test_result_uses_canonical_technical_design_shape() -> None:
    payload = _result().to_dict()

    assert payload == {
        "quality_gates": [],
        "retrievers": {
            "keyword": {
                "metrics": {
                    "hit_rate@1": 0.5,
                    "hit_rate@2": 1.0,
                    "recall@1": 0.5,
                    "recall@2": 1.0,
                },
                "per_query": [
                    {
                        "metrics": {
                            "hit_rate@1": 0.0,
                            "hit_rate@2": 1.0,
                            "recall@1": 0.0,
                            "recall@2": 1.0,
                        },
                        "query_id": "q-1",
                        "retrieved_ids": ["doc-2", "doc-1"],
                    }
                ],
            }
        },
        "run": {
            "id": "run-日本語",
            "manifest": {"dataset": "日本語データ", "seed": 42},
        },
        "schema_version": 1,
    }


def test_json_is_deterministic_preserves_japanese_and_has_final_newline() -> None:
    result = _result()

    first = result.to_json()
    second = result.to_json()

    assert first == second
    assert "日本語" in first
    assert "\\u65e5" not in first
    assert first.endswith("\n") and not first.endswith("\n\n")
    assert json.loads(first) == result.to_dict()
    assert first.index('"quality_gates"') < first.index('"retrievers"')


def test_save_json_creates_parents_and_writes_utf8(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "結果.json"

    _result().save_json(output)

    assert output.read_text(encoding="utf-8") == _result().to_json()


def test_save_json_wraps_low_level_write_errors(tmp_path: Path) -> None:
    directory = tmp_path / "already-a-directory"
    directory.mkdir()

    with pytest.raises(EvaluationError, match="could not be saved"):
        _result().save_json(directory)


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "run_id": "",
            "metrics": {"keyword": _retriever_metrics()},
            "query_results": {"keyword": [_query_evaluation()]},
        },
        {"run_id": "run", "metrics": {}, "query_results": {}},
        {
            "run_id": "run",
            "metrics": {"": _retriever_metrics()},
            "query_results": {"": [_query_evaluation()]},
        },
        {
            "run_id": "run",
            "metrics": {"keyword": object()},
            "query_results": {"keyword": [_query_evaluation()]},
        },
        {
            "run_id": "run",
            "metrics": {"keyword": _retriever_metrics()},
            "query_results": {"other": [_query_evaluation()]},
        },
        {
            "run_id": "run",
            "metrics": {"keyword": _retriever_metrics()},
            "query_results": {"keyword": []},
        },
        {
            "run_id": "run",
            "metrics": {"keyword": _retriever_metrics()},
            "query_results": {"keyword": [_query_evaluation(), _query_evaluation()]},
        },
        {
            "run_id": "run",
            "metrics": {"keyword": _retriever_metrics()},
            "query_results": {"keyword": [object()]},
        },
        {
            "run_id": "run",
            "metrics": {"keyword": _retriever_metrics()},
            "query_results": {"keyword": [_query_evaluation()]},
            "schema_version": 2,
        },
    ],
)
def test_evaluation_result_rejects_invalid_inputs(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(EvaluationError):
        EvaluationResult(**arguments)  # type: ignore[arg-type]


def test_result_rejects_non_json_manifest() -> None:
    with pytest.raises(EvaluationError, match="manifest"):
        EvaluationResult(
            run_id="run",
            metrics={"keyword": _retriever_metrics()},
            query_results={"keyword": [_query_evaluation()]},
            manifest={"bad": object()},  # type: ignore[dict-item]
        )
