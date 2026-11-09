import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from retrieval_lab.artifacts import results as results_module
from retrieval_lab.domain import (
    EvaluationResult,
    LatencyStats,
    QualityGateCheck,
    QualityGateReport,
    QualityGateResult,
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


def test_query_evaluation_round_trips_cutoff_ranking_evidence() -> None:
    query = QueryEvaluation(
        query_id="q-1",
        retrieved_ids=("doc-a", "doc-b"),
        metrics_by_cutoff={
            2: {"recall": 0.0},
            3: {"recall": 1.0},
        },
        retrieved_ids_by_cutoff={
            2: ("doc-a",),
            3: ("doc-a", "doc-b"),
        },
    )
    result = EvaluationResult(
        run_id="run",
        metrics={"system": RetrieverMetrics({2: {"recall": 0.0}, 3: {"recall": 1.0}})},
        query_results={"system": [query]},
    )

    loaded = EvaluationResult.from_json(result.to_json())

    assert loaded.query_results["system"][0].retrieved_ids_by_cutoff == {
        2: ("doc-a",),
        3: ("doc-a", "doc-b"),
    }


@pytest.mark.parametrize(
    "evidence",
    [
        {2: ("doc-a",)},
        {2: ("doc-a", "doc-a"), 3: ("doc-a", "doc-b")},
        {2: ("doc-b",), 3: ("doc-a", "doc-b")},
        {True: ("doc-a",), 3: ("doc-a", "doc-b")},
        {2.0: ("doc-a",), 3: ("doc-a", "doc-b")},
        {"2": ("doc-a",), 3: ("doc-a", "doc-b")},
        {2: ("doc-a", "doc-b"), 3: ("doc-a",)},
        {2: (), 3: ("doc-a", "doc-b")},
    ],
)
def test_query_evaluation_rejects_invalid_cutoff_evidence(
    evidence: object,
) -> None:
    with pytest.raises(EvaluationError, match="retrieved_ids_by_cutoff"):
        QueryEvaluation(
            query_id="q-1",
            retrieved_ids=("doc-a", "doc-b"),
            metrics_by_cutoff={2: {"recall": 0.0}, 3: {"recall": 1.0}},
            retrieved_ids_by_cutoff=evidence,  # type: ignore[arg-type]
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


def test_load_result_rejects_known_oversize_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.json"
    path.write_text(_result().to_json(), encoding="utf-8")
    original_open = results_module.os.open
    original_read = results_module.os.read
    read_sizes: list[int] = []

    def recording_open(source: Path, flags: int) -> int:
        assert source == path
        return original_open(source, flags)

    def recording_read(descriptor: int, size: int) -> bytes:
        read_sizes.append(size)
        return original_read(descriptor, size)

    monkeypatch.setattr(results_module.os, "open", recording_open)
    monkeypatch.setattr(results_module.os, "read", recording_read)
    with pytest.raises(EvaluationError, match="max_bytes"):
        EvaluationResult.load_json(path, max_bytes=1)

    assert read_sizes == []


def test_load_result_handles_short_reads_and_sysmax_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.json"
    valid_result = EvaluationResult(
        run_id="run-日本語",
        metrics={
            "keyword": RetrieverMetrics(
                {
                    1: {"recall": 0.0, "hit_rate": 0.0},
                    2: {"recall": 1.0, "hit_rate": 1.0},
                }
            )
        },
        query_results={"keyword": [_query_evaluation()]},
        manifest={"dataset": "日本語データ", "seed": 42},
    )
    path.write_text(valid_result.to_json(), encoding="utf-8")
    original_read = results_module.os.read
    calls = 0

    def short_read(descriptor: int, size: int) -> bytes:
        nonlocal calls
        calls += 1
        return original_read(descriptor, min(size, 1))

    monkeypatch.setattr(results_module.os, "read", short_read)

    assert (
        EvaluationResult.load_json(path, max_bytes=sys.maxsize).run_id == "run-日本語"
    )
    assert calls > 100


def test_load_result_detects_growth_after_fstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.json"
    path.write_text(_result().to_json(), encoding="utf-8")
    original_fstat = results_module.os.fstat

    def pretend_empty(descriptor: int) -> os.stat_result:
        result = original_fstat(descriptor)
        values = list(result)
        values[6] = 0
        return os.stat_result(values)

    monkeypatch.setattr(results_module.os, "fstat", pretend_empty)
    with pytest.raises(EvaluationError, match="max_bytes"):
        EvaluationResult.load_json(path, max_bytes=1)


def test_load_result_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes are unavailable")
    path = tmp_path / "result.json"
    try:
        os.mkfifo(path)
    except OSError:
        pytest.skip("named pipes are unavailable")

    with pytest.raises(EvaluationError, match="read result JSON"):
        EvaluationResult.load_json(path)


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


def _parsed_payload() -> dict[str, object]:
    payload = json.loads(_result().to_json())
    second = dict(payload["retrievers"]["keyword"]["per_query"][0])
    second["query_id"] = "q-2"
    second["metrics"] = {
        "hit_rate@1": 1.0,
        "hit_rate@2": 1.0,
        "recall@1": 1.0,
        "recall@2": 1.0,
    }
    payload["retrievers"]["keyword"]["per_query"].append(second)
    return payload


_GATE: dict[str, object] = {
    "baseline_run_id": None,
    "candidate_run_id": "run-日本語",
    "checks": [
        {
            "absolute_tolerance": 1e-12,
            "actual": 0.5,
            "baseline_run_id": None,
            "candidate_run_id": "run-日本語",
            "constraint": "min_value",
            "metric": "recall@1",
            "passed": True,
            "reason": "candidate meets minimum",
            "relative_tolerance": 1e-09,
            "retriever": "keyword",
            "status": "defined",
            "threshold": 0.4,
        }
    ],
    "gate_index": 0,
    "metric": "recall@1",
    "passed": True,
    "retriever": "keyword",
}

_LATENCY: dict[str, object] = {
    "mean_ms": 1.0,
    "p50_ms": 1.0,
    "p95_ms": 1.0,
    "max_ms": 1.0,
    "sample_count": 2,
    "failure_count": 0,
    "warnings": [],
}


def _gated_payload() -> dict[str, object]:
    payload = _parsed_payload()
    payload["quality_gates"] = [json.loads(json.dumps(_GATE))]
    return payload


def _latency_payload() -> dict[str, object]:
    payload = _parsed_payload()
    payload["retrievers"]["keyword"]["latency"] = dict(_LATENCY)  # type: ignore[index]
    return payload


def test_result_from_dict_accepts_canonical_payload_shapes() -> None:
    plain = results_module.result_from_dict(_parsed_payload())
    assert plain.run_id == "run-日本語"

    gated = results_module.result_from_dict(_gated_payload())
    assert gated.quality_gates[0].checks[0].actual == 0.5

    with_latency = results_module.result_from_dict(_latency_payload())
    assert with_latency.latency is not None
    assert with_latency.latency["keyword"].sample_count == 2


def _keyword(payload: dict[str, object]) -> dict[str, object]:
    return payload["retrievers"]["keyword"]  # type: ignore[index,return-value]


def _first_query(payload: dict[str, object]) -> dict[str, object]:
    return _keyword(payload)["per_query"][0]  # type: ignore[index,return-value]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda payload: payload.__setitem__("retrievers", {}),
            r"result\.retrievers must not be empty",
        ),
        (
            lambda payload: payload.__setitem__("retrievers", []),
            r"result\.retrievers must be an object",
        ),
        (
            lambda payload: payload["run"].__setitem__("manifest", {1: "x"}),  # type: ignore[attr-defined]
            r"result\.run\.manifest keys must be strings",
        ),
        (
            lambda payload: payload["run"].__setitem__("id", "  "),  # type: ignore[attr-defined]
            r"result\.run\.id must be a non-empty string",
        ),
        (
            lambda payload: _keyword(payload).__setitem__("metrics", {}),
            "metrics must not be empty",
        ),
        (
            lambda payload: _keyword(payload)["metrics"].__setitem__(  # type: ignore[index]
                "recall@1", "high"
            ),
            "must be a finite number",
        ),
        (
            lambda payload: _keyword(payload)["metrics"].__setitem__(  # type: ignore[index]
                "recall@1", 10**400
            ),
            "must be a finite number",
        ),
        (
            lambda payload: _keyword(payload)["metrics"].__setitem__(  # type: ignore[index]
                "recall@1", float("inf")
            ),
            "must be a finite number",
        ),
        (
            lambda payload: _first_query(payload).__setitem__("retrieved_ids", "doc-2"),
            "retrieved_ids must be a list of strings",
        ),
        (
            lambda payload: _first_query(payload).__setitem__("warnings", "oops"),
            "warnings must be a list of strings",
        ),
        (
            lambda payload: _first_query(payload).__setitem__(
                "search_latency_ms", "fast"
            ),
            "search_latency_ms must be a finite number",
        ),
        (
            lambda payload: _first_query(payload).__setitem__(
                "retrieved_ids_by_cutoff", {"x": ["doc-2"]}
            ),
            "keys must be canonical positive integers",
        ),
        (
            lambda payload: _first_query(payload).__setitem__(
                "retrieved_ids_by_cutoff", {"01": ["doc-2"]}
            ),
            "keys must be canonical positive integers",
        ),
        (
            lambda payload: _first_query(payload).__setitem__(
                "retrieved_ids_by_cutoff", {"1": "doc-2"}
            ),
            r"retrieved_ids_by_cutoff\[1\] must be a list",
        ),
        (
            lambda payload: _first_query(payload).__setitem__(
                "metrics", {"recall@1": 0.0}
            ),
            "metrics shape differs",
        ),
        (
            lambda payload: _keyword(payload)["metrics"].__setitem__(  # type: ignore[index]
                "recall@2", 0.5
            ),
            "does not match per-query aggregate",
        ),
    ],
)
def test_result_from_dict_rejects_invalid_payloads(
    mutate: Callable[[dict[str, object]], None], expected: str
) -> None:
    payload = _parsed_payload()
    mutate(payload)

    with pytest.raises(EvaluationError, match=expected):
        results_module.result_from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("sample_count", -1, "must be a non-negative integer"),
        ("warnings", "oops", "warnings must be a list of strings"),
        ("mean_ms", "fast", "must be a finite number"),
    ],
)
def test_result_from_dict_rejects_invalid_latency(
    field: str, value: object, expected: str
) -> None:
    payload = _latency_payload()
    _keyword(payload)["latency"][field] = value  # type: ignore[index]

    with pytest.raises(EvaluationError, match=expected):
        results_module.result_from_dict(payload)


def _gate(payload: dict[str, object]) -> dict[str, object]:
    return payload["quality_gates"][0]  # type: ignore[index,return-value]


def _gate_check(payload: dict[str, object]) -> dict[str, object]:
    return _gate(payload)["checks"][0]  # type: ignore[index,return-value]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda payload: payload.__setitem__("quality_gates", {}),
            r"result\.quality_gates must be a list",
        ),
        (
            lambda payload: _gate(payload).__setitem__("checks", []),
            "checks must be a non-empty list",
        ),
        (
            lambda payload: _gate(payload).__setitem__("candidate_run_id", "other"),
            "candidate_run_id does not match result run",
        ),
        (
            lambda payload: _gate(payload).__setitem__("gate_index", -1),
            "must be a non-negative integer",
        ),
        (
            lambda payload: _gate_check(payload).__setitem__("passed", "yes"),
            "must be boolean",
        ),
        (
            lambda payload: _gate_check(payload).__setitem__("retriever", "bm25"),
            "retriever does not match gate",
        ),
        (
            lambda payload: _gate_check(payload).__setitem__("metric", "recall@2"),
            "metric does not match gate",
        ),
        (
            lambda payload: _gate_check(payload).__setitem__(
                "candidate_run_id", "other"
            ),
            "candidate_run_id does not match gate",
        ),
        (
            lambda payload: _gate_check(payload).__setitem__("baseline_run_id", "base"),
            "baseline_run_id does not match gate",
        ),
    ],
)
def test_result_from_dict_rejects_invalid_quality_gates(
    mutate: Callable[[dict[str, object]], None], expected: str
) -> None:
    payload = _gated_payload()
    mutate(payload)

    with pytest.raises(EvaluationError, match=expected):
        results_module.result_from_dict(payload)


def test_atomic_write_reports_unpreparable_output_path(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    with pytest.raises(EvaluationError, match="could not prepare output path"):
        results_module._atomic_write_text(blocker / "result.json", "{}")

    with pytest.raises(EvaluationError, match="could not be saved"):
        _result().save_json(blocker / "result.json")


def _two_retriever_payload() -> dict[str, object]:
    payload = _parsed_payload()
    payload["retrievers"]["bm25"] = json.loads(  # type: ignore[index]
        json.dumps(payload["retrievers"]["keyword"])  # type: ignore[index]
    )
    return payload


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda section: section.__setitem__("per_query", "q-1"),
            "per_query must be a list",
        ),
        (
            lambda section: section.__setitem__("per_query", []),
            "per_query must not be empty",
        ),
        (
            lambda section: section["per_query"].append(  # type: ignore[attr-defined]
                section["per_query"][0]  # type: ignore[index]
            ),
            "query IDs must be unique",
        ),
        (
            lambda section: section.__setitem__(
                "per_query",
                list(reversed(section["per_query"])),  # type: ignore[arg-type]
            ),
            "IDs/order differ",
        ),
        (
            lambda section: section["metrics"].pop("recall@2"),  # type: ignore[attr-defined]
            "metrics shape differs",
        ),
    ],
)
def test_result_from_dict_rejects_inconsistent_second_retriever(
    mutate: Callable[[dict[str, object]], None], expected: str
) -> None:
    payload = _two_retriever_payload()
    assert results_module.result_from_dict(_two_retriever_payload()).run_id

    mutate(payload["retrievers"]["bm25"])  # type: ignore[index,arg-type]

    with pytest.raises(EvaluationError, match=expected):
        results_module.result_from_dict(payload)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"retrieved_ids_by_cutoff": [(2, ("doc-a",))]},
            "retrieved_ids_by_cutoff must be a mapping",
        ),
        (
            {"retrieved_ids_by_cutoff": {2: 5, 3: ("doc-a", "doc-b")}},
            r"retrieved_ids_by_cutoff\[2\] must be a sequence",
        ),
        (
            {"search_latency_ms": "fast"},
            "search_latency_ms must be a finite non-negative number",
        ),
        (
            {"search_latency_ms": -1.0},
            "search_latency_ms must be a finite non-negative number",
        ),
        (
            {"search_latency_ms": float("inf")},
            "search_latency_ms must be a finite non-negative number",
        ),
        ({"warnings": 5}, "warnings must be a sequence"),
        (
            {"warnings": ["ok", " "]},
            "warnings must contain non-empty strings",
        ),
    ],
)
def test_query_evaluation_rejects_invalid_optional_fields(
    overrides: dict[str, object], expected: str
) -> None:
    with pytest.raises(EvaluationError, match=expected):
        QueryEvaluation(
            query_id="q-1",
            retrieved_ids=("doc-a", "doc-b"),
            metrics_by_cutoff={2: {"recall": 0.0}, 3: {"recall": 1.0}},
            **overrides,  # type: ignore[arg-type]
        )


def test_query_evaluation_serializes_warnings_without_latency() -> None:
    query = QueryEvaluation(
        query_id="q-1",
        retrieved_ids=("doc-a",),
        metrics_by_cutoff={1: {"recall": 1.0}},
        warnings=("careful",),
    )

    payload = query.to_dict()

    assert payload["warnings"] == ["careful"]
    assert "search_latency_ms" not in payload


def _base_kwargs() -> dict[str, object]:
    return {
        "run_id": "run-日本語",
        "metrics": {"keyword": _retriever_metrics()},
        "query_results": {"keyword": [_query_evaluation()]},
        "manifest": {"dataset": "日本語データ", "seed": 42},
    }


def _latency_stats() -> LatencyStats:
    return LatencyStats(
        mean_ms=2.0,
        p50_ms=2.0,
        p95_ms=2.0,
        max_ms=2.0,
        sample_count=1,
        failure_count=0,
    )


def _timed_kwargs() -> dict[str, object]:
    kwargs = _base_kwargs()
    kwargs["query_results"] = {
        "keyword": [
            QueryEvaluation(
                query_id="q-1",
                retrieved_ids=("doc-2", "doc-1"),
                metrics_by_cutoff={
                    1: {"recall": 0.0, "hit_rate": 0.0},
                    2: {"recall": 1.0, "hit_rate": 1.0},
                },
                search_latency_ms=2.0,
            )
        ]
    }
    kwargs["latency"] = {"keyword": _latency_stats()}
    return kwargs


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {
                "latency": {"keyword": _latency_stats()},
                "latency_stats": {"keyword": _latency_stats()},
            },
            "provide only one of latency or latency_stats",
        ),
        ({"latency": {}}, "latency must not be empty"),
        ({"latency": {"keyword": object()}}, "must be LatencyStats"),
        (
            {"latency": {"bm25": _latency_stats()}},
            "must contain the same retriever names",
        ),
        ({"query_results": {}}, "query_results must not be empty"),
        ({"query_results": {"keyword": "q-1"}}, "must be a sequence"),
    ],
)
def test_result_rejects_invalid_latency_and_query_results(
    overrides: dict[str, object], expected: str
) -> None:
    kwargs = _timed_kwargs()
    kwargs.update(overrides)

    with pytest.raises(EvaluationError, match=expected):
        EvaluationResult(**kwargs)  # type: ignore[arg-type]


def test_result_latency_stats_alias_returns_latency() -> None:
    result = EvaluationResult(**_timed_kwargs())  # type: ignore[arg-type]

    assert result.latency_stats["keyword"].mean_ms == 2.0


def _make_gate(
    *,
    metric: str = "recall@1",
    actual: float = 0.5,
    candidate: str = "run-日本語",
    retriever: str = "keyword",
    gate_index: int = 0,
    constraint: str = "min_value",
    threshold: float = 0.4,
    reason: str = "candidate meets minimum",
) -> QualityGateResult:
    check = QualityGateCheck(
        retriever=retriever,
        metric=metric,
        constraint=constraint,  # type: ignore[arg-type]
        actual=actual,
        threshold=threshold,
        passed=True,
        reason=reason,
        candidate_run_id=candidate,
    )
    return QualityGateResult(
        retriever=retriever,
        metric=metric,
        checks=(check,),
        passed=True,
        candidate_run_id=candidate,
        gate_index=gate_index,
    )


@pytest.mark.parametrize(
    ("gates", "expected"),
    [
        ("keyword", "quality_gates must be a sequence"),
        ([object()], "must contain QualityGateResult"),
        ([_make_gate(candidate="other-run")], "candidate run ID differs"),
        (
            [_make_gate(), _make_gate(metric="hit_rate@1")],
            "must not contain duplicate gate indexes",
        ),
        (
            [_make_gate(retriever="bm25")],
            "quality gate references an unavailable retriever",
        ),
        (
            [
                _make_gate(
                    metric="latency_mean_ms",
                    actual=2.0,
                    constraint="max_value",
                    threshold=3.0,
                    reason="candidate meets maximum",
                )
            ],
            "quality gate references unavailable latency",
        ),
        ([_make_gate(metric="bogus")], "quality gate metric is invalid"),
        ([_make_gate(metric="@1")], "quality gate metric is invalid"),
        ([_make_gate(metric="recall@x")], "quality gate metric is invalid"),
        ([_make_gate(metric="recall@01")], "quality gate metric is invalid"),
        ([_make_gate(metric="recall@0")], "quality gate metric is invalid"),
        (
            [_make_gate(metric="recall@" + "9" * 5000)],
            "quality gate metric is invalid",
        ),
    ],
)
def test_result_rejects_invalid_quality_gates(gates: object, expected: str) -> None:
    kwargs = _base_kwargs()
    kwargs["quality_gates"] = gates

    with pytest.raises(EvaluationError, match=expected):
        EvaluationResult(**kwargs)  # type: ignore[arg-type]


def test_result_accepts_latency_quality_gate() -> None:
    kwargs = _timed_kwargs()
    kwargs["quality_gates"] = [
        _make_gate(
            metric="latency_mean_ms",
            actual=2.0,
            constraint="max_value",
            threshold=3.0,
            reason="candidate meets maximum",
        )
    ]

    result = EvaluationResult(**kwargs)  # type: ignore[arg-type]

    assert result.quality_gates[0].metric == "latency_mean_ms"


def test_with_quality_gates_accepts_sequence_and_report() -> None:
    result = _result()
    gate = _make_gate()

    updated = result.with_quality_gates([gate])
    assert updated.quality_gates == (gate,)
    assert result.quality_gates == ()

    report = QualityGateReport(
        candidate_run_id="run-日本語",
        baseline_run_id=None,
        results=(gate,),
    )
    assert result.with_quality_gates(report).quality_gates == (gate,)


def test_with_quality_gates_rejects_mismatched_report() -> None:
    report = QualityGateReport(
        candidate_run_id="other-run",
        baseline_run_id=None,
        results=(_make_gate(candidate="other-run"),),
    )

    with pytest.raises(
        EvaluationError, match="quality gate report candidate run ID differs"
    ):
        _result().with_quality_gates(report)


def test_with_quality_gates_rejects_invalid_type() -> None:
    with pytest.raises(
        EvaluationError, match="must be a QualityGateReport or a sequence"
    ):
        _result().with_quality_gates("gates")  # type: ignore[arg-type]


def test_result_to_json_wraps_unserializable_state() -> None:
    result = _result()
    object.__setattr__(result, "manifest", {"callback": object()})

    with pytest.raises(EvaluationError, match="could not be serialized"):
        result.to_json()
