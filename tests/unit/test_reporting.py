from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import retrieval_lab.artifacts.results as result_artifacts
from retrieval_lab import (
    ComparabilityIssue,
    ComparabilityReport,
    EvaluationResult,
    LatencyStats,
    MetricComparison,
    MetricDelta,
    QueryEvaluation,
    RetrieverMetrics,
    RunComparison,
    compare_runs,
    load_result,
)
from retrieval_lab.exceptions import EvaluationError
from retrieval_lab.reporting import comparison_html
from retrieval_lab.reporting._safety import safe_identifier
from retrieval_lab.reporting.html import _config_view


def _query(query_id: str, value: float, *, timed: bool = True) -> QueryEvaluation:
    return QueryEvaluation(
        query_id=query_id,
        retrieved_ids=("doc-1",),
        metrics_by_cutoff={1: {"recall": value, "hit_rate": value}},
        search_latency_ms=2.5 if timed else None,
        warnings=("<script>alert(1)</script>",) if timed else (),
    )


def _result(*, timed: bool = True) -> EvaluationResult:
    queries = (_query("q-1", 1.0, timed=timed), _query("q-2", 0.0, timed=timed))
    return EvaluationResult(
        run_id="run-<unsafe>",
        metrics={"keyword": RetrieverMetrics({1: {"recall": 0.5, "hit_rate": 0.5}})},
        query_results={"keyword": queries},
        latency=(
            {
                "keyword": LatencyStats(
                    mean_ms=2.5,
                    p50_ms=2.5,
                    p95_ms=2.5,
                    max_ms=3.0,
                    sample_count=2,
                    warnings=("<img onerror=alert(1)>",),
                )
            }
            if timed
            else None
        ),
        manifest={
            "dataset_hash": "d" * 64,
            "relevance_level": "document",
            "top_k": [1],
            "metric_version": 1,
            "config": {
                "experiment": {"seed": 42, "token": "do-not-show"},
                "corpus": {
                    "path": "/private/absolute/path",
                    "chunker": {
                        "type": "recursive_characters",
                        "size": 32,
                        "overlap": 4,
                    },
                },
                "dataset": {
                    "path": "/private/eval.jsonl",
                    "relevance_level": "document",
                },
                "retrievers": [
                    {
                        "name": "keyword",
                        "type": "keyword",
                        "model": "do-not-show",
                        "normalize_embeddings": True,
                    }
                ],
                "evaluation": {"top_k": [1], "metrics": ["recall"]},
            },
        },
    )


def test_result_json_round_trip_and_public_loader(tmp_path: Path) -> None:
    result = _result()
    loaded = EvaluationResult.from_json(result.to_json())
    assert loaded.to_json() == result.to_json()

    path = tmp_path / "nested" / "result.json"
    result.save_json(path)
    assert EvaluationResult.load_json(path).to_dict() == result.to_dict()
    assert load_result(path).run_id == result.run_id


def test_legacy_result_without_latency_round_trips() -> None:
    result = _result(timed=False)
    loaded = EvaluationResult.from_dict(result.to_dict())
    assert loaded.latency == {}
    assert loaded.query_results["keyword"][0].search_latency_ms is None
    assert loaded.to_dict() == result.to_dict()


def test_loader_allows_additive_unknown_fields() -> None:
    payload = _result().to_dict()
    payload["future"] = {"new": True}  # type: ignore[typeddict-item]
    payload["run"]["future_run"] = 1  # type: ignore[index]
    payload["retrievers"]["keyword"]["future_retriever"] = "ok"  # type: ignore[index]
    payload["retrievers"]["keyword"]["per_query"][0]["future_query"] = 2  # type: ignore[index]
    assert EvaluationResult.from_dict(payload).run_id == "run-<unsafe>"


def test_loader_rejects_duplicate_keys_at_nested_level() -> None:
    text = (
        _result()
        .to_json()
        .replace('"run": {', '"run": {"id": "first", "id": "second",', 1)
    )
    with pytest.raises(EvaluationError, match="duplicate JSON key"):
        EvaluationResult.from_json(text)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_loader_rejects_non_finite_numbers(value: float) -> None:
    payload = _result().to_dict()
    payload["retrievers"]["keyword"]["metrics"]["recall@1"] = value  # type: ignore[index]
    text = json.dumps(payload, allow_nan=True)
    with pytest.raises(EvaluationError, match=r"non-finite|invalid"):
        EvaluationResult.from_json(text)


@pytest.mark.parametrize(
    "key",
    ["recall@@1", "recall@01", "recall@1.0", "@1", "recall@0"],
)
def test_loader_rejects_non_canonical_metric_keys(key: str) -> None:
    payload = _result().to_dict()
    metrics = payload["retrievers"]["keyword"]["metrics"]  # type: ignore[index]
    value = metrics.pop("recall@1")
    metrics[key] = value
    with pytest.raises(EvaluationError, match=r"metric|cutoff"):
        EvaluationResult.from_dict(payload)


def test_loader_rejects_metric_shape_and_aggregate_mismatches() -> None:
    payload = _result().to_dict()
    payload["retrievers"]["keyword"]["metrics"]["precision@1"] = 0.5  # type: ignore[index]
    with pytest.raises(EvaluationError, match="shape"):
        EvaluationResult.from_dict(payload)

    payload = _result().to_dict()
    payload["retrievers"]["keyword"]["metrics"]["recall@1"] = 0.7  # type: ignore[index]
    with pytest.raises(EvaluationError, match="aggregate"):
        EvaluationResult.from_dict(payload)


def test_loader_rejects_partial_latency_and_nonempty_quality_gates() -> None:
    payload = _result().to_dict()
    payload["quality_gates"] = [{"metric": "recall"}]
    with pytest.raises(EvaluationError, match="quality_gates"):
        EvaluationResult.from_dict(payload)

    payload = _result().to_dict()
    del payload["retrievers"]["keyword"]["latency"]  # type: ignore[index]
    payload["retrievers"]["other"] = copy.deepcopy(
        _result().to_dict()["retrievers"]["keyword"]
    )  # type: ignore[index]
    with pytest.raises(EvaluationError, match=r"retriever|latency"):
        EvaluationResult.from_dict(payload)


def test_loader_rejects_broken_utf8_and_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff")
    with pytest.raises(EvaluationError, match="read"):
        EvaluationResult.load_json(path)

    path.write_text(_result().to_json(), encoding="utf-8")
    with pytest.raises(EvaluationError, match="max_bytes"):
        EvaluationResult.load_json(path, max_bytes=1)
    with pytest.raises(EvaluationError, match="max_bytes"):
        EvaluationResult.load_json(path, max_bytes=True)  # type: ignore[arg-type]


def test_loader_reports_root_and_field_type_errors() -> None:
    with pytest.raises(EvaluationError, match="root must be an object"):
        EvaluationResult.from_json("[]")
    with pytest.raises(EvaluationError, match="required fields"):
        EvaluationResult.from_dict({"schema_version": 1})
    with pytest.raises(EvaluationError, match="schema_version"):
        EvaluationResult.from_dict(
            {
                "schema_version": True,
                "run": {},
                "retrievers": {},
                "quality_gates": [],
            }
        )
    with pytest.raises(EvaluationError, match="JSON text"):
        EvaluationResult.from_json(1)  # type: ignore[arg-type]


def test_html_handles_legacy_result_without_config_or_latency() -> None:
    rendered = _result(timed=False).to_html()
    assert "Data unavailable" in rendered
    assert "Configuration" in rendered


def test_csv_is_deterministic_safe_and_excludes_retrieved_ids() -> None:
    result = _result()
    summary = result.to_summary_csv()
    per_query = result.to_per_query_csv()
    assert summary.startswith("retriever,metric,cutoff,value,warnings\n")
    assert "latency_mean_ms" in summary and "latency_failure_count" in summary
    assert per_query.startswith(
        "retriever,query_id,metric,cutoff,value,search_latency_ms,warnings\n"
    )
    assert "doc-1" not in per_query
    assert "<script>" in per_query
    assert "'=" not in per_query  # warnings are JSON arrays, not formulas


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@", "\t", "\r", "\n"])
def test_csv_prefixes_formula_like_identifier_cells(prefix: str) -> None:
    retriever_name = f"{prefix}retriever"
    query_id = f"{prefix}query"
    query = QueryEvaluation(
        query_id=query_id,
        retrieved_ids=("doc",),
        metrics_by_cutoff={1: {"recall": 1.0}},
    )
    result = EvaluationResult(
        run_id="run",
        metrics={retriever_name: RetrieverMetrics({1: {"recall": 1.0}})},
        query_results={retriever_name: (query,)},
    )
    assert f"'{retriever_name}" in result.to_summary_csv()
    assert f"'{retriever_name}" in result.to_per_query_csv()
    assert f"'{query_id}" in result.to_per_query_csv()


def test_reports_redact_absolute_path_identifiers() -> None:
    query = QueryEvaluation(
        query_id=r"C:\private\query",
        retrieved_ids=("doc",),
        metrics_by_cutoff={1: {"recall": 1.0}},
    )
    result = EvaluationResult(
        run_id="/private/run",
        metrics={"/private/retriever": RetrieverMetrics({1: {"recall": 1.0}})},
        query_results={"/private/retriever": (query,)},
    )
    for report in (
        result.summary(),
        result.to_summary_csv(),
        result.to_per_query_csv(),
        result.to_html(),
    ):
        assert "/private/run" not in report
        assert "/private/retriever" not in report
        assert r"C:\private\query" not in report
        assert "[redacted path]" in report


def test_path_redaction_is_distinct_and_does_not_collide_with_placeholder() -> None:
    left = EvaluationResult(
        run_id="/private/left",
        metrics={"/private/retriever-left": RetrieverMetrics({1: {"recall": 1.0}})},
        query_results={
            "/private/retriever-left": (
                QueryEvaluation(
                    query_id="q-left",
                    retrieved_ids=("doc",),
                    metrics_by_cutoff={1: {"recall": 1.0}},
                ),
            )
        },
    )
    right = EvaluationResult(
        run_id="/private/right",
        metrics={"/private/retriever-right": RetrieverMetrics({1: {"recall": 1.0}})},
        query_results={
            "/private/retriever-right": (
                QueryEvaluation(
                    query_id="q-right",
                    retrieved_ids=("doc",),
                    metrics_by_cutoff={1: {"recall": 1.0}},
                ),
            )
        },
    )
    left_report = left.to_html()
    right_report = right.to_html()

    assert left_report != right_report
    left_token = left_report.split("[redacted path]#", 1)[1].split("]", 1)[0]
    right_token = right_report.split("[redacted path]#", 1)[1].split("]", 1)[0]
    assert left_token != right_token
    assert "/private/left" not in left_report
    assert "/private/right" not in right_report
    assert "[redacted path]#" in left_report
    assert "[redacted path]#" in right_report
    assert safe_identifier("/private/left") != safe_identifier("[redacted path]")


def test_html_sanitizes_path_like_retriever_names_in_configuration() -> None:
    query = QueryEvaluation(
        query_id="q",
        retrieved_ids=("doc",),
        metrics_by_cutoff={1: {"recall": 1.0}},
    )
    result = EvaluationResult(
        run_id="run",
        metrics={"/private/retriever": RetrieverMetrics({1: {"recall": 1.0}})},
        query_results={"/private/retriever": (query,)},
        manifest={
            "config": {"retrievers": [{"name": "/private/retriever", "type": "custom"}]}
        },
    )

    rendered = result.to_html()
    assert "/private/retriever" not in rendered
    assert "[redacted path]#" in rendered


def test_html_recommendations_use_recall_cutoffs_and_ignore_empty_latency() -> None:
    query = QueryEvaluation(
        query_id="q",
        retrieved_ids=("doc",),
        metrics_by_cutoff={1: {"recall": 0.5}, 5: {"precision": 1.0}},
    )
    result = EvaluationResult(
        run_id="run",
        metrics={
            "fast-but-empty": RetrieverMetrics(
                {1: {"recall": 0.4}, 5: {"precision": 1.0}}
            )
        },
        query_results={"fast-but-empty": (query,)},
        latency={
            "fast-but-empty": LatencyStats(
                mean_ms=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                max_ms=0.0,
                sample_count=0,
            )
        },
    )

    rendered = result.to_html()
    assert "Recall@1" in rendered
    assert "Recall@5" not in rendered
    assert "<strong>Latency:</strong> Data unavailable" in rendered
    assert "<strong>Balanced:</strong> Data unavailable" in rendered


def test_config_view_skips_malformed_sections_and_entries() -> None:
    manifest = {
        "config": {
            "experiment": {"seed": "42"},
            "corpus": {"chunker": {"unknown": 1}},
            "dataset": {"relevance_level": 3},
            "retrievers": [
                "keyword",
                {"model": "not-a-displayed-key"},
                {"name": "bm25", "type": "bm25", "k1": 1.5},
            ],
            "evaluation": {"unknown": 1},
        }
    }

    assert _config_view(manifest) == {
        "retrievers": [{"name": "bm25", "type": "bm25", "k1": 1.5}]
    }
    assert _config_view({"config": {"retrievers": [{"model": "x"}]}}) == {}
    assert _config_view({"config": {"corpus": {"chunker": "not-a-mapping"}}}) == {}
    assert _config_view({"config": {"retrievers": {"name": "bm25"}}}) == {}


def test_html_recommendations_skip_retrievers_missing_the_recall_cutoff() -> None:
    def query(metrics: dict[int, dict[str, float]]) -> QueryEvaluation:
        return QueryEvaluation(
            query_id="q",
            retrieved_ids=("doc",),
            metrics_by_cutoff=metrics,
        )

    result = EvaluationResult(
        run_id="run",
        metrics={
            "wide": RetrieverMetrics({3: {"recall": 0.5}}),
            "narrow": RetrieverMetrics({1: {"recall": 0.9}}),
        },
        query_results={
            "wide": (query({3: {"recall": 0.5}}),),
            "narrow": (query({1: {"recall": 0.9}}),),
        },
    )

    rendered = result.to_html()

    assert "<strong>Quality:</strong> wide (Recall@3)" in rendered


def test_html_recommendations_report_unavailable_balanced_pick() -> None:
    def query(metrics: dict[int, dict[str, float]]) -> QueryEvaluation:
        return QueryEvaluation(
            query_id="q",
            retrieved_ids=("doc",),
            metrics_by_cutoff=metrics,
        )

    result = EvaluationResult(
        run_id="run",
        metrics={
            "quality-only": RetrieverMetrics({1: {"recall": 0.5}}),
            "latency-only": RetrieverMetrics({1: {"precision": 0.5}}),
        },
        query_results={
            "quality-only": (query({1: {"recall": 0.5}}),),
            "latency-only": (query({1: {"precision": 0.5}}),),
        },
        latency={
            "quality-only": LatencyStats(
                mean_ms=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                max_ms=0.0,
                sample_count=0,
            ),
            "latency-only": LatencyStats(
                mean_ms=5.0,
                p50_ms=5.0,
                p95_ms=5.0,
                max_ms=5.0,
                sample_count=3,
            ),
        },
    )

    rendered = result.to_html()

    assert "<strong>Quality:</strong> quality-only (Recall@1)" in rendered
    assert "<strong>Latency:</strong> latency-only (p95)" in rendered
    assert "<strong>Balanced:</strong> Data unavailable" in rendered


def test_html_recommendations_report_unavailable_quality_without_recall() -> None:
    query = QueryEvaluation(
        query_id="q",
        retrieved_ids=("doc",),
        metrics_by_cutoff={1: {"precision": 0.5}},
    )
    result = EvaluationResult(
        run_id="run",
        metrics={"bm25": RetrieverMetrics({1: {"precision": 0.5}})},
        query_results={"bm25": (query,)},
    )

    rendered = result.to_html()

    assert "<strong>Quality:</strong> Data unavailable" in rendered


def test_save_csv_rejects_invalid_output_path() -> None:
    with pytest.raises(EvaluationError, match="valid path"):
        _result().save_csv(object())  # type: ignore[arg-type]


def test_html_is_standalone_escaped_and_redacted() -> None:
    rendered = _result().to_html()
    assert "&lt;unsafe&gt;" in rendered
    assert "&lt;script&gt;" in rendered
    assert "<script" not in rendered
    assert "&lt;img onerror=alert(1)&gt;" in rendered
    assert "http://" not in rendered and "https://" not in rendered
    assert "do-not-show" not in rendered
    assert "/private/absolute/path" not in rendered
    assert "doc-1" not in rendered
    assert "Quality:" in rendered and "Latency:" in rendered and "Balanced:" in rendered
    assert "Metrics and latency" in rendered


def test_summary_does_not_print(capsys: pytest.CaptureFixture[str]) -> None:
    text = _result().summary()
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert "run_id: run-<unsafe>" in text
    assert "latency" in text and "recall@1" in text


def test_all_report_saves_are_atomic_and_create_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    output_dir = tmp_path / "reports"
    paths = result.save_csv(output_dir)
    result.save_html(output_dir / "report.html")
    assert all(path.exists() for path in paths)
    assert (output_dir / "report.html").exists()
    assert not list(output_dir.glob(".*.tmp"))

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(result_artifacts.os, "replace", fail_replace)
    with pytest.raises(EvaluationError, match="saved"):
        result.save_json(tmp_path / "atomic" / "result.json")
    assert not list((tmp_path / "atomic").glob(".*.tmp"))


def _diff_result(run_id: str, values: dict[str, float]) -> EvaluationResult:
    queries = tuple(
        QueryEvaluation(
            query_id=query_id,
            retrieved_ids=("doc-1",),
            metrics_by_cutoff={1: {"recall": value}},
        )
        for query_id, value in values.items()
    )
    return EvaluationResult(
        run_id=run_id,
        metrics={
            "keyword": RetrieverMetrics(
                {1: {"recall": sum(values.values()) / len(values)}}
            )
        },
        query_results={"keyword": queries},
        latency=None,
        manifest={
            "dataset_hash": "d" * 64,
            "query_ids": sorted(values),
            "relevance_level": "document",
            "top_k": [1],
            "metric_version": 1,
        },
    )


def test_comparison_html_shows_hand_calculated_deltas() -> None:
    baseline = _diff_result("baseline-run", {"q-1": 1.0, "q-2": 0.0})
    candidate = _diff_result("candidate-run", {"q-1": 1.0, "q-2": 1.0})
    comparison = compare_runs(baseline, candidate)

    rendered = comparison_html(comparison)

    assert rendered == comparison.to_html()
    assert "Baseline run ID: baseline-run" in rendered
    assert "Candidate run ID: candidate-run" in rendered
    assert "Comparable: yes" in rendered
    assert (
        "<td>run_id</td><td>experimental variable differs</td><td>no</td>" in rendered
    )
    assert (
        "<tr class='improved'><td>keyword</td><td>recall</td><td>1</td>"
        "<td>0.5</td><td>1</td><td>+0.5</td><td>+100%</td>"
        "<td>improved</td></tr>"
    ) in rendered
    improved_row = (
        "<tr class='improved'><td>keyword</td><td>q-2</td><td>recall</td>"
        "<td>1</td><td>0</td><td>1</td><td>+1</td><td></td>"
        "<td>improved</td></tr>"
    )
    unchanged_row = (
        "<tr class='unchanged'><td>keyword</td><td>q-1</td><td>recall</td>"
        "<td>1</td><td>1</td><td>1</td><td>+0</td><td>+0%</td>"
        "<td>unchanged</td></tr>"
    )
    assert improved_row in rendered
    assert unchanged_row in rendered
    assert rendered.index(improved_row) < rendered.index(unchanged_row)


def test_comparison_html_escapes_run_ids() -> None:
    baseline = _diff_result("base-<script>", {"q-1": 0.0})
    candidate = _diff_result("cand-<script>", {"q-1": 1.0})

    rendered = compare_runs(baseline, candidate).to_html()

    assert "base-&lt;script&gt;" in rendered
    assert "cand-&lt;script&gt;" in rendered
    assert "base-<script>" not in rendered
    assert "cand-<script>" not in rendered


def test_comparison_html_reports_blocking_issues_without_values() -> None:
    comparison = RunComparison(
        baseline_run_id="base",
        candidate_run_id="cand",
        comparability=ComparabilityReport(
            issues=(
                ComparabilityIssue(
                    field="dataset_hash",
                    reason="values differ",
                    baseline_value="/private/base.json",
                    candidate_value="/private/cand.json",
                ),
            ),
            variable_differences=(
                ComparabilityIssue(
                    field="seed",
                    reason="experimental variable changed",
                    blocking=False,
                ),
            ),
        ),
        metrics={},
    )

    rendered = comparison_html(comparison)

    assert "Comparable: no" in rendered
    assert "<td>dataset_hash</td><td>values differ</td><td>yes</td>" in rendered
    assert "<td>seed</td><td>experimental variable changed</td><td>no</td>" in rendered
    assert "/private/base.json" not in rendered
    assert "/private/cand.json" not in rendered
    assert "<p>No metric changes.</p>" in rendered
    assert "<p>No per-query metric changes.</p>" in rendered


def test_comparison_save_html_round_trips(tmp_path: Path) -> None:
    baseline = _diff_result("baseline-run", {"q-1": 0.0})
    candidate = _diff_result("candidate-run", {"q-1": 1.0})
    comparison = compare_runs(baseline, candidate)

    path = tmp_path / "nested" / "diff.html"
    comparison.save_html(path)

    assert path.read_text(encoding="utf-8") == comparison.to_html()


def _delta(**overrides: object) -> MetricDelta:
    values: dict[str, object] = {
        "retriever": "keyword",
        "metric": "recall",
        "cutoff": 1,
        "baseline": 1.0,
        "candidate": 0.5,
        "absolute_delta": -0.5,
        "relative_delta": -0.5,
        "relative_status": "defined",
        "classification": "regressed",
        "direction": "higher_is_better",
    }
    values.update(overrides)
    return MetricDelta(**values)


def _hand_comparison(
    aggregate: MetricDelta,
    query_deltas: tuple[MetricDelta, ...] = (),
    **report_overrides: object,
) -> RunComparison:
    return RunComparison(
        baseline_run_id="base",
        candidate_run_id="cand",
        comparability=ComparabilityReport(**report_overrides),
        metrics={
            aggregate.retriever: (
                MetricComparison(
                    retriever=aggregate.retriever,
                    metric=aggregate.metric,
                    cutoff=aggregate.cutoff,
                    aggregate=aggregate,
                    query_deltas=query_deltas,
                ),
            )
        },
    )


def test_comparison_html_shows_regressed_rows() -> None:
    baseline = _diff_result("baseline-run", {"q-1": 1.0})
    candidate = _diff_result("candidate-run", {"q-1": 0.0})

    rendered = compare_runs(baseline, candidate).to_html()

    assert (
        "<tr class='regressed'><td>keyword</td><td>recall</td><td>1</td>"
        "<td>1</td><td>0</td><td>-1</td><td>-100%</td>"
        "<td>regressed</td></tr>"
    ) in rendered
    assert (
        "<tr class='regressed'><td>keyword</td><td>q-1</td><td>recall</td>"
        "<td>1</td><td>1</td><td>0</td><td>-1</td><td>-100%</td>"
        "<td>regressed</td></tr>"
    ) in rendered


def test_comparison_html_renders_none_cutoff() -> None:
    aggregate = _delta(cutoff=None)
    query = _delta(cutoff=None, query_id="q-1")

    rendered = comparison_html(_hand_comparison(aggregate, (query,)))

    assert (
        "<tr class='regressed'><td>keyword</td><td>recall</td><td></td>"
        "<td>1</td><td>0.5</td><td>-0.5</td><td>-50%</td>"
        "<td>regressed</td></tr>"
    ) in rendered
    assert (
        "<tr class='regressed'><td>keyword</td><td>q-1</td><td>recall</td>"
        "<td></td><td>1</td><td>0.5</td><td>-0.5</td><td>-50%</td>"
        "<td>regressed</td></tr>"
    ) in rendered


def test_comparison_html_normalizes_negative_zero() -> None:
    aggregate = _delta(
        baseline=-0.0,
        candidate=0.0,
        absolute_delta=-0.0,
        relative_delta=-0.0,
        classification="unchanged",
    )

    rendered = comparison_html(_hand_comparison(aggregate))

    assert "<td>0</td><td>0</td><td>+0</td><td>+0%</td>" in rendered
    assert "-0" not in rendered


def test_comparison_html_renders_diagnostics() -> None:
    comparison = _hand_comparison(
        _delta(),
        diagnostics=(
            ComparabilityIssue(
                field="latency",
                reason="presence differs",
                blocking=False,
            ),
        ),
    )

    rendered = comparison_html(comparison)

    assert "<td>latency</td><td>presence differs</td><td>no</td>" in rendered


def test_comparison_html_escapes_all_text_fields() -> None:
    aggregate = _delta(retriever="ret-<script>", metric="met-<script>")
    query = _delta(
        retriever="ret-<script>",
        metric="met-<script>",
        query_id="q-<script>",
    )
    comparison = _hand_comparison(
        aggregate,
        (query,),
        issues=(
            ComparabilityIssue(
                field="fld-<script>",
                reason="rsn-<script>",
            ),
        ),
    )

    rendered = comparison_html(comparison)

    for raw in (
        "ret-<script>",
        "met-<script>",
        "q-<script>",
        "fld-<script>",
        "rsn-<script>",
    ):
        assert raw not in rendered
    for escaped in (
        "ret-&lt;script&gt;",
        "met-&lt;script&gt;",
        "q-&lt;script&gt;",
        "fld-&lt;script&gt;",
        "rsn-&lt;script&gt;",
    ):
        assert escaped in rendered


def test_comparison_html_sorts_retrievers() -> None:
    zeta = MetricComparison(
        retriever="zeta",
        metric="recall",
        cutoff=1,
        aggregate=_delta(retriever="zeta"),
    )
    alpha = MetricComparison(
        retriever="alpha",
        metric="recall",
        cutoff=1,
        aggregate=_delta(retriever="alpha"),
    )
    comparison = RunComparison(
        baseline_run_id="base",
        candidate_run_id="cand",
        comparability=ComparabilityReport(),
        metrics={"zeta": (zeta,), "alpha": (alpha,)},
    )

    rendered = comparison_html(comparison)

    assert rendered.index("<td>alpha</td>") < rendered.index("<td>zeta</td>")


def test_comparison_save_html_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _hand_comparison(_delta())

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(result_artifacts.os, "replace", fail_replace)
    with pytest.raises(EvaluationError, match="atomically"):
        comparison.save_html(tmp_path / "atomic" / "diff.html")
    assert not list((tmp_path / "atomic").glob(".*.tmp"))
