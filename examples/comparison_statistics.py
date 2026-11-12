"""Compare two runs, bootstrap a metric delta, and save a diff report.

The baseline ranking puts each query's relevant document second; the
candidate puts it first. compare_runs() turns that into per-metric
deltas, bootstrap_metric_delta() attaches a confidence interval and
p-value to the recall@1 delta, and the comparison is written out as a
standalone HTML diff report.
"""

import tempfile
from pathlib import Path

from retrieval_lab import (
    EvaluationDataset,
    EvaluationQuery,
    RetrievedQueryResult,
    bootstrap_metric_delta,
    compare_runs,
    evaluate_results,
)

QUERIES = ("q1", "q2", "q3", "q4")

dataset = EvaluationDataset(
    queries=tuple(
        EvaluationQuery(query_id, f"text of {query_id}", frozenset({f"d-{query_id}"}))
        for query_id in QUERIES
    )
)

baseline = evaluate_results(
    dataset=dataset,
    retrieved_results=[
        RetrievedQueryResult(query_id, ("distractor", f"d-{query_id}"))
        for query_id in QUERIES
    ],
    top_k=(1, 3),
    name="keyword",
)
candidate = evaluate_results(
    dataset=dataset,
    retrieved_results=[
        RetrievedQueryResult(query_id, (f"d-{query_id}", "distractor"))
        for query_id in QUERIES
    ],
    top_k=(1, 3),
    name="keyword",
)

comparison = compare_runs(baseline, candidate)
recall_at_1 = next(
    metric_comparison
    for metric_comparison in comparison.metrics["keyword"]
    if metric_comparison.metric == "recall" and metric_comparison.cutoff == 1
)
assert recall_at_1.aggregate.absolute_delta == 1.0

bootstrap = bootstrap_metric_delta(recall_at_1)
assert bootstrap.observed_delta == 1.0
assert (bootstrap.ci_lower, bootstrap.ci_upper) == (1.0, 1.0)
assert bootstrap.significant is True

with tempfile.TemporaryDirectory() as directory:
    report_path = Path(directory) / "diff.html"
    comparison.save_html(report_path)
    report = report_path.read_text(encoding="utf-8")
assert report.startswith("<!doctype html>")

print("recall@1 delta:", bootstrap.observed_delta)
print("significant:", bootstrap.significant)
