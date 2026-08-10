"""Plain-text deterministic result summaries."""

from __future__ import annotations

from retrieval_lab.domain.results import EvaluationResult
from retrieval_lab.reporting._safety import safe_identifier


def result_summary(result: EvaluationResult) -> str:
    """Return all aggregate metrics and latency observations without printing."""

    lines = [f"run_id: {safe_identifier(result.run_id)}"]
    for name in sorted(result.metrics):
        lines.append(f"retriever: {safe_identifier(name)}")
        for cutoff in sorted(result.metrics[name].metrics_by_cutoff):
            for metric in sorted(result.metrics[name].metrics_by_cutoff[cutoff]):
                value = result.metrics[name].metrics_by_cutoff[cutoff][metric]
                lines.append(f"  {metric}@{cutoff}: {value:.12g}")
        if result.latency:
            stats = result.latency[name]
            lines.append(
                "  latency: "
                f"mean_ms={stats.mean_ms:.12g}, "
                f"p50_ms={stats.p50_ms:.12g}, "
                f"p95_ms={stats.p95_ms:.12g}, "
                f"max_ms={stats.max_ms:.12g}, "
                f"sample_count={stats.sample_count}, "
                f"failure_count={stats.failure_count}"
            )
            for warning in stats.warnings:
                lines.append(f"  warning: {warning}")
        for query in result.query_results[name]:
            for warning in query.warnings:
                lines.append(
                    f"  query_warning[{safe_identifier(query.query_id)}]: {warning}"
                )
    return "\n".join(lines) + "\n"


__all__ = ["result_summary"]
