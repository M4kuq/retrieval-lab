"""Deterministic, formula-safe CSV reports."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable

from retrieval_lab.domain.results import EvaluationResult
from retrieval_lab.reporting._safety import safe_identifier

_SUMMARY_FIELDS = ("retriever", "metric", "cutoff", "value", "warnings")
_QUERY_FIELDS = (
    "retriever",
    "query_id",
    "metric",
    "cutoff",
    "value",
    "search_latency_ms",
    "warnings",
)
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")


def _cell(value: object) -> object:
    if not isinstance(value, str) or not value.startswith(_FORMULA_PREFIXES):
        return value
    return "'" + value


def _write(fields: tuple[str, ...], rows: Iterable[tuple[object, ...]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(fields)
    for row in rows:
        writer.writerow([_cell(value) for value in row])
    return stream.getvalue()


def summary_csv(result: EvaluationResult) -> str:
    """Render aggregate metrics and latency statistics in long form."""

    rows: list[tuple[object, ...]] = []
    for name in sorted(result.metrics):
        metrics = result.metrics[name].metrics_by_cutoff
        for cutoff in sorted(metrics):
            for metric in sorted(metrics[cutoff]):
                rows.append(
                    (
                        safe_identifier(name),
                        metric,
                        cutoff,
                        metrics[cutoff][metric],
                        "",
                    )
                )
        if result.latency:
            stats = result.latency[name]
            values: tuple[tuple[str, float | int], ...] = (
                ("latency_mean_ms", stats.mean_ms),
                ("latency_p50_ms", stats.p50_ms),
                ("latency_p95_ms", stats.p95_ms),
                ("latency_max_ms", stats.max_ms),
                ("latency_sample_count", stats.sample_count),
                ("latency_failure_count", stats.failure_count),
            )
            for metric, value in values:
                rows.append((safe_identifier(name), metric, "", value, ""))
            for warning in stats.warnings:
                rows.append((safe_identifier(name), "latency_warning", "", "", warning))
    return _write(_SUMMARY_FIELDS, rows)


def per_query_csv(result: EvaluationResult) -> str:
    """Render per-query metrics without query or document text."""

    rows: list[tuple[object, ...]] = []
    for name in sorted(result.query_results):
        for query in result.query_results[name]:
            warnings = json.dumps(
                list(query.warnings), ensure_ascii=False, separators=(",", ":")
            )
            latency = query.search_latency_ms
            for cutoff in sorted(query.metrics_by_cutoff):
                for metric in sorted(query.metrics_by_cutoff[cutoff]):
                    rows.append(
                        (
                            safe_identifier(name),
                            safe_identifier(query.query_id),
                            metric,
                            cutoff,
                            query.metrics_by_cutoff[cutoff][metric],
                            latency if latency is not None else "",
                            warnings,
                        )
                    )
    return _write(_QUERY_FIELDS, rows)


__all__ = ["per_query_csv", "summary_csv"]
