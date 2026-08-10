"""Standalone escaped HTML reporting with no external resources."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping

from retrieval_lab.domain.results import EvaluationResult
from retrieval_lab.reporting._safety import safe_identifier


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _json(value: object) -> str:
    return _escape(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _config_view(manifest: Mapping[str, object]) -> dict[str, object]:
    raw = manifest.get("config")
    if not isinstance(raw, Mapping):
        return {}
    view: dict[str, object] = {}
    experiment = raw.get("experiment")
    if isinstance(experiment, Mapping) and isinstance(experiment.get("seed"), int):
        view["experiment"] = {"seed": experiment["seed"]}
    corpus = raw.get("corpus")
    if isinstance(corpus, Mapping):
        chunker = corpus.get("chunker")
        if isinstance(chunker, Mapping):
            selected = {
                key: chunker[key]
                for key in ("type", "size", "overlap")
                if isinstance(chunker.get(key), (str, int, float, bool))
            }
            if selected:
                view["corpus"] = {"chunker": selected}
    dataset = raw.get("dataset")
    if isinstance(dataset, Mapping) and isinstance(dataset.get("relevance_level"), str):
        view["dataset"] = {"relevance_level": dataset["relevance_level"]}
    retrievers = raw.get("retrievers")
    if isinstance(retrievers, list):
        selected_retrievers: list[dict[str, object]] = []
        for item in retrievers:
            if not isinstance(item, Mapping):
                continue
            retriever_view: dict[str, object] = {}
            for key in (
                "name",
                "type",
                "batch_size",
                "normalize_embeddings",
                "k1",
                "b",
                "rrf_k",
                "candidate_k",
            ):
                value = item.get(key)
                if isinstance(value, (str, int, float, bool)):
                    retriever_view[key] = (
                        safe_identifier(value)
                        if key == "name" and isinstance(value, str)
                        else value
                    )
            if retriever_view:
                selected_retrievers.append(retriever_view)
        if selected_retrievers:
            view["retrievers"] = selected_retrievers
    evaluation = raw.get("evaluation")
    if isinstance(evaluation, Mapping):
        selected_evaluation = {
            key: evaluation[key]
            for key in ("top_k", "metrics", "repetitions", "concurrency")
            if isinstance(evaluation.get(key), (list, int, float, bool))
        }
        if selected_evaluation:
            view["evaluation"] = selected_evaluation
    return view


def _metric_value(
    result: EvaluationResult, name: str, metric: str, cutoff: int
) -> float | None:
    values = result.metrics[name].metrics_by_cutoff.get(cutoff)
    if values is None:
        return None
    value = values.get(metric)
    return None if value is None else float(value)


def _recommendations(result: EvaluationResult) -> str:
    names = sorted(result.metrics)
    recall_cutoffs = [
        cutoff
        for name in names
        for cutoff, values in result.metrics[name].metrics_by_cutoff.items()
        if "recall" in values
    ]
    cutoff = max(recall_cutoffs, default=0)
    quality: dict[str, float] = {}
    if cutoff:
        for name in names:
            value = _metric_value(result, name, "recall", cutoff)
            if value is not None:
                quality[name] = value
    quality_line = (
        "Data unavailable"
        if not quality
        else f"{safe_identifier(max(quality, key=lambda name: (quality[name], name)))} "
        f"(Recall@{cutoff})"
    )
    latency = (
        {
            name: result.latency[name].p95_ms
            for name in names
            if result.latency[name].sample_count > 0
        }
        if result.latency
        else {}
    )
    latency_line = (
        "Data unavailable"
        if not latency
        else f"{safe_identifier(min(latency, key=lambda name: (latency[name], name)))} "
        "(p95)"
    )
    balanced_line = "Data unavailable"
    if quality and latency:
        quality_order = {
            name: rank
            for rank, name in enumerate(
                sorted(quality, key=lambda item: quality[item], reverse=True)
            )
        }
        latency_order = {
            name: rank
            for rank, name in enumerate(sorted(latency, key=lambda item: latency[item]))
        }
        common = set(quality) & set(latency)
        if common:
            balanced = min(
                common,
                key=lambda name: (quality_order[name] + latency_order[name], name),
            )
            balanced_line = safe_identifier(balanced)
    return (
        "<section><h2>Recommendations</h2>"
        f"<p><strong>Quality:</strong> {_escape(quality_line)}</p>"
        f"<p><strong>Latency:</strong> {_escape(latency_line)}</p>"
        f"<p><strong>Balanced:</strong> {_escape(balanced_line)}</p></section>"
    )


def result_html(result: EvaluationResult) -> str:
    """Render metrics, latency, warnings, and safe config fields only."""

    manifest = result.manifest
    metadata: list[str] = []
    metadata_values: dict[str, object] = {}
    dataset_hash = manifest.get("dataset_hash")
    if (
        isinstance(dataset_hash, str)
        and dataset_hash
        and not dataset_hash.startswith(("/", "\\"))
    ):
        metadata_values["dataset_hash"] = dataset_hash
    relevance = manifest.get("relevance_level")
    if relevance in ("document", "chunk"):
        metadata_values["relevance_level"] = relevance
    top_k = manifest.get("top_k")
    if isinstance(top_k, list) and all(
        isinstance(value, int) and not isinstance(value, bool) for value in top_k
    ):
        metadata_values["top_k"] = top_k
    metric_version = manifest.get("metric_version")
    if isinstance(metric_version, int) and not isinstance(metric_version, bool):
        metadata_values["metric_version"] = metric_version
    for key in ("dataset_hash", "relevance_level", "top_k", "metric_version"):
        if key in metadata_values:
            metadata.append(
                f"<dt>{_escape(key)}</dt><dd>{_json(metadata_values[key])}</dd>"
            )
    config = _config_view(manifest)
    summary_rows: list[str] = []
    query_rows: list[str] = []
    attention: list[str] = []
    for name in sorted(result.metrics):
        for cutoff in sorted(result.metrics[name].metrics_by_cutoff):
            for metric in sorted(result.metrics[name].metrics_by_cutoff[cutoff]):
                value = result.metrics[name].metrics_by_cutoff[cutoff][metric]
                summary_rows.append(
                    f"<tr><td>{_escape(safe_identifier(name))}</td>"
                    f"<td>{_escape(metric)}</td>"
                    f"<td>{cutoff}</td><td>{value:.12g}</td></tr>"
                )
        if result.latency:
            stats = result.latency[name]
            summary_rows.append(
                f"<tr><td>{_escape(safe_identifier(name))}</td>"
                f"<td>latency_mean_ms</td><td></td>"
                f"<td>{stats.mean_ms:.12g}</td></tr>"
            )
            summary_rows.append(
                f"<tr><td>{_escape(safe_identifier(name))}</td>"
                f"<td>latency_p95_ms</td><td></td>"
                f"<td>{stats.p95_ms:.12g}</td></tr>"
            )
            for warning in stats.warnings:
                summary_rows.append(
                    f"<tr class='warning'><td>{_escape(safe_identifier(name))}</td>"
                    f"<td>warning</td><td></td><td>{_escape(warning)}</td></tr>"
                )
        for query in result.query_results[name]:
            recall_values: list[float] = []
            for values in query.metrics_by_cutoff.values():
                recall = values.get("recall")
                if recall is not None:
                    recall_values.append(recall)
            if recall_values and min(recall_values) < 1.0:
                attention.append(safe_identifier(query.query_id))
            for cutoff in sorted(query.metrics_by_cutoff):
                for metric in sorted(query.metrics_by_cutoff[cutoff]):
                    value = query.metrics_by_cutoff[cutoff][metric]
                    warning = "; ".join(query.warnings)
                    latency = (
                        ""
                        if query.search_latency_ms is None
                        else f"{query.search_latency_ms:.12g}"
                    )
                    query_rows.append(
                        f"<tr><td>{_escape(safe_identifier(name))}</td>"
                        f"<td>{_escape(safe_identifier(query.query_id))}</td>"
                        f"<td>{_escape(metric)}</td><td>{cutoff}</td><td>{value:.12g}</td>"
                        f"<td>{_escape(latency)}</td><td>{_escape(warning)}</td></tr>"
                    )
    attention_html = "".join(
        f"<li>{_escape(item)}</li>" for item in sorted(set(attention))
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Retrieval Lab report</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:1100px;"
        "margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "th,td{border:1px solid #ccc;padding:.35rem;text-align:left}"
        ".warning{background:#fff3cd}.config{white-space:pre-wrap;font-family:monospace}"
        "</style></head><body>"
        "<h1>Retrieval Lab report</h1>"
        f"<p>Run ID: {_escape(safe_identifier(result.run_id))}</p>"
        f"<section><h2>Run metadata</h2><dl>{''.join(metadata)}</dl></section>"
        f"<section><h2>Configuration</h2><div class='config'>{_json(config)}"
        "</div></section>"
        "<section><h2>Metrics and latency</h2><table><thead><tr>"
        "<th>Retriever</th><th>Metric</th><th>Cutoff</th><th>Value</th>"
        f"</tr></thead><tbody>{''.join(summary_rows)}</tbody></table></section>"
        "<section><h2>Per-query metrics</h2><table><thead><tr>"
        "<th>Retriever</th><th>Query ID</th><th>Metric</th><th>Cutoff</th>"
        "<th>Value</th><th>Search latency (ms)</th><th>Warnings</th>"
        f"</tr></thead><tbody>{''.join(query_rows)}</tbody></table></section>"
        f"<section><h2>Attention query IDs</h2><ul>"
        f"{attention_html or '<li>None</li>'}</ul></section>"
        f"{_recommendations(result)}</body></html>\n"
    )


__all__ = ["result_html"]
