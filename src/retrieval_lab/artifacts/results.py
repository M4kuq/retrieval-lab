"""Safe loading and atomic persistence for evaluation result JSON."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast

from retrieval_lab.domain.json_types import JSONValue
from retrieval_lab.evaluation.latency import LatencyStats
from retrieval_lab.exceptions import EvaluationError

if TYPE_CHECKING:
    from retrieval_lab.domain import EvaluationResult, QueryEvaluation, RetrieverMetrics


_MAX_RESULT_BYTES = 64 * 1024 * 1024
_REQUIRED_ROOT = {"schema_version", "run", "retrievers", "quality_gates"}
_REQUIRED_RUN = {"id", "manifest"}
_REQUIRED_RETRIEVER = {"metrics", "per_query"}
_REQUIRED_QUERY = {"query_id", "retrieved_ids", "metrics"}
_REQUIRED_LATENCY = {
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "max_ms",
    "sample_count",
    "failure_count",
}


class _DuplicateKey(ValueError):
    pass


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _parse_json(text: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, _DuplicateKey, TypeError, ValueError) as exc:
        raise EvaluationError(f"evaluation result JSON is invalid: {exc}") from exc
    if not isinstance(value, Mapping):
        raise EvaluationError("evaluation result JSON root must be an object")
    return value


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise EvaluationError(f"{path} keys must be strings")
    return cast(Mapping[str, object], value)


def _required_mapping(
    value: object, path: str, fields: set[str]
) -> Mapping[str, object]:
    result = _mapping(value, path)
    missing = sorted(fields - set(result))
    if missing:
        raise EvaluationError(f"{path} missing required fields {missing!r}")
    return result


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{path} must be a non-empty string")
    return value


def _finite_number(value: object, path: str) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationError(f"{path} must be a finite number")
    return result


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvaluationError(f"{path} must be a positive integer")
    return value


def _metric_key(value: object, path: str) -> tuple[str, int]:
    key = _string(value, path)
    if key.count("@") != 1:
        raise EvaluationError(f"{path} must use exactly one metric@cutoff separator")
    name, raw_cutoff = key.split("@")
    if not name or not raw_cutoff or not raw_cutoff.isdecimal():
        raise EvaluationError(f"{path} must use a non-empty metric and cutoff")
    if raw_cutoff != str(int(raw_cutoff)):
        raise EvaluationError(f"{path} cutoff must be canonical decimal")
    cutoff = _positive_int(int(raw_cutoff), f"{path} cutoff")
    return name, cutoff


def _flattened_metrics(value: object, path: str) -> dict[int, dict[str, float]]:
    mapping = _mapping(value, path)
    if not mapping:
        raise EvaluationError(f"{path} must not be empty")
    result: dict[int, dict[str, float]] = {}
    for raw_key, raw_value in mapping.items():
        name, cutoff = _metric_key(raw_key, f"{path}.{raw_key}")
        metrics = result.setdefault(cutoff, {})
        if name in metrics:
            raise EvaluationError(f"{path} contains duplicate metric {name!r}")
        metrics[name] = _finite_number(raw_value, f"{path}.{raw_key}")
    return result


def _parse_latency(value: object, path: str) -> LatencyStats:
    mapping = _required_mapping(value, path, _REQUIRED_LATENCY)
    warnings_value = mapping.get("warnings", [])
    if not isinstance(warnings_value, Sequence) or isinstance(
        warnings_value, (str, bytes)
    ):
        raise EvaluationError(f"{path}.warnings must be a list of strings")
    warnings = tuple(_string(item, f"{path}.warnings[]") for item in warnings_value)
    return LatencyStats(
        mean_ms=_finite_number(mapping["mean_ms"], f"{path}.mean_ms"),
        p50_ms=_finite_number(mapping["p50_ms"], f"{path}.p50_ms"),
        p95_ms=_finite_number(mapping["p95_ms"], f"{path}.p95_ms"),
        max_ms=_finite_number(mapping["max_ms"], f"{path}.max_ms"),
        sample_count=_positive_or_zero_int(
            mapping["sample_count"], f"{path}.sample_count"
        ),
        failure_count=_positive_or_zero_int(
            mapping["failure_count"], f"{path}.failure_count"
        ),
        warnings=warnings,
    )


def _positive_or_zero_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(f"{path} must be a non-negative integer")
    return value


def _parse_query(value: object, path: str) -> QueryEvaluation:
    from retrieval_lab.domain import QueryEvaluation

    mapping = _required_mapping(value, path, _REQUIRED_QUERY)
    retrieved = mapping["retrieved_ids"]
    if not isinstance(retrieved, Sequence) or isinstance(retrieved, (str, bytes)):
        raise EvaluationError(f"{path}.retrieved_ids must be a list of strings")
    retrieved_ids = tuple(
        _string(item, f"{path}.retrieved_ids[]") for item in retrieved
    )
    latency_value = mapping.get("search_latency_ms")
    search_latency = (
        None
        if latency_value is None
        else _finite_number(latency_value, f"{path}.search_latency_ms")
    )
    warnings_value = mapping.get("warnings", [])
    if not isinstance(warnings_value, Sequence) or isinstance(
        warnings_value, (str, bytes)
    ):
        raise EvaluationError(f"{path}.warnings must be a list of strings")
    warnings = tuple(_string(item, f"{path}.warnings[]") for item in warnings_value)
    return QueryEvaluation(
        query_id=_string(mapping["query_id"], f"{path}.query_id"),
        retrieved_ids=retrieved_ids,
        metrics_by_cutoff=_flattened_metrics(mapping["metrics"], f"{path}.metrics"),
        search_latency_ms=search_latency,
        warnings=warnings,
    )


def _same_metric_shape(
    left: Mapping[int, Mapping[str, float]],
    right: Mapping[int, Mapping[str, float]],
) -> bool:
    return {(cutoff, name) for cutoff, values in left.items() for name in values} == {
        (cutoff, name) for cutoff, values in right.items() for name in values
    }


def _validate_aggregate(
    aggregate: RetrieverMetrics,
    queries: Sequence[QueryEvaluation],
    path: str,
) -> None:
    expected: dict[int, dict[str, float]] = {}
    for cutoff, metric_values in aggregate.metrics_by_cutoff.items():
        expected[cutoff] = {}
        for name in metric_values:
            expected[cutoff][name] = sum(
                query.metrics_by_cutoff[cutoff][name] for query in queries
            ) / len(queries)
    for cutoff, values in aggregate.metrics_by_cutoff.items():
        for name, actual in values.items():
            if abs(actual - expected[cutoff][name]) > 1e-12:
                raise EvaluationError(
                    f"{path}.metrics.{name}@{cutoff} does not match per-query aggregate"
                )


def result_from_dict(payload: Mapping[str, object]) -> EvaluationResult:
    """Reconstruct a validated result from the canonical JSON-compatible shape."""

    from retrieval_lab.domain import EvaluationResult, RetrieverMetrics

    root = _required_mapping(payload, "result", _REQUIRED_ROOT)
    schema_version = root["schema_version"]
    if isinstance(schema_version, bool) or schema_version != 1:
        raise EvaluationError("result.schema_version must be exactly 1")
    quality_gates = root["quality_gates"]
    if not isinstance(quality_gates, list):
        raise EvaluationError("result.quality_gates must be a list")
    if quality_gates:
        raise EvaluationError(
            "result.quality_gates is not loadable in schema version 1"
        )
    run = _required_mapping(root["run"], "result.run", _REQUIRED_RUN)
    run_id = _string(run["id"], "result.run.id")
    manifest = _mapping(run["manifest"], "result.run.manifest")
    retriever_mapping = _mapping(root["retrievers"], "result.retrievers")
    if not retriever_mapping:
        raise EvaluationError("result.retrievers must not be empty")

    metrics: dict[str, RetrieverMetrics] = {}
    query_results: dict[str, tuple[QueryEvaluation, ...]] = {}
    latency: dict[str, LatencyStats] = {}
    expected_query_ids: tuple[str, ...] | None = None
    expected_metric_shape: Mapping[int, Mapping[str, float]] | None = None
    latency_presence: bool | None = None
    for name, raw_retriever in retriever_mapping.items():
        retriever_name = _string(name, "result.retrievers name")
        mapping = _required_mapping(
            raw_retriever,
            f"result.retrievers[{retriever_name!r}]",
            _REQUIRED_RETRIEVER,
        )
        aggregate_shape = _flattened_metrics(
            mapping["metrics"],
            f"result.retrievers[{retriever_name!r}].metrics",
        )
        aggregate = RetrieverMetrics(metrics_by_cutoff=aggregate_shape)
        raw_queries = mapping["per_query"]
        if not isinstance(raw_queries, Sequence) or isinstance(
            raw_queries, (str, bytes)
        ):
            raise EvaluationError(
                f"result.retrievers[{retriever_name!r}].per_query must be a list"
            )
        queries = tuple(
            _parse_query(
                item, f"result.retrievers[{retriever_name!r}].per_query[{index}]"
            )
            for index, item in enumerate(raw_queries)
        )
        if not queries:
            raise EvaluationError(
                f"result.retrievers[{retriever_name!r}].per_query must not be empty"
            )
        query_ids = tuple(query.query_id for query in queries)
        if len(set(query_ids)) != len(query_ids):
            raise EvaluationError(
                f"result.retrievers[{retriever_name!r}].per_query query IDs "
                "must be unique"
            )
        if expected_query_ids is None:
            expected_query_ids = query_ids
        elif query_ids != expected_query_ids:
            raise EvaluationError(
                f"result.retrievers[{retriever_name!r}].per_query query "
                "IDs/order differ"
            )
        if expected_metric_shape is None:
            expected_metric_shape = aggregate_shape
        elif not _same_metric_shape(expected_metric_shape, aggregate_shape):
            raise EvaluationError(
                f"result.retrievers[{retriever_name!r}].metrics shape differs"
            )
        for index, query in enumerate(queries):
            if not _same_metric_shape(aggregate_shape, query.metrics_by_cutoff):
                raise EvaluationError(
                    f"result.retrievers[{retriever_name!r}].per_query[{index}]"
                    ".metrics shape differs"
                )
        _validate_aggregate(
            aggregate,
            queries,
            f"result.retrievers[{retriever_name!r}]",
        )
        has_latency = "latency" in mapping
        if latency_presence is None:
            latency_presence = has_latency
        elif latency_presence != has_latency:
            raise EvaluationError(
                "result.retrievers must either all contain latency or all omit it"
            )
        if has_latency:
            latency[retriever_name] = _parse_latency(
                mapping["latency"], f"result.retrievers[{retriever_name!r}].latency"
            )
        metrics[retriever_name] = aggregate
        query_results[retriever_name] = queries

    return EvaluationResult(
        run_id=run_id,
        metrics=metrics,
        query_results=query_results,
        manifest=cast(Mapping[str, JSONValue], manifest),
        schema_version=1,
        latency=latency if latency_presence else None,
    )


def result_from_json(text: str) -> EvaluationResult:
    """Load a result from JSON text without reading a path or executing objects."""

    if not isinstance(text, str):
        raise EvaluationError("result JSON text must be a string")
    return result_from_dict(_parse_json(text))


def _validate_max_bytes(max_bytes: object) -> int:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise EvaluationError("max_bytes must be a positive integer")
    return max_bytes


def load_result(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = _MAX_RESULT_BYTES,
) -> EvaluationResult:
    """Load and validate a UTF-8 result JSON file under a size limit."""

    limit = _validate_max_bytes(max_bytes)
    try:
        source = Path(path)
        size = source.stat().st_size
        if size > limit:
            raise EvaluationError(f"result file exceeds max_bytes ({limit})")
        raw = source.read_bytes()
        if len(raw) > limit:
            raise EvaluationError(f"result file exceeds max_bytes ({limit})")
        text = raw.decode("utf-8")
    except EvaluationError:
        raise
    except (OSError, TypeError, ValueError, UnicodeError) as exc:
        raise EvaluationError(f"could not read result JSON from {path!s}") from exc
    return result_from_json(text)


def _atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    """Write UTF-8 text through fsync and same-directory replacement."""

    try:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, TypeError, ValueError) as exc:
        raise EvaluationError(f"could not prepare output path {path!s}") from exc
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
        return output
    except (OSError, TypeError, ValueError) as exc:
        raise EvaluationError(f"could not atomically write {output}") from exc
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


__all__ = [
    "_atomic_write_text",
    "load_result",
    "result_from_dict",
    "result_from_json",
]
