"""Deterministic, local-only benchmark generation and execution.

This module deliberately stays outside the package API. It exercises the
public Retrieval Lab runner and result records, then keeps only aggregate
metrics, latency, and cache observations in its report.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from time import perf_counter_ns
from typing import Literal, cast

from retrieval_lab import Document, EvaluationDataset, EvaluationQuery, EvaluationRunner
from retrieval_lab.domain import JSONValue
from retrieval_lab.exceptions import EvaluationError

BenchmarkSize = Literal["small", "medium"]


@dataclass(frozen=True)
class BenchmarkSpec:
    """Validated parameters for one synthetic benchmark run."""

    size: BenchmarkSize = "small"
    seed: int = 42
    top_k: tuple[int, ...] = (1, 3, 5)
    repetitions: int = 1

    def __post_init__(self) -> None:
        if self.size not in ("small", "medium"):
            raise EvaluationError("benchmark size must be small or medium")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EvaluationError("benchmark seed must be a non-negative integer")
        if self.seed < 0:
            raise EvaluationError("benchmark seed must be a non-negative integer")
        if isinstance(self.top_k, (str, bytes)) or not isinstance(self.top_k, Sequence):
            raise EvaluationError("benchmark top_k must be a sequence")
        normalized = tuple(self.top_k)
        if not normalized or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in normalized
        ):
            raise EvaluationError("benchmark top_k must contain positive integers")
        if len(set(normalized)) != len(normalized):
            raise EvaluationError("benchmark top_k must not contain duplicates")
        if normalized != tuple(sorted(normalized)):
            raise EvaluationError("benchmark top_k must be sorted")
        object.__setattr__(self, "top_k", normalized)
        if isinstance(self.repetitions, bool) or self.repetitions != 1:
            raise EvaluationError(
                "benchmark repetitions must be 1; the v0.1 runner is single-run"
            )

    @property
    def document_count(self) -> int:
        """Return the deterministic corpus size for this benchmark tier."""

        return 24 if self.size == "small" else 160

    @property
    def query_count(self) -> int:
        """Return the deterministic query count for this benchmark tier."""

        return 8 if self.size == "small" else 40


@dataclass(frozen=True)
class SyntheticBenchmarkData:
    """In-memory synthetic documents and graded document-level qrels."""

    documents: tuple[Document, ...]
    dataset: EvaluationDataset
    identifier: str = "retrieval-lab.synthetic"
    version: str = "1"


def generate_synthetic_data(spec: BenchmarkSpec) -> SyntheticBenchmarkData:
    """Generate a deterministic local corpus and graded qrels dataset."""

    rng = random.Random(spec.seed)
    topics = ("retrieval", "cache", "metrics", "safety", "reproducibility")
    documents: list[Document] = []
    for index in range(spec.document_count):
        topic = topics[(index + rng.randrange(len(topics))) % len(topics)]
        marker = f"marker-{spec.seed}-{index:04d}"
        documents.append(
            Document(
                id=f"synthetic-doc-{index:04d}",
                text=(
                    f"synthetic topic {topic} {marker}; "
                    "offline deterministic benchmark document"
                ),
                source=f"synthetic-{index:04d}.md",
            )
        )

    queries: list[EvaluationQuery] = []
    grades: dict[str, Mapping[str, int]] = {}
    for query_index in range(spec.query_count):
        document_index = rng.randrange(spec.document_count)
        document_id = f"synthetic-doc-{document_index:04d}"
        query_id = f"synthetic-query-{query_index:04d}"
        marker = f"marker-{spec.seed}-{document_index:04d}"
        queries.append(
            EvaluationQuery(
                id=query_id,
                query=f"find {marker}",
                relevant_document_ids={document_id},
            )
        )
        grades[query_id] = {document_id: 3}

    dataset = EvaluationDataset(
        queries,
        relevance_level="document",
        relevance_grades_by_query=grades,
    )
    return SyntheticBenchmarkData(tuple(documents), dataset)


def run_benchmark(spec: BenchmarkSpec) -> dict[str, JSONValue]:
    """Run cold and warm cached evaluations in one process."""

    started_ns = perf_counter_ns()
    started_at = _utc_timestamp()
    data = generate_synthetic_data(spec)
    with tempfile.TemporaryDirectory(prefix="retrieval-lab-benchmark-") as cache_dir:
        cold = _run_phase(data, spec, cache_dir=cache_dir)
        warm = _run_phase(data, spec, cache_dir=cache_dir)

    if cold["run_id"] != warm["run_id"]:
        raise EvaluationError("cold and warm benchmark run IDs differ")
    if cold["metrics"] != warm["metrics"]:
        raise EvaluationError("cold and warm benchmark metrics differ")

    finished_at = _utc_timestamp()
    duration_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
    environment = _environment()
    payload: dict[str, JSONValue] = {
        "benchmark": {
            "dataset_id": data.identifier,
            "dataset_version": data.version,
            "document_count": len(data.documents),
            "query_count": len(data.dataset.queries),
            "relevance_level": data.dataset.relevance_level,
            "retrievers": ["keyword", "bm25"],
            "seed": spec.seed,
            "size": spec.size,
            "top_k": list(spec.top_k),
            "repetitions": spec.repetitions,
        },
        "comparisons": {
            "metrics_equal": True,
            "run_id_equal": True,
        },
        "duration_ms": duration_ms,
        "environment": environment,
        "finished_at_utc": finished_at,
        "runs": {
            "cold": cold,
            "warm": warm,
        },
        "schema_version": 1,
        "started_at_utc": started_at,
    }
    _ensure_finite_json(payload)
    return payload


def save_benchmark(
    payload: Mapping[str, JSONValue], output: str | os.PathLike[str]
) -> Path:
    """Atomically save a validated report without exposing its destination path."""

    destination = _safe_output_path(output)
    _ensure_finite_json(payload)
    try:
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError("benchmark report is not strict JSON") from exc

    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    except OSError as exc:
        raise EvaluationError("could not atomically write benchmark report") from exc
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()
    return destination


def load_benchmark(path: str | os.PathLike[str]) -> dict[str, JSONValue]:
    """Load one benchmark report with duplicate-key and non-finite rejection."""

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvaluationError("benchmark report is not valid strict JSON") from exc
    if not isinstance(value, dict):
        raise EvaluationError("benchmark report root must be an object")
    _ensure_finite_json(value)
    _validate_report_shape(value)
    return cast(dict[str, JSONValue], value)


def _run_phase(
    data: SyntheticBenchmarkData,
    spec: BenchmarkSpec,
    *,
    cache_dir: str,
) -> dict[str, JSONValue]:
    phase_started_ns = perf_counter_ns()
    result = EvaluationRunner.from_dataset(
        documents=data.documents,
        dataset=data.dataset,
        strategies=("keyword", "bm25"),
        top_k=spec.top_k,
        cache_dir=cache_dir,
        seed=spec.seed,
    ).run()
    runtime = result.manifest.get("runtime", {})
    runtime_mapping = runtime if isinstance(runtime, Mapping) else {}
    events = runtime_mapping.get("cache_events", [])
    cache_events = _cache_event_summary(events)
    build_ms = _finite_mapping(runtime_mapping.get("build_ms", {}))
    index_sizes = _integer_mapping(runtime_mapping.get("index_sizes_bytes", {}))
    metrics: dict[str, JSONValue] = {
        name: cast(JSONValue, result.metrics[name].to_dict())
        for name in sorted(result.metrics)
    }
    latency: dict[str, JSONValue] = {
        name: cast(JSONValue, result.latency[name].to_dict())
        for name in sorted(result.latency)
    }
    return {
        "build_ms": build_ms,
        "cache": "cold" if _is_cold(cache_events) else "warm",
        "cache_events": cast(JSONValue, cache_events),
        "duration_ms": (perf_counter_ns() - phase_started_ns) / 1_000_000.0,
        "index_sizes_bytes": index_sizes,
        "latency": latency,
        "metrics": metrics,
        "run_id": result.run_id,
    }


def _is_cold(events: list[dict[str, JSONValue]]) -> bool:
    return any(event.get("status") != "hit" for event in events)


def _cache_event_summary(value: object) -> list[dict[str, JSONValue]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    summary: list[dict[str, JSONValue]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        artifact = raw.get("artifact")
        status = raw.get("status")
        duration = raw.get("duration_ms")
        if not isinstance(artifact, str) or not isinstance(status, str):
            continue
        event: dict[str, JSONValue] = {"artifact": artifact, "status": status}
        if (
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and math.isfinite(float(duration))
            and float(duration) >= 0.0
        ):
            event["duration_ms"] = float(duration)
        summary.append(event)
    return summary


def _finite_mapping(value: object) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, JSONValue] = {}
    for key in sorted(value):
        number = value[key]
        if (
            isinstance(key, str)
            and isinstance(number, (int, float))
            and not isinstance(number, bool)
            and math.isfinite(float(number))
            and float(number) >= 0.0
        ):
            result[key] = float(number)
    return result


def _integer_mapping(value: object) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: number
        for key, number in sorted(value.items())
        if isinstance(key, str)
        and isinstance(number, int)
        and not isinstance(number, bool)
        and number >= 0
    }


def _environment() -> dict[str, JSONValue]:
    try:
        version = importlib_metadata.version("retrieval-lab")
    except importlib_metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "cpu": platform.processor() or platform.machine() or "unknown",
        "os": {
            "machine": platform.machine() or "unknown",
            "release": platform.release() or "unknown",
            "system": platform.system() or "unknown",
        },
        "python": platform.python_version(),
        "retrieval_lab_version": version,
    }


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_output_path(value: str | os.PathLike[str]) -> Path:
    try:
        path = Path(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationError("benchmark output must be a valid file path") from exc
    if not str(path) or path.name in ("", ".", ".."):
        raise EvaluationError("benchmark output must be a file path")
    absolute = path.absolute()
    if _has_symlink_component(absolute):
        raise EvaluationError("benchmark output path must not contain symlinks")
    if absolute.exists() and absolute.is_dir():
        raise EvaluationError("benchmark output must be a file path")
    if absolute.exists() and absolute.is_symlink():
        raise EvaluationError("benchmark output must not be a symlink")
    return absolute


def _has_symlink_component(path: Path) -> bool:
    current = path
    while current != current.parent:
        if current.is_symlink():
            return True
        current = current.parent
    return current.is_symlink()


def _ensure_finite_json(value: object) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluationError("benchmark report must not contain NaN or infinity")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvaluationError("benchmark report keys must be strings")
            _ensure_finite_json(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _ensure_finite_json(item)
        return
    raise EvaluationError("benchmark report contains an unsupported value")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def _validate_report_shape(value: Mapping[str, object]) -> None:
    required = {
        "benchmark",
        "comparisons",
        "duration_ms",
        "environment",
        "finished_at_utc",
        "runs",
        "schema_version",
        "started_at_utc",
    }
    if set(value) != required:
        raise EvaluationError("benchmark report has an invalid schema")
    if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
        raise EvaluationError("benchmark report schema_version must be 1")
    if not isinstance(value["benchmark"], Mapping):
        raise EvaluationError("benchmark metadata must be an object")
    if not isinstance(value["environment"], Mapping):
        raise EvaluationError("benchmark environment must be an object")
    runs = value["runs"]
    if not isinstance(runs, Mapping) or set(runs) != {"cold", "warm"}:
        raise EvaluationError("benchmark runs must contain cold and warm")
    for phase in ("cold", "warm"):
        item = runs[phase]
        if not isinstance(item, Mapping):
            raise EvaluationError(f"benchmark {phase} run must be an object")
        for field in (
            "cache",
            "run_id",
            "metrics",
            "latency",
            "cache_events",
            "duration_ms",
        ):
            if field not in item:
                raise EvaluationError(f"benchmark {phase} run is missing {field}")


__all__ = [
    "BenchmarkSpec",
    "SyntheticBenchmarkData",
    "generate_synthetic_data",
    "load_benchmark",
    "run_benchmark",
    "save_benchmark",
]
