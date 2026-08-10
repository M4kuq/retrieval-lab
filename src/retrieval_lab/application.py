"""Application services used by the Retrieval Lab command-line adapter.

The functions in this module intentionally contain no presentation logic.  They
are a small, typed bridge between configuration loading, ``EvaluationRunner``,
and the result persistence API.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from retrieval_lab.artifacts.results import load_result
from retrieval_lab.comparison import MetricDelta, RunComparison, compare_runs
from retrieval_lab.config import RetrievalConfig, load_config
from retrieval_lab.domain import (
    EvaluationResult,
    JSONValue,
    QualityGateReport,
)
from retrieval_lab.exceptions import (
    ConfigurationError,
    EvaluationError,
)
from retrieval_lab.quality import evaluate_quality_gates
from retrieval_lab.runner import EvaluationRunner

_SUPPORTED_FORMATS = frozenset({"json", "csv", "html"})

_TEMPLATE_FILES: dict[Path, str] = {
    Path("retrieval-lab.yaml"): """schema_version: 1
experiment:
  seed: 42
corpus:
  path: corpus
  include: [\"*.md\", \"**/*.md\"]
  chunker:
    type: recursive_characters
    size: 256
    overlap: 32
dataset:
  path: evaluation.jsonl
  format: native_jsonl
  relevance_level: document
retrievers:
  - name: keyword
    type: keyword
  - name: bm25
    type: bm25
evaluation:
  top_k: [1, 3]
  metrics: [hit_rate, recall, precision, mrr, ndcg, ap]
  repetitions: 1
  concurrency: 1
quality_gates: []
report:
  output_dir: reports
  formats: [json, csv, html]
""",
    Path("corpus/example.md"): """# Retrieval Lab のサンプル

これはローカルで検索評価を実行するための最小コーパスです。検索品質と再現性を確認できます。
""",
    Path("evaluation.jsonl"): (
        '{"query_id":"q-example","query":"検索品質",'
        '"relevant":[{"id":"example.md","relevance":1}]}\n'
    ),
}


@dataclass(frozen=True)
class InitializedProject:
    """Files created by :func:`initialize_project`."""

    target: Path
    files: tuple[Path, ...]


@dataclass(frozen=True)
class ValidationResult:
    """Validated configuration and the input counts discovered by the runner."""

    config: RetrievalConfig
    document_count: int
    query_count: int
    retriever_names: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentOutput:
    """An evaluation result and the report files written for it."""

    result: EvaluationResult
    formats: tuple[str, ...]
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class QueryEvidence:
    """Safe, deterministic evidence for one retriever/query pair."""

    retriever: str
    query_id: str
    retrieved_ids: tuple[str, ...]
    metrics: tuple[tuple[str, float], ...]
    search_latency_ms: float | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the machine-readable query evidence shape."""

        return {
            "metrics": {name: value for name, value in self.metrics},
            "query_id": self.query_id,
            "retrieved_ids": list(self.retrieved_ids),
            "retriever": self.retriever,
            "search_latency_ms": self.search_latency_ms,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class InspectionOutput:
    """Loaded result metadata and optional per-query evidence."""

    result: EvaluationResult
    query_id: str | None
    evidence: tuple[QueryEvidence, ...]

    @property
    def gate_status(self) -> tuple[tuple[int, str, str, bool], ...]:
        """Return gate index, retriever, metric, and pass state."""

        return tuple(
            (
                gate.gate_index,
                gate.retriever,
                gate.metric,
                gate.passed,
            )
            for gate in self.result.quality_gates
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return deterministic machine-readable inspection data."""

        payload: dict[str, JSONValue] = {
            "command": "inspect",
            "quality_gates": [
                {
                    "gate_index": index,
                    "metric": metric,
                    "passed": passed,
                    "retriever": retriever,
                }
                for index, retriever, metric, passed in self.gate_status
            ],
            "run_id": self.result.run_id,
            "schema_version": self.result.schema_version,
            "retrievers": list(sorted(self.result.metrics)),
            "summary": self.result.summary(),
        }
        if self.query_id is not None:
            payload["query"] = {
                "evidence": [item.to_dict() for item in self.evidence],
                "query_id": self.query_id,
            }
        return payload

    def to_json(self) -> str:
        """Return deterministic strict JSON for automation."""

        return _strict_json(self.to_dict())


@dataclass(frozen=True)
class ComparisonRow:
    """One aggregate metric delta prepared for deterministic display."""

    retriever: str
    metric: str
    cutoff: int | None
    baseline: float
    candidate: float
    absolute_delta: float
    relative_delta: float | None
    relative_status: str
    direction: str
    classification: str

    @classmethod
    def from_delta(cls, delta: MetricDelta) -> ComparisonRow:
        """Build a display row from the comparison domain record."""

        return cls(
            retriever=delta.retriever,
            metric=delta.metric,
            cutoff=delta.cutoff,
            baseline=delta.baseline,
            candidate=delta.candidate,
            absolute_delta=delta.absolute_delta,
            relative_delta=delta.relative_delta,
            relative_status=delta.relative_status,
            direction=delta.direction,
            classification=delta.classification,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the machine-readable comparison row shape."""

        return {
            "absolute_delta": self.absolute_delta,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "classification": self.classification,
            "cutoff": self.cutoff,
            "direction": self.direction,
            "metric": self.metric,
            "relative_delta": self.relative_delta,
            "relative_status": self.relative_status,
            "retriever": self.retriever,
        }


@dataclass(frozen=True)
class ComparisonOutput:
    """Comparable saved runs and deterministic aggregate metric rows."""

    comparison: RunComparison
    rows: tuple[ComparisonRow, ...]

    def to_dict(self) -> dict[str, JSONValue]:
        """Return deterministic machine-readable comparison data."""

        report = self.comparison.comparability
        variable_differences: JSONValue = [
            {
                "field": item.field,
                "reason": item.reason,
            }
            for item in report.variable_differences
        ]
        return {
            "baseline_run_id": self.comparison.baseline_run_id,
            "candidate_run_id": self.comparison.candidate_run_id,
            "command": "compare",
            "common_retrievers": list(report.common_retrievers),
            "diagnostics": [
                {
                    "candidate_value": item.candidate_value,
                    "field": item.field,
                    "reason": item.reason,
                    "baseline_value": item.baseline_value,
                }
                for item in report.diagnostics
            ],
            "metrics": [row.to_dict() for row in self.rows],
            "variable_differences": variable_differences,
        }

    def to_json(self) -> str:
        """Return deterministic strict JSON for automation."""

        return _strict_json(self.to_dict())


@dataclass(frozen=True)
class GateOutput:
    """Configured quality-gate evaluation and its pass/fail state."""

    report: QualityGateReport

    def to_dict(self) -> dict[str, JSONValue]:
        """Return deterministic machine-readable gate data."""

        return {"command": "gate", **self.report.to_dict()}

    def to_json(self) -> str:
        """Return deterministic strict JSON for automation."""

        return _strict_json(self.to_dict())


def initialize_project(
    target: str | os.PathLike[str], *, force: bool = False
) -> InitializedProject:
    """Create a safe, offline, runnable Retrieval Lab project template.

    Only the three files owned by this template are ever written.  Existing
    files are preserved unless ``force`` is true; unrelated files are never
    removed.
    """

    if not isinstance(force, bool):
        raise ConfigurationError("force must be boolean")
    project = _path_from_input(target, "project target")
    _reject_symlink_components(project, stop=project)
    try:
        if project.exists() and not project.is_dir():
            raise ConfigurationError("project target must be a directory")
    except ConfigurationError:
        raise
    except (OSError, ValueError) as exc:
        raise ConfigurationError("project target is not accessible") from exc
    try:
        project.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError("could not create project directory") from exc

    paths = tuple(project / relative for relative in _TEMPLATE_FILES)
    for path in paths:
        _reject_symlink_components(path, stop=project)
        try:
            if path.exists() and not path.is_file():
                raise ConfigurationError("template path must be a regular file")
        except ConfigurationError:
            raise
        except (OSError, ValueError) as exc:
            raise ConfigurationError("template path is not accessible") from exc
    if not force and any(path.exists() for path in paths):
        raise ConfigurationError("project already contains a Retrieval Lab file")

    try:
        for relative, content in _TEMPLATE_FILES.items():
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ConfigurationError("could not write project template") from exc
    return InitializedProject(target=project, files=paths)


def validate_config_inputs(
    config_path: str | os.PathLike[str],
) -> ValidationResult:
    """Validate configuration, corpus, dataset, and gold IDs without searching."""

    path = _path_from_input(config_path, "config path")
    config = load_config(path)
    runner = EvaluationRunner.from_config(config)

    # Runner construction validates document-level judgments.  Chunk-level
    # judgments are checked here after deterministic chunking, still without
    # indexing, searching, or calculating evaluation metrics.
    if runner._dataset is not None and runner._relevance_level == "chunk":
        chunks = runner._chunker.chunk(runner._documents)
        runner._dataset.validate(chunks=chunks)

    return _validation_result(config, runner)


def run_configured_experiment(
    config_path: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    formats: Sequence[str] | None = None,
) -> ExperimentOutput:
    """Run a configured experiment and persist the selected report formats."""

    path = _path_from_input(config_path, "config path")
    config = load_config(path)
    selected = _normalize_formats(config.report.formats if formats is None else formats)
    destination = (
        _path_from_input(output_dir, "output directory")
        if output_dir is not None
        else config.report.output_dir
    )
    if destination is None:
        raise ConfigurationError(
            "an output directory is required when report.formats are selected"
        )
    _validate_output_directory(destination)
    runner = EvaluationRunner.from_config(config)
    result = runner.run()

    written: list[Path] = []
    try:
        if "json" in selected:
            json_path = destination / "result.json"
            result.save_json(json_path)
            written.append(json_path)
        if "csv" in selected:
            summary, per_query = result.save_csv(destination)
            written.extend((summary, per_query))
        if "html" in selected:
            html_path = destination / "report.html"
            result.save_html(html_path)
            written.append(html_path)
    except EvaluationError as exc:
        raise EvaluationError("could not save experiment reports") from exc
    return ExperimentOutput(
        result=result,
        formats=selected,
        paths=tuple(written),
    )


def inspect_result(
    result_path: str | os.PathLike[str],
    *,
    query_id: str | None = None,
    max_bytes: int = 64 * 1024 * 1024,
) -> InspectionOutput:
    """Load a saved result and prepare deterministic inspection data."""

    result = load_result(
        _checked_input_file(result_path, "result path"), max_bytes=max_bytes
    )
    if query_id is None:
        evidence: tuple[QueryEvidence, ...] = ()
    else:
        if not isinstance(query_id, str) or not query_id.strip():
            raise ConfigurationError("query_id must be a non-empty string")
        rows: list[QueryEvidence] = []
        for retriever in sorted(result.query_results):
            matching = tuple(
                query
                for query in result.query_results[retriever]
                if query.query_id == query_id
            )
            if not matching:
                continue
            query = matching[0]
            metrics = tuple(
                (
                    f"{metric}@{cutoff}",
                    value,
                )
                for cutoff in sorted(query.metrics_by_cutoff)
                for metric, value in sorted(query.metrics_by_cutoff[cutoff].items())
            )
            rows.append(
                QueryEvidence(
                    retriever=retriever,
                    query_id=query.query_id,
                    retrieved_ids=query.retrieved_ids,
                    metrics=metrics,
                    search_latency_ms=query.search_latency_ms,
                    warnings=query.warnings,
                )
            )
        if not rows:
            raise ConfigurationError(f"unknown query_id {query_id!r}")
        evidence = tuple(rows)
    return InspectionOutput(result=result, query_id=query_id, evidence=evidence)


def compare_result_files(
    baseline_path: str | os.PathLike[str],
    candidate_path: str | os.PathLike[str],
    *,
    max_bytes: int = 64 * 1024 * 1024,
) -> ComparisonOutput:
    """Load and compare two saved results through the strict public APIs."""

    baseline = load_result(
        _checked_input_file(baseline_path, "baseline result path"),
        max_bytes=max_bytes,
    )
    candidate = load_result(
        _checked_input_file(candidate_path, "candidate result path"),
        max_bytes=max_bytes,
    )
    comparison = compare_runs(baseline, candidate)
    rows = tuple(
        ComparisonRow.from_delta(item.aggregate)
        for retriever in sorted(comparison.metrics)
        for item in sorted(
            comparison.metrics[retriever],
            key=lambda value: (
                value.metric,
                -1 if value.cutoff is None else value.cutoff,
            ),
        )
    )
    return ComparisonOutput(comparison=comparison, rows=rows)


def evaluate_configured_quality_gates(
    config_path: str | os.PathLike[str],
    candidate_path: str | os.PathLike[str],
    *,
    baseline_path: str | os.PathLike[str] | None = None,
    max_bytes: int = 64 * 1024 * 1024,
) -> GateOutput:
    """Load config/results and evaluate its typed quality gates."""

    config = load_config(_checked_input_file(config_path, "config path"))
    candidate = load_result(
        _checked_input_file(candidate_path, "candidate result path"),
        max_bytes=max_bytes,
    )
    baseline = (
        None
        if baseline_path is None
        else load_result(
            _checked_input_file(baseline_path, "baseline result path"),
            max_bytes=max_bytes,
        )
    )
    report = evaluate_quality_gates(
        candidate,
        config.quality_gates,
        baseline=baseline,
    )
    return GateOutput(report=report)


def _path_from_input(value: str | os.PathLike[str] | None, field: str) -> Path:
    if value is None:
        raise ConfigurationError(f"{field} must be provided")
    if isinstance(value, str) and not value.strip():
        raise ConfigurationError(f"{field} must not be empty")
    try:
        path = Path(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field} must be a valid path") from exc
    if not str(path):
        raise ConfigurationError(f"{field} must not be empty")
    return path


def _checked_input_file(value: str | os.PathLike[str], field: str) -> Path:
    path = _path_from_input(value, field)
    absolute = path.absolute()
    _reject_symlink_components(absolute, stop=absolute)
    try:
        if not path.exists() or not path.is_file():
            raise ConfigurationError(f"{field} must be an existing file")
    except ConfigurationError:
        raise
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"{field} is not accessible") from exc
    return path


def _strict_json(value: dict[str, JSONValue]) -> str:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationError("application output could not be serialized") from exc


def _validation_result(
    config: RetrievalConfig, runner: EvaluationRunner
) -> ValidationResult:
    """Adapt runner's validated input snapshot to the application result."""

    # EvaluationRunner intentionally owns corpus/dataset construction.  This
    # is the only adapter that reads its validated private snapshot; no CLI
    # code depends on runner internals.
    return ValidationResult(
        config=config,
        document_count=len(runner._documents),
        query_count=len(runner._queries),
        retriever_names=tuple(retriever.name for retriever in runner._retrievers),
    )


def _reject_symlink_components(path: Path, *, stop: Path) -> None:
    """Reject existing symlinks from ``path`` through the owned ``stop`` path."""

    current = path
    try:
        while True:
            if current.is_symlink():
                raise ConfigurationError("path symlinks are not allowed")
            if current == stop:
                break
            parent = current.parent
            if parent == current:
                break
            current = parent
    except ConfigurationError:
        raise
    except (OSError, ValueError) as exc:
        raise ConfigurationError("path is not accessible") from exc


def _validate_output_directory(path: Path) -> None:
    _reject_symlink_components(path, stop=path)
    try:
        if path.exists() and not path.is_dir():
            raise ConfigurationError("output directory must be a directory")
    except ConfigurationError:
        raise
    except (OSError, ValueError) as exc:
        raise ConfigurationError("output directory is not accessible") from exc


def _normalize_formats(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ConfigurationError("formats must be a sequence of values")
    selected: list[str] = []
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ConfigurationError("formats must be a sequence of values") from exc
    for value in iterator:
        if not isinstance(value, str) or value not in _SUPPORTED_FORMATS:
            raise ConfigurationError("format must be one of json, csv, or html")
        if value not in selected:
            selected.append(value)
    if not selected:
        raise ConfigurationError("at least one report format is required")
    return tuple(
        format_name
        for format_name in ("json", "csv", "html")
        if format_name in selected
    )


__all__ = [
    "ComparisonOutput",
    "ComparisonRow",
    "ExperimentOutput",
    "GateOutput",
    "InitializedProject",
    "InspectionOutput",
    "QueryEvidence",
    "ValidationResult",
    "compare_result_files",
    "evaluate_configured_quality_gates",
    "initialize_project",
    "inspect_result",
    "run_configured_experiment",
    "validate_config_inputs",
]
