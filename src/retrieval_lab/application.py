"""Application services used by the Retrieval Lab command-line adapter.

The functions in this module intentionally contain no presentation logic.  They
are a small, typed bridge between configuration loading, ``EvaluationRunner``,
and the result persistence API.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from retrieval_lab.config import RetrievalConfig, load_config
from retrieval_lab.domain import EvaluationResult
from retrieval_lab.exceptions import ConfigurationError, EvaluationError
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
    "ExperimentOutput",
    "InitializedProject",
    "ValidationResult",
    "initialize_project",
    "run_configured_experiment",
    "validate_config_inputs",
]
