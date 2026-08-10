"""Immutable, typed configuration records for Retrieval Lab.

The records in this module contain no I/O and are intentionally independent of
the YAML parser.  They are also the normalized configuration representation
recorded in an evaluation manifest.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Literal, TypeAlias

from retrieval_lab.domain import JSONValue
from retrieval_lab.exceptions import ConfigurationError

ChunkerType = Literal["recursive_characters"]
DatasetFormat = Literal["native_jsonl"]
RelevanceLevel = Literal["document", "chunk"]
RetrieverType = Literal["keyword", "bm25", "dense", "hybrid"]
SUPPORTED_METRICS = ("hit_rate", "recall", "precision", "mrr", "ndcg", "ap")
SUPPORTED_LATENCY_METRICS = (
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_max_ms",
)
SUPPORTED_REPORT_FORMATS = ("json", "csv", "html")


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field_name} must be a non-empty string")
    return value


def _positive(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{field_name} must be a positive integer")
    return value


def _non_negative(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(f"{field_name} must be a non-negative integer")
    return value


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field_name} must be a finite number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ConfigurationError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ConfigurationError(f"{field_name} must be a finite number")
    return normalized


def _path(value: object, field_name: str) -> Path:
    if not isinstance(value, Path):
        raise ConfigurationError(f"{field_name} must be a Path")
    return value


def _quality_metric(value: str) -> None:
    if value in SUPPORTED_LATENCY_METRICS:
        return
    if value.count("@") != 1:
        raise ConfigurationError(
            "quality_gates.metric must be metric@positive_cutoff or a supported "
            "latency metric"
        )
    name, raw_cutoff = value.split("@")
    if name not in SUPPORTED_METRICS or not raw_cutoff.isdecimal():
        raise ConfigurationError("quality_gates.metric is not supported")
    try:
        cutoff = int(raw_cutoff)
    except ValueError as exc:
        raise ConfigurationError(
            "quality_gates.metric cutoff must be a canonical positive integer"
        ) from exc
    if cutoff <= 0 or raw_cutoff != str(cutoff):
        raise ConfigurationError(
            "quality_gates.metric cutoff must be canonical positive integer"
        )


@dataclass(frozen=True)
class ChunkerConfig:
    """Configuration for the deterministic character chunker."""

    type: ChunkerType = "recursive_characters"
    size: int = 512
    overlap: int = 64

    def __post_init__(self) -> None:
        if self.type != "recursive_characters":
            raise ConfigurationError(
                "corpus.chunker.type must be 'recursive_characters'"
            )
        size = _positive(self.size, "corpus.chunker.size")
        overlap = _non_negative(self.overlap, "corpus.chunker.overlap")
        if overlap >= size:
            raise ConfigurationError(
                "corpus.chunker.overlap must be smaller than corpus.chunker.size"
            )
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "overlap", overlap)


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment identity and reproducibility settings."""

    name: str | None = None
    seed: int = 42
    workspace: Path | None = None
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.name is not None:
            _non_empty(self.name, "experiment.name")
        _non_negative(self.seed, "experiment.seed")
        if self.workspace is not None:
            _path(self.workspace, "experiment.workspace")
        if self.cache_dir is not None:
            _path(self.cache_dir, "experiment.cache_dir")

    @property
    def effective_cache_dir(self) -> Path | None:
        """Return explicit cache_dir, falling back to workspace."""

        if self.cache_dir is not None:
            return self.cache_dir
        return self.workspace / "cache" if self.workspace is not None else None


@dataclass(frozen=True)
class CorpusConfig:
    """Corpus source, include filters, and chunking settings."""

    path: Path
    include: tuple[str, ...] = ()
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)

    def __post_init__(self) -> None:
        _path(self.path, "corpus.path")
        if isinstance(self.include, (str, bytes)) or not isinstance(
            self.include, Sequence
        ):
            raise ConfigurationError("corpus.include must be a sequence")
        if any(
            not isinstance(value, str) or not value.strip() for value in self.include
        ):
            raise ConfigurationError("corpus.include must contain non-empty strings")
        object.__setattr__(self, "include", tuple(self.include))
        if not isinstance(self.chunker, ChunkerConfig):
            raise ConfigurationError("corpus.chunker must be a ChunkerConfig")


@dataclass(frozen=True)
class DatasetConfig:
    """Native JSONL evaluation dataset settings."""

    path: Path
    format: DatasetFormat = "native_jsonl"
    relevance_level: RelevanceLevel = "document"

    def __post_init__(self) -> None:
        _path(self.path, "dataset.path")
        if self.format != "native_jsonl":
            raise ConfigurationError("dataset.format must be 'native_jsonl'")
        if self.relevance_level not in ("document", "chunk"):
            raise ConfigurationError(
                "dataset.relevance_level must be 'document' or 'chunk'"
            )


@dataclass(frozen=True)
class KeywordRetrieverConfig:
    """Configuration for the deterministic keyword retriever."""

    name: str
    type: Literal["keyword"] = "keyword"

    def __post_init__(self) -> None:
        _non_empty(self.name, "retriever.name")
        if self.type != "keyword":
            raise ConfigurationError("retriever.type must be 'keyword'")


@dataclass(frozen=True)
class BM25RetrieverConfig:
    """Configuration for dependency-free BM25 retrieval."""

    name: str
    type: Literal["bm25"] = "bm25"
    tokenizer: Literal["default"] = "default"
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        _non_empty(self.name, "retriever.name")
        if self.type != "bm25":
            raise ConfigurationError("retriever.type must be 'bm25'")
        if self.tokenizer != "default":
            raise ConfigurationError("retriever.tokenizer must be 'default'")
        k1 = _finite(self.k1, "retriever.k1")
        b = _finite(self.b, "retriever.b")
        if k1 <= 0.0:
            raise ConfigurationError("retriever.k1 must be greater than zero")
        if not 0.0 <= b <= 1.0:
            raise ConfigurationError("retriever.b must be between zero and one")
        object.__setattr__(self, "k1", k1)
        object.__setattr__(self, "b", b)


@dataclass(frozen=True)
class DenseRetrieverConfig:
    """Configuration for lazy optional exact dense retrieval."""

    name: str
    type: Literal["dense"] = "dense"
    model: str = "intfloat/multilingual-e5-small"
    model_revision: str | None = None
    normalize_embeddings: bool = True
    query_prompt: str = "query: "
    document_prompt: str = "passage: "
    batch_size: int = 32

    def __post_init__(self) -> None:
        _non_empty(self.name, "retriever.name")
        if self.type != "dense":
            raise ConfigurationError("retriever.type must be 'dense'")
        _non_empty(self.model, "retriever.model")
        if self.model_revision is not None:
            _non_empty(self.model_revision, "retriever.model_revision")
        if not isinstance(self.normalize_embeddings, bool):
            raise ConfigurationError("retriever.normalize_embeddings must be boolean")
        if not isinstance(self.query_prompt, str):
            raise ConfigurationError("retriever.query_prompt must be a string")
        if not isinstance(self.document_prompt, str):
            raise ConfigurationError("retriever.document_prompt must be a string")
        _positive(self.batch_size, "retriever.batch_size")


@dataclass(frozen=True)
class HybridRetrieverConfig:
    """Configuration for reciprocal-rank fusion over named sources."""

    name: str
    type: Literal["hybrid"] = "hybrid"
    sources: tuple[str, ...] = ()
    fusion: Literal["rrf"] = "rrf"
    rrf_k: int = 60
    candidate_k: int = 100

    def __post_init__(self) -> None:
        _non_empty(self.name, "retriever.name")
        if self.type != "hybrid":
            raise ConfigurationError("retriever.type must be 'hybrid'")
        if self.fusion != "rrf":
            raise ConfigurationError("retriever.fusion must be 'rrf'")
        if isinstance(self.sources, (str, bytes)) or not isinstance(
            self.sources, Sequence
        ):
            raise ConfigurationError("retriever.sources must be a sequence")
        if len(self.sources) < 2:
            raise ConfigurationError(
                "retriever.sources must contain at least two names"
            )
        if any(
            not isinstance(value, str) or not value.strip() for value in self.sources
        ):
            raise ConfigurationError("retriever.sources must contain non-empty strings")
        if len(set(self.sources)) != len(self.sources):
            raise ConfigurationError("retriever.sources must not contain duplicates")
        object.__setattr__(self, "sources", tuple(self.sources))
        _positive(self.rrf_k, "retriever.rrf_k")
        _positive(self.candidate_k, "retriever.candidate_k")


RetrieverConfig: TypeAlias = (
    KeywordRetrieverConfig
    | BM25RetrieverConfig
    | DenseRetrieverConfig
    | HybridRetrieverConfig
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Metric cutoffs and v0.1 execution controls."""

    top_k: tuple[int, ...] = (1, 3, 5, 10)
    metrics: tuple[str, ...] = SUPPORTED_METRICS
    repetitions: int = 1
    concurrency: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.top_k, (str, bytes)) or not isinstance(self.top_k, Sequence):
            raise ConfigurationError("evaluation.top_k must be a sequence")
        if not self.top_k:
            raise ConfigurationError("evaluation.top_k must not be empty")
        if any(
            isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in self.top_k
        ):
            raise ConfigurationError("evaluation.top_k must contain positive integers")
        if len(set(self.top_k)) != len(self.top_k):
            raise ConfigurationError("evaluation.top_k must not contain duplicates")
        if isinstance(self.metrics, (str, bytes)) or not isinstance(
            self.metrics, Sequence
        ):
            raise ConfigurationError("evaluation.metrics must be a sequence")
        if any(
            not isinstance(metric, str) or not metric.strip() for metric in self.metrics
        ):
            raise ConfigurationError(
                "evaluation.metrics must contain non-empty strings"
            )
        if len(set(self.metrics)) != len(self.metrics):
            raise ConfigurationError("evaluation.metrics must not contain duplicates")
        if set(self.metrics) != set(SUPPORTED_METRICS):
            raise ConfigurationError(
                "evaluation.metrics must contain hit_rate, recall, precision, mrr, "
                "ndcg, and ap in v0.1"
            )
        if isinstance(self.repetitions, bool) or self.repetitions != 1:
            raise ConfigurationError("evaluation.repetitions must be 1 in v0.1")
        if isinstance(self.concurrency, bool) or self.concurrency != 1:
            raise ConfigurationError("evaluation.concurrency must be 1 in v0.1")
        object.__setattr__(self, "top_k", tuple(sorted(self.top_k)))
        object.__setattr__(self, "metrics", SUPPORTED_METRICS)


@dataclass(frozen=True)
class QualityGateConfig:
    """A validated but not-yet-executed quality gate."""

    retriever: str
    metric: str
    min_value: float | None = None
    max_value: float | None = None
    max_absolute_drop: float | None = None
    max_relative_drop: float | None = None

    def __post_init__(self) -> None:
        _non_empty(self.retriever, "quality_gates.retriever")
        _non_empty(self.metric, "quality_gates.metric")
        _quality_metric(self.metric)
        if all(
            value is None
            for value in (
                self.min_value,
                self.max_value,
                self.max_absolute_drop,
                self.max_relative_drop,
            )
        ):
            raise ConfigurationError(
                "quality_gates entries require at least one constraint"
            )
        if self.min_value is not None:
            object.__setattr__(
                self, "min_value", _finite(self.min_value, "quality_gates.min_value")
            )
        if self.max_value is not None:
            object.__setattr__(
                self, "max_value", _finite(self.max_value, "quality_gates.max_value")
            )
        for field_name in ("max_absolute_drop", "max_relative_drop"):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = _finite(value, f"quality_gates.{field_name}")
            if normalized < 0.0:
                raise ConfigurationError(
                    f"quality_gates.{field_name} must be non-negative"
                )
            object.__setattr__(self, field_name, normalized)
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ConfigurationError(
                "quality_gates.min_value must not exceed max_value"
            )


@dataclass(frozen=True)
class ReportConfig:
    """Report settings retained for future report execution."""

    output_dir: Path | None = None
    formats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.output_dir is not None:
            _path(self.output_dir, "report.output_dir")
        if isinstance(self.formats, (str, bytes)) or not isinstance(
            self.formats, Sequence
        ):
            raise ConfigurationError("report.formats must be a sequence")
        if any(
            not isinstance(value, str) or not value.strip() for value in self.formats
        ):
            raise ConfigurationError("report.formats must contain non-empty strings")
        if len(set(self.formats)) != len(self.formats):
            raise ConfigurationError("report.formats must not contain duplicates")
        unknown = sorted(set(self.formats) - set(SUPPORTED_REPORT_FORMATS))
        if unknown:
            raise ConfigurationError(
                f"report.formats contains unsupported values: {unknown!r}"
            )
        object.__setattr__(self, "formats", tuple(self.formats))


@dataclass(frozen=True)
class RetrievalConfig:
    """Complete schema-versioned Retrieval Lab configuration."""

    schema_version: Literal[1]
    corpus: CorpusConfig
    dataset: DatasetConfig
    retrievers: tuple[RetrieverConfig, ...]
    evaluation: EvaluationConfig
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    quality_gates: tuple[QualityGateConfig, ...] = ()
    report: ReportConfig = field(default_factory=ReportConfig)
    source_dir: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ConfigurationError("schema_version must be 1")
        if not isinstance(self.corpus, CorpusConfig):
            raise ConfigurationError("corpus must be a CorpusConfig")
        if not isinstance(self.dataset, DatasetConfig):
            raise ConfigurationError("dataset must be a DatasetConfig")
        if not isinstance(self.evaluation, EvaluationConfig):
            raise ConfigurationError("evaluation must be an EvaluationConfig")
        if not isinstance(self.experiment, ExperimentConfig):
            raise ConfigurationError("experiment must be an ExperimentConfig")
        if not isinstance(self.report, ReportConfig):
            raise ConfigurationError("report must be a ReportConfig")
        if isinstance(self.retrievers, (str, bytes)) or not isinstance(
            self.retrievers, Sequence
        ):
            raise ConfigurationError("retrievers must be a sequence")
        if not self.retrievers:
            raise ConfigurationError("retrievers must not be empty")
        if not all(
            isinstance(
                retriever,
                (
                    KeywordRetrieverConfig,
                    BM25RetrieverConfig,
                    DenseRetrieverConfig,
                    HybridRetrieverConfig,
                ),
            )
            for retriever in self.retrievers
        ):
            raise ConfigurationError("retrievers contains an invalid config")
        object.__setattr__(self, "retrievers", tuple(self.retrievers))
        names = [retriever.name for retriever in self.retrievers]
        if len(set(names)) != len(names):
            raise ConfigurationError("retrievers names must be unique")
        for retriever in self.retrievers:
            if retriever.name != retriever.type:
                raise ConfigurationError(
                    "built-in retriever names must equal their type in v0.1"
                )
        if isinstance(self.quality_gates, (str, bytes)) or not isinstance(
            self.quality_gates, Sequence
        ):
            raise ConfigurationError("quality_gates must be a sequence")
        if not all(isinstance(gate, QualityGateConfig) for gate in self.quality_gates):
            raise ConfigurationError("quality_gates contains an invalid config")
        object.__setattr__(self, "quality_gates", tuple(self.quality_gates))
        configured_names = set(names)
        max_k = max(self.evaluation.top_k)
        for retriever in self.retrievers:
            if not isinstance(retriever, HybridRetrieverConfig):
                continue
            if retriever.candidate_k < max_k:
                raise ConfigurationError(
                    "hybrid candidate_k must be at least max(evaluation.top_k)"
                )
            if retriever.name in retriever.sources:
                raise ConfigurationError("hybrid retriever cannot reference itself")
            for source in retriever.sources:
                if source not in configured_names:
                    raise ConfigurationError(
                        f"hybrid retriever references unknown source {source!r}"
                    )
        for gate in self.quality_gates:
            if gate.retriever not in configured_names:
                raise ConfigurationError(
                    f"quality gate references unknown retriever {gate.retriever!r}"
                )
            if gate.metric not in SUPPORTED_LATENCY_METRICS:
                _, raw_cutoff = gate.metric.split("@")
                cutoff = int(raw_cutoff)
                if cutoff not in self.evaluation.top_k:
                    raise ConfigurationError(
                        "quality gate cutoff must be present in evaluation.top_k"
                    )
        if self.source_dir is not None:
            _path(self.source_dir, "source_dir")

    @property
    def cache_dir(self) -> Path | None:
        """Return the configured cache path, if any."""

        return self.experiment.effective_cache_dir

    def normalized_settings(self) -> dict[str, JSONValue]:
        """Return JSON-compatible settings with no absolute filesystem paths."""

        base = self.source_dir
        return {
            "corpus": {
                "chunker": {
                    "overlap": self.corpus.chunker.overlap,
                    "size": self.corpus.chunker.size,
                    "type": self.corpus.chunker.type,
                },
                "include": list(self.corpus.include),
                "path": _display_path(self.corpus.path, base),
            },
            "dataset": {
                "format": self.dataset.format,
                "path": _display_path(self.dataset.path, base),
                "relevance_level": self.dataset.relevance_level,
            },
            "evaluation": {
                "concurrency": self.evaluation.concurrency,
                "metrics": list(self.evaluation.metrics),
                "repetitions": self.evaluation.repetitions,
                "top_k": list(self.evaluation.top_k),
            },
            "experiment": {
                "cache_dir": (
                    _display_path(self.experiment.cache_dir, base)
                    if self.experiment.cache_dir is not None
                    else None
                ),
                "name": self.experiment.name,
                "seed": self.experiment.seed,
                "workspace": (
                    _display_path(self.experiment.workspace, base)
                    if self.experiment.workspace is not None
                    else None
                ),
            },
            "quality_gates": [
                {
                    "max_value": gate.max_value,
                    "max_absolute_drop": gate.max_absolute_drop,
                    "max_relative_drop": gate.max_relative_drop,
                    "metric": gate.metric,
                    "min_value": gate.min_value,
                    "retriever": gate.retriever,
                }
                for gate in self.quality_gates
            ],
            "report": {
                "formats": list(self.report.formats),
                "output_dir": (
                    _display_path(self.report.output_dir, base)
                    if self.report.output_dir is not None
                    else None
                ),
            },
            "retrievers": [_retriever_settings(item) for item in self.retrievers],
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, JSONValue]:
        """Alias for :meth:`normalized_settings` used by public callers."""

        return self.normalized_settings()


def _display_path(path: Path, base: Path | None) -> str:
    if base is not None:
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            pass
    if path.is_absolute() or PureWindowsPath(path.as_posix()).is_absolute():
        return f"<absolute>/{path.name}"
    return path.as_posix()


def _retriever_settings(value: RetrieverConfig) -> dict[str, JSONValue]:
    if isinstance(value, KeywordRetrieverConfig):
        return {"name": value.name, "type": value.type}
    if isinstance(value, BM25RetrieverConfig):
        return {
            "b": value.b,
            "k1": value.k1,
            "name": value.name,
            "tokenizer": value.tokenizer,
            "type": value.type,
        }
    if isinstance(value, DenseRetrieverConfig):
        return {
            "batch_size": value.batch_size,
            "document_prompt": value.document_prompt,
            "model": value.model,
            "model_revision": value.model_revision,
            "name": value.name,
            "normalize_embeddings": value.normalize_embeddings,
            "query_prompt": value.query_prompt,
            "type": value.type,
        }
    return {
        "candidate_k": value.candidate_k,
        "fusion": value.fusion,
        "name": value.name,
        "rrf_k": value.rrf_k,
        "sources": list(value.sources),
        "type": value.type,
    }


__all__ = [
    "SUPPORTED_LATENCY_METRICS",
    "SUPPORTED_METRICS",
    "SUPPORTED_REPORT_FORMATS",
    "BM25RetrieverConfig",
    "ChunkerConfig",
    "CorpusConfig",
    "DatasetConfig",
    "DenseRetrieverConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "HybridRetrieverConfig",
    "KeywordRetrieverConfig",
    "QualityGateConfig",
    "ReportConfig",
    "RetrievalConfig",
    "RetrieverConfig",
]
