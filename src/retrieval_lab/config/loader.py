"""Strict, safe YAML loading for the Retrieval Lab configuration schema."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Literal, cast

import yaml

from retrieval_lab.config.models import (
    SUPPORTED_METRICS,
    SUPPORTED_REPORT_FORMATS,
    BM25RetrieverConfig,
    ChunkerConfig,
    CorpusConfig,
    DatasetConfig,
    DenseRetrieverConfig,
    EvaluationConfig,
    ExperimentConfig,
    HybridRetrieverConfig,
    KeywordRetrieverConfig,
    QualityGateConfig,
    ReportConfig,
    RetrievalConfig,
    RetrieverConfig,
)
from retrieval_lab.exceptions import ConfigurationError


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader with duplicate mapping-key detection."""


def _construct_mapping(
    loader: yaml.SafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


class _Errors:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, path: str, message: str, hint: str | None = None) -> None:
        suffix = f" Hint: {hint}" if hint else ""
        self.items.append(
            f"{path}: {message}.{suffix}" if suffix else f"{path}: {message}."
        )

    def raise_if_any(self) -> None:
        if self.items:
            raise ConfigurationError(
                "invalid Retrieval Lab configuration:\n"
                + "\n".join(f"- {item}" for item in self.items)
            )


def load_config(path: str | Path) -> RetrievalConfig:
    """Load and validate a schema-versioned YAML configuration.

    Only ``yaml.SafeLoader`` behaviour is used.  Paths are resolved relative
    to the configuration file and no environment-variable or template
    expansion is performed.
    """

    try:
        config_path = Path(path)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("config path must be a valid path") from exc
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config_path = config_path.resolve()
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"cannot read configuration {config_path}: {exc}"
        ) from exc
    try:
        raw = yaml.load(text, Loader=_StrictSafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {config_path}: {exc}") from exc

    errors = _Errors()
    root = _mapping(
        raw,
        "config",
        {
            "schema_version",
            "experiment",
            "cache_dir",
            "corpus",
            "dataset",
            "retrievers",
            "evaluation",
            "quality_gates",
            "report",
        },
        errors,
    )
    schema_version = _int(root.get("schema_version"), "schema_version", errors)
    if "schema_version" not in root:
        errors.add("schema_version", "field is required", "set schema_version: 1")
    elif schema_version != 1:
        errors.add("schema_version", "must be exactly 1", "set schema_version: 1")

    base = config_path.parent
    experiment = _parse_experiment(root.get("experiment", {}), base, errors)
    root_cache = _path(root.get("cache_dir"), "cache_dir", base, errors, optional=True)
    if root_cache is not None and experiment.cache_dir is not None:
        errors.add("cache_dir", "duplicates experiment.cache_dir", "keep one setting")
    if root_cache is not None and experiment.cache_dir is None:
        experiment = ExperimentConfig(
            name=experiment.name,
            seed=experiment.seed,
            workspace=experiment.workspace,
            cache_dir=root_cache,
        )
    corpus = _parse_corpus(root.get("corpus"), base, errors)
    dataset = _parse_dataset(root.get("dataset"), base, errors)
    evaluation = _parse_evaluation(root.get("evaluation", {}), errors)
    retrievers = _parse_retrievers(root.get("retrievers"), errors)
    quality_gates = _parse_quality_gates(root.get("quality_gates", []), errors)
    report = _parse_report(root.get("report", {}), base, errors)
    _validate_retriever_graph(retrievers, evaluation, errors)
    retriever_names = {item.name for item in retrievers}
    for index, gate in enumerate(quality_gates):
        if gate.retriever not in retriever_names:
            errors.add(
                f"quality_gates[{index}].retriever",
                f"unknown retriever {gate.retriever!r}",
                "reference a configured retriever",
            )
    errors.raise_if_any()
    # All required values have been checked above; the assertions keep the
    # typed construction below explicit for static type checkers.
    assert corpus is not None and dataset is not None
    return RetrievalConfig(
        schema_version=1,
        corpus=corpus,
        dataset=dataset,
        retrievers=tuple(retrievers),
        evaluation=evaluation,
        experiment=experiment,
        quality_gates=tuple(quality_gates),
        report=report,
        source_dir=base,
    )


def _mapping(
    value: object,
    path: str,
    allowed: set[str],
    errors: _Errors,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        errors.add(path, "must be a mapping", "provide an object with named fields")
        return {}
    result = dict(value)
    for key in result:
        if not isinstance(key, str):
            errors.add(path, "field names must be strings")
        elif key not in allowed:
            errors.add(
                f"{path}.{key}", "unknown field", "remove it or use a documented field"
            )
    return {key: item for key, item in result.items() if isinstance(key, str)}


def _str(
    value: object, path: str, errors: _Errors, *, optional: bool = False
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        errors.add(path, "must be a non-empty string")
        return None
    return value


def _text(value: object, path: str, errors: _Errors, *, default: str) -> str:
    """Read an optional string field where an empty string is meaningful."""

    if value is None:
        return default
    if not isinstance(value, str):
        errors.add(path, "must be a string")
        return default
    return value


def _bool(value: object, path: str, errors: _Errors, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        errors.add(path, "must be boolean")
        return default
    return value


def _int(
    value: object, path: str, errors: _Errors, *, default: int | None = None
) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        errors.add(path, "must be an integer")
        return default
    return value


def _number(
    value: object, path: str, errors: _Errors, *, default: float | None = None
) -> float | None:
    if value is None:
        return default
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        errors.add(path, "must be a finite number")
        return default
    return float(value)


def _path(
    value: object,
    path: str,
    base: Path,
    errors: _Errors,
    *,
    optional: bool = False,
) -> Path | None:
    text = _str(value, path, errors, optional=optional)
    if text is None:
        return None
    normalized = text.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute() or PureWindowsPath(normalized).is_absolute():
        return candidate
    return (base / candidate).resolve()


def _parse_experiment(value: object, base: Path, errors: _Errors) -> ExperimentConfig:
    data = _mapping(
        value, "experiment", {"name", "seed", "workspace", "cache_dir"}, errors
    )
    seed = _int(data.get("seed"), "experiment.seed", errors, default=42)
    workspace = _path(
        data.get("workspace"), "experiment.workspace", base, errors, optional=True
    )
    cache_dir = _path(
        data.get("cache_dir"), "experiment.cache_dir", base, errors, optional=True
    )
    try:
        return ExperimentConfig(
            name=_str(data.get("name"), "experiment.name", errors, optional=True),
            seed=seed if seed is not None else 42,
            workspace=workspace,
            cache_dir=cache_dir,
        )
    except ConfigurationError as exc:
        errors.add("experiment", str(exc), "fix the experiment fields")
        return ExperimentConfig()


def _parse_corpus(value: object, base: Path, errors: _Errors) -> CorpusConfig | None:
    data = _mapping(value, "corpus", {"path", "include", "chunker"}, errors)
    path = _path(data.get("path"), "corpus.path", base, errors)
    chunk_data = _mapping(
        data.get("chunker", {}),
        "corpus.chunker",
        {"type", "size", "overlap"},
        errors,
    )
    chunk_type = (
        _str(chunk_data.get("type"), "corpus.chunker.type", errors, optional=True)
        or "recursive_characters"
    )
    size_value = _int(
        chunk_data.get("size"), "corpus.chunker.size", errors, default=512
    )
    overlap_value = _int(
        chunk_data.get("overlap"), "corpus.chunker.overlap", errors, default=64
    )
    size = 512 if size_value is None else size_value
    overlap = 64 if overlap_value is None else overlap_value
    try:
        chunker = ChunkerConfig(
            type=cast(Literal["recursive_characters"], chunk_type),
            size=size,
            overlap=overlap,
        )
    except ConfigurationError as exc:
        errors.add(
            "corpus.chunker",
            str(exc),
            "use recursive_characters with 0 <= overlap < size",
        )
        chunker = ChunkerConfig()
    include_value = data.get("include", [])
    include: tuple[str, ...] = ()
    if include_value is not None:
        if not isinstance(include_value, Sequence) or isinstance(
            include_value, (str, bytes)
        ):
            errors.add("corpus.include", "must be a list of glob strings")
        else:
            values = [_str(item, "corpus.include[]", errors) for item in include_value]
            include = tuple(item for item in values if item is not None)
    if path is None:
        return None
    return CorpusConfig(path=path, include=include, chunker=chunker)


def _parse_dataset(value: object, base: Path, errors: _Errors) -> DatasetConfig | None:
    data = _mapping(value, "dataset", {"path", "format", "relevance_level"}, errors)
    path = _path(data.get("path"), "dataset.path", base, errors)
    format_value = (
        _str(data.get("format"), "dataset.format", errors, optional=True)
        or "native_jsonl"
    )
    level = (
        _str(
            data.get("relevance_level"),
            "dataset.relevance_level",
            errors,
            optional=True,
        )
        or "document"
    )
    if format_value != "native_jsonl":
        errors.add("dataset.format", "must be 'native_jsonl'")
        format_value = "native_jsonl"
    if level not in ("document", "chunk"):
        errors.add("dataset.relevance_level", "must be 'document' or 'chunk'")
        level = "document"
    if path is None:
        return None
    return DatasetConfig(
        path=path,
        format=cast(Literal["native_jsonl"], format_value),
        relevance_level=cast(Literal["document", "chunk"], level),
    )


def _parse_evaluation(value: object, errors: _Errors) -> EvaluationConfig:
    data = _mapping(
        value, "evaluation", {"top_k", "metrics", "repetitions", "concurrency"}, errors
    )
    top_k_value = data.get("top_k", [1, 3, 5, 10])
    top_k: tuple[int, ...] = (1, 3, 5, 10)
    if not isinstance(top_k_value, Sequence) or isinstance(top_k_value, (str, bytes)):
        errors.add("evaluation.top_k", "must be a list of positive integers")
    else:
        parsed = [
            _int(item, f"evaluation.top_k[{index}]", errors)
            for index, item in enumerate(top_k_value)
        ]
        top_k = tuple(item for item in parsed if item is not None)
        for index, cutoff in enumerate(parsed):
            if cutoff is not None and cutoff <= 0:
                errors.add(f"evaluation.top_k[{index}]", "must be a positive integer")
        if len(set(top_k)) != len(top_k):
            errors.add("evaluation.top_k", "must not contain duplicates")
    metrics_value = data.get(
        "metrics", ["hit_rate", "recall", "precision", "mrr", "ndcg", "ap"]
    )
    metrics: tuple[str, ...] = ()
    if not isinstance(metrics_value, Sequence) or isinstance(
        metrics_value, (str, bytes)
    ):
        errors.add("evaluation.metrics", "must be a list of strings")
    else:
        parsed_metrics = [
            _str(item, f"evaluation.metrics[{index}]", errors)
            for index, item in enumerate(metrics_value)
        ]
        metrics = tuple(item for item in parsed_metrics if item is not None)
        for index, metric in enumerate(metrics):
            if metric not in SUPPORTED_METRICS:
                errors.add(
                    f"evaluation.metrics[{index}]",
                    f"unsupported metric {metric!r}",
                    "use hit_rate, recall, precision, mrr, ndcg, or ap",
                )
        if len(set(metrics)) != len(metrics):
            errors.add("evaluation.metrics", "must not contain duplicates")
        if set(metrics) != set(SUPPORTED_METRICS):
            errors.add(
                "evaluation.metrics",
                "must contain all six v0.1 metrics",
                "use hit_rate, recall, precision, mrr, ndcg, and ap",
            )
    repetitions_value = _int(
        data.get("repetitions"), "evaluation.repetitions", errors, default=1
    )
    concurrency_value = _int(
        data.get("concurrency"), "evaluation.concurrency", errors, default=1
    )
    repetitions = 1 if repetitions_value is None else repetitions_value
    concurrency = 1 if concurrency_value is None else concurrency_value
    if repetitions != 1:
        errors.add(
            "evaluation.repetitions",
            "must be 1 in v0.1",
            "remove the field or set it to 1",
        )
        repetitions = 1
    if concurrency != 1:
        errors.add(
            "evaluation.concurrency",
            "must be 1 in v0.1",
            "remove the field or set it to 1",
        )
        concurrency = 1
    try:
        return EvaluationConfig(
            top_k=top_k,
            metrics=SUPPORTED_METRICS,
            repetitions=repetitions,
            concurrency=concurrency,
        )
    except ConfigurationError as exc:
        errors.add("evaluation", str(exc), "check positive unique top_k values")
        return EvaluationConfig()


def _parse_retrievers(value: object, errors: _Errors) -> list[RetrieverConfig]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        errors.add(
            "retrievers", "must be a list", "add at least one built-in retriever"
        )
        return []
    parsed: list[RetrieverConfig] = []
    names_seen: set[str] = set()
    for index, item in enumerate(value):
        path = f"retrievers[{index}]"
        data = _mapping(
            item,
            path,
            {
                "name",
                "type",
                "tokenizer",
                "k1",
                "b",
                "model",
                "model_revision",
                "normalize_embeddings",
                "query_prompt",
                "document_prompt",
                "batch_size",
                "sources",
                "fusion",
                "rrf_k",
                "candidate_k",
            },
            errors,
        )
        name = _str(data.get("name"), f"{path}.name", errors) or f"invalid_{index}"
        kind = _str(data.get("type"), f"{path}.type", errors) or "keyword"
        allowed_by_kind = {
            "keyword": {"name", "type"},
            "bm25": {"name", "type", "tokenizer", "k1", "b"},
            "dense": {
                "name",
                "type",
                "model",
                "model_revision",
                "normalize_embeddings",
                "query_prompt",
                "document_prompt",
                "batch_size",
            },
            "hybrid": {
                "name",
                "type",
                "sources",
                "fusion",
                "rrf_k",
                "candidate_k",
            },
        }
        if kind in allowed_by_kind:
            for field_name in sorted(set(data) - allowed_by_kind[kind]):
                errors.add(
                    f"{path}.{field_name}",
                    f"field is not valid for a {kind} retriever",
                    "remove the field",
                )
        if name != kind:
            errors.add(
                f"{path}.name",
                f"must equal type {kind!r} for built-in retrievers",
                "use canonical names keyword, bm25, dense, or hybrid",
            )
        if name in names_seen:
            errors.add(
                f"{path}.name",
                f"duplicate retriever name {name!r}",
                "use each canonical built-in name once",
            )
        names_seen.add(name)
        try:
            if kind == "keyword":
                parsed.append(KeywordRetrieverConfig(name=name))
            elif kind == "bm25":
                tokenizer = (
                    _str(
                        data.get("tokenizer"),
                        f"{path}.tokenizer",
                        errors,
                        optional=True,
                    )
                    or "default"
                )
                if tokenizer != "default":
                    errors.add(
                        f"{path}.tokenizer",
                        f"unsupported tokenizer {tokenizer!r}",
                        "use default; custom tokenizers are available through the "
                        "Python API",
                    )
                k1_value = _number(data.get("k1"), f"{path}.k1", errors, default=1.5)
                k1 = 1.5 if k1_value is None else k1_value
                b = _number(data.get("b"), f"{path}.b", errors, default=0.75)
                if k1 <= 0.0:
                    errors.add(f"{path}.k1", "must be greater than zero")
                    k1 = 1.5
                normalized_b = 0.75 if b is None else b
                if not 0.0 <= normalized_b <= 1.0:
                    errors.add(f"{path}.b", "must be between zero and one")
                    normalized_b = 0.75
                parsed.append(
                    BM25RetrieverConfig(
                        name=name,
                        tokenizer="default",
                        k1=k1,
                        b=normalized_b,
                    )
                )
            elif kind == "dense":
                model = (
                    _str(data.get("model"), f"{path}.model", errors, optional=True)
                    or "intfloat/multilingual-e5-small"
                )
                revision = _str(
                    data.get("model_revision"),
                    f"{path}.model_revision",
                    errors,
                    optional=True,
                )
                batch_value = _int(
                    data.get("batch_size"), f"{path}.batch_size", errors, default=32
                )
                batch = 32 if batch_value is None else batch_value
                if batch <= 0:
                    errors.add(f"{path}.batch_size", "must be a positive integer")
                    batch = 32
                query_prompt = _text(
                    data.get("query_prompt"),
                    f"{path}.query_prompt",
                    errors,
                    default="query: ",
                )
                document_prompt = _text(
                    data.get("document_prompt"),
                    f"{path}.document_prompt",
                    errors,
                    default="passage: ",
                )
                parsed.append(
                    DenseRetrieverConfig(
                        name=name,
                        model=model,
                        model_revision=revision,
                        normalize_embeddings=_bool(
                            data.get("normalize_embeddings"),
                            f"{path}.normalize_embeddings",
                            errors,
                            True,
                        ),
                        query_prompt=query_prompt,
                        document_prompt=document_prompt,
                        batch_size=batch,
                    )
                )
            elif kind == "hybrid":
                source_value = data.get("sources", [])
                if not isinstance(source_value, Sequence) or isinstance(
                    source_value, (str, bytes)
                ):
                    errors.add(f"{path}.sources", "must be a list of retriever names")
                    sources: tuple[str, ...] = ()
                else:
                    source_items = [
                        _str(item, f"{path}.sources[]", errors) for item in source_value
                    ]
                    sources = tuple(item for item in source_items if item is not None)
                    if len(set(sources)) != len(sources):
                        errors.add(
                            f"{path}.sources",
                            "must not contain duplicate names",
                            "list each source once",
                        )
                        sources = tuple(dict.fromkeys(sources))
                fusion = (
                    _str(data.get("fusion"), f"{path}.fusion", errors, optional=True)
                    or "rrf"
                )
                rrf_value = _int(data.get("rrf_k"), f"{path}.rrf_k", errors, default=60)
                candidate_value = _int(
                    data.get("candidate_k"), f"{path}.candidate_k", errors, default=100
                )
                rrf_k = 60 if rrf_value is None else rrf_value
                candidate_k = 100 if candidate_value is None else candidate_value
                if fusion != "rrf":
                    errors.add(f"{path}.fusion", "must be 'rrf'", "set fusion: rrf")
                    fusion = "rrf"
                if rrf_k <= 0:
                    errors.add(f"{path}.rrf_k", "must be a positive integer")
                    rrf_k = 60
                if candidate_k <= 0:
                    errors.add(f"{path}.candidate_k", "must be a positive integer")
                    candidate_k = 100
                parsed.append(
                    HybridRetrieverConfig(
                        name=name,
                        sources=sources,
                        fusion=cast(Literal["rrf"], fusion),
                        rrf_k=rrf_k,
                        candidate_k=candidate_k,
                    )
                )
            else:
                errors.add(
                    f"{path}.type",
                    f"unsupported built-in type {kind!r}",
                    "use keyword, bm25, dense, or hybrid",
                )
        except ConfigurationError as exc:
            errors.add(path, str(exc), "correct the retriever fields")
    return parsed


def _parse_quality_gates(value: object, errors: _Errors) -> list[QualityGateConfig]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        errors.add("quality_gates", "must be a list")
        return []
    result: list[QualityGateConfig] = []
    for index, item in enumerate(value):
        path = f"quality_gates[{index}]"
        data = _mapping(
            item, path, {"retriever", "metric", "min_value", "max_value"}, errors
        )
        retriever = (
            _str(data.get("retriever"), f"{path}.retriever", errors) or "invalid"
        )
        metric = _str(data.get("metric"), f"{path}.metric", errors) or "invalid"
        minimum = _number(data.get("min_value"), f"{path}.min_value", errors)
        maximum = _number(data.get("max_value"), f"{path}.max_value", errors)
        try:
            result.append(
                QualityGateConfig(
                    retriever=retriever,
                    metric=metric,
                    min_value=minimum,
                    max_value=maximum,
                )
            )
        except ConfigurationError as exc:
            errors.add(path, str(exc), "set min_value or max_value")
    return result


def _parse_report(value: object, base: Path, errors: _Errors) -> ReportConfig:
    data = _mapping(value, "report", {"output_dir", "formats"}, errors)
    output_dir = _path(
        data.get("output_dir"), "report.output_dir", base, errors, optional=True
    )
    formats_value = data.get("formats", [])
    formats: tuple[str, ...] = ()
    if not isinstance(formats_value, Sequence) or isinstance(
        formats_value, (str, bytes)
    ):
        errors.add("report.formats", "must be a list of strings")
    else:
        values = [
            _str(item, f"report.formats[{index}]", errors)
            for index, item in enumerate(formats_value)
        ]
        formats = tuple(item for item in values if item is not None)
        if len(set(formats)) != len(formats):
            errors.add("report.formats", "must not contain duplicates")
        for index, format_name in enumerate(formats):
            if format_name not in SUPPORTED_REPORT_FORMATS:
                errors.add(
                    f"report.formats[{index}]",
                    f"unsupported format {format_name!r}",
                    "use json, csv, or html",
                )
    safe_formats = tuple(
        dict.fromkeys(
            format_name
            for format_name in formats
            if format_name in SUPPORTED_REPORT_FORMATS
        )
    )
    return ReportConfig(output_dir=output_dir, formats=safe_formats)


def _validate_retriever_graph(
    retrievers: Sequence[RetrieverConfig],
    evaluation: EvaluationConfig,
    errors: _Errors,
) -> None:
    names = {retriever.name for retriever in retrievers}
    max_k = max(evaluation.top_k, default=0)
    for index, retriever in enumerate(retrievers):
        if not isinstance(retriever, HybridRetrieverConfig):
            continue
        path = f"retrievers[{index}]"
        if retriever.candidate_k < max_k:
            errors.add(
                f"{path}.candidate_k",
                f"must be at least max(evaluation.top_k) ({max_k})",
                "increase candidate_k or lower top_k",
            )
        for source_index, source in enumerate(retriever.sources):
            source_path = f"{path}.sources[{source_index}]"
            if source not in names:
                errors.add(
                    source_path,
                    f"unknown retriever {source!r}",
                    "name an earlier non-hybrid retriever",
                )
            if source == retriever.name:
                errors.add(source_path, "cannot reference itself")


__all__ = ["load_config"]
