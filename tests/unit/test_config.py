from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from retrieval_lab import (
    BM25RetrieverConfig,
    ChunkerConfig,
    ConfigurationError,
    CorpusConfig,
    DatasetConfig,
    DenseRetrieverConfig,
    EvaluationConfig,
    EvaluationDataset,
    EvaluationRunner,
    ExperimentConfig,
    FixedSizeChunker,
    HybridRetrieverConfig,
    KeywordRetriever,
    KeywordRetrieverConfig,
    QualityGateConfig,
    ReportConfig,
    RetrievalConfig,
    load_config,
    load_documents,
)


def _write_fixture(tmp_path: Path, *, extra: str = "") -> Path:
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "a.txt").write_text("one relevant", encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text(
        '{"query_id":"q1","query":"one","relevant":[{"id":"a.txt","relevance":1}]}\n',
        encoding="utf-8",
    )
    lines = [
        "schema_version: 1",
        "experiment:",
        "  seed: 42",
        "  cache_dir: cache",
        "corpus:",
        "  path: docs",
        '  include: ["**/*.txt"]',
        "  chunker:",
        "    type: recursive_characters",
        "    size: 32",
        "    overlap: 4",
        "dataset:",
        "  path: eval.jsonl",
        "  format: native_jsonl",
        "  relevance_level: document",
        "retrievers:",
        "  - name: keyword",
        "    type: keyword",
        "evaluation:",
        "  top_k: [1, 3]",
        "  metrics: [hit_rate, recall, precision, mrr, ndcg, ap]",
        "  repetitions: 1",
        "  concurrency: 1",
        "quality_gates:",
        "  - retriever: keyword",
        "    metric: recall@1",
        "    min_value: 0.5",
        "report:",
        "  output_dir: reports",
        "  formats: [json, csv, html]",
    ]
    config = tmp_path / "experiment.yaml"
    config.write_text("\n".join(lines) + "\n" + extra, encoding="utf-8")
    return config


def test_load_config_and_yaml_runner_are_deterministic(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    config = load_config(config_path)

    assert isinstance(config, RetrievalConfig)
    assert config.corpus.path == (tmp_path / "docs").resolve()
    assert config.dataset.path == (tmp_path / "eval.jsonl").resolve()
    assert config.normalized_settings()["corpus"] == {
        "chunker": {"overlap": 4, "size": 32, "type": "recursive_characters"},
        "include": ["**/*.txt"],
        "path": "docs",
    }
    result = EvaluationRunner.from_config(config_path).run()
    typed_result = EvaluationRunner.from_config(config).run()
    direct = EvaluationRunner(
        documents=load_documents(tmp_path / "docs"),
        dataset=EvaluationDataset.from_jsonl(tmp_path / "eval.jsonl"),
        retrievers=[KeywordRetriever()],
        top_k=[1, 3],
        chunker=FixedSizeChunker(size=32, overlap=4),
        seed=42,
    ).run()
    assert result.run_id == direct.run_id
    assert typed_result.run_id == result.run_id
    assert typed_result.metrics == result.metrics
    assert result.metrics == direct.metrics
    assert result.manifest["config"]["experiment"]["cache_dir"] == "cache"  # type: ignore[index]
    assert str(tmp_path) not in result.to_json()


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ("unknown: true\n", "config.unknown"),
        ("schema_version: 2\n", "schema_version"),
        ("corpus:\n  typo: true\n", "corpus.typo"),
    ],
)
def test_config_rejects_schema_and_unknown_fields(
    tmp_path: Path, extra: str, expected: str
) -> None:
    path = _write_fixture(tmp_path)
    if extra.startswith("schema_version"):
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "schema_version: 1", "schema_version: 2"
            ),
            encoding="utf-8",
        )
    elif extra.startswith("corpus"):
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "  path: docs", "  path: docs\n  typo: true"
            ),
            encoding="utf-8",
        )
    else:
        path.write_text(path.read_text(encoding="utf-8") + extra, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=expected):
        load_config(path)


def test_config_rejects_tags_and_does_not_expand_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_fixture(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "path: docs", "path: !!python/object/apply:os.system ['x']"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="invalid YAML"):
        load_config(path)

    path = _write_fixture(tmp_path)
    monkeypatch.setenv("CORPUS_DIR", str(tmp_path / "docs"))
    text = path.read_text(encoding="utf-8").replace("path: docs", "path: ${CORPUS_DIR}")
    path.write_text(text, encoding="utf-8")
    loaded = load_config(path)
    assert loaded.corpus.path.name == "${CORPUS_DIR}"
    assert loaded.corpus.path != tmp_path / "docs"


def test_config_reports_independent_retriever_errors(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    text = path.read_text(encoding="utf-8")
    start = text.index("retrievers:\n")
    end = text.index("evaluation:\n")
    replacement = (
        "retrievers:\n"
        "  - name: bm25\n"
        "    type: bm25\n"
        "    b: 2\n"
        "  - name: keyword\n"
        "    type: keyword\n"
        "  - name: hybrid\n"
        "    type: hybrid\n"
        "    sources: [bm25, bm25, keyword]\n"
        "    candidate_k: 1\n"
    )
    path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
    with pytest.raises(ConfigurationError) as captured:
        load_config(path)
    message = str(captured.value)
    assert "retrievers[0].b" in message
    assert "retrievers[2].candidate_k" in message
    assert "retrievers[2].sources" in message


def test_config_supports_windows_relative_path_spelling(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    text = path.read_text(encoding="utf-8").replace("path: docs", "path: docs\\docs")
    path.write_text(text, encoding="utf-8")
    loaded = load_config(path)
    assert loaded.corpus.path == (tmp_path / "docs" / "docs").resolve()


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("size: 32", "size: 0", "corpus.chunker.size"),
        ("top_k: [1, 3]", "top_k: [0]", r"evaluation.top_k\[0\]"),
        ("seed: 42", "seed: -1", "experiment.seed"),
        ("type: keyword", "type: keyword\n    batch_size: 2", "batch_size"),
        ("formats: [json, csv, html]", "formats: [xml]", "report.formats"),
        (
            "metrics: [hit_rate, recall, precision, mrr, ndcg, ap]",
            "metrics: [recall]",
            "evaluation.metrics",
        ),
    ],
)
def test_config_rejects_invalid_boundaries_and_type_specific_fields(
    tmp_path: Path, old: str, new: str, expected: str
) -> None:
    path = _write_fixture(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match=expected):
        load_config(path)


def test_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8") + "schema_version: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="duplicate key 'schema_version'"):
        load_config(path)


def test_typed_config_applies_workspace_cache_and_cross_validation(
    tmp_path: Path,
) -> None:
    corpus = CorpusConfig(path=tmp_path / "docs", chunker=ChunkerConfig())
    dataset = DatasetConfig(path=tmp_path / "eval.jsonl")
    evaluation = EvaluationConfig(top_k=(1, 3))
    config = RetrievalConfig(
        schema_version=1,
        corpus=corpus,
        dataset=dataset,
        retrievers=(KeywordRetrieverConfig(name="keyword"),),
        evaluation=evaluation,
        experiment=ExperimentConfig(workspace=tmp_path / ".retrieval-lab"),
        report=ReportConfig(formats=("json",)),
        source_dir=tmp_path,
    )

    assert config.cache_dir == tmp_path / ".retrieval-lab" / "cache"
    assert config.normalized_settings()["experiment"] == {
        "cache_dir": None,
        "name": None,
        "seed": 42,
        "workspace": ".retrieval-lab",
    }

    with pytest.raises(ConfigurationError, match="candidate_k"):
        RetrievalConfig(
            schema_version=1,
            corpus=corpus,
            dataset=dataset,
            retrievers=(
                KeywordRetrieverConfig(name="keyword"),
                BM25RetrieverConfig(name="bm25"),
                HybridRetrieverConfig(
                    name="hybrid", sources=("keyword", "bm25"), candidate_k=1
                ),
            ),
            evaluation=EvaluationConfig(top_k=(3,)),
        )


def test_hybrid_sources_may_follow_hybrid_in_yaml(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    text = path.read_text(encoding="utf-8")
    start = text.index("retrievers:\n")
    end = text.index("evaluation:\n")
    replacement = (
        "retrievers:\n"
        "  - name: hybrid\n"
        "    type: hybrid\n"
        "    sources: [keyword, bm25]\n"
        "    candidate_k: 3\n"
        "  - name: keyword\n"
        "    type: keyword\n"
        "  - name: bm25\n"
        "    type: bm25\n"
    )
    path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")

    result = EvaluationRunner.from_config(path).run()

    assert set(result.metrics) == {"hybrid", "keyword", "bm25"}


def test_dense_config_is_lazy_about_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_fixture(tmp_path)
    text = path.read_text(encoding="utf-8")
    start = text.index("retrievers:\n")
    end = text.index("evaluation:\n")
    dense = "retrievers:\n  - name: dense\n    type: dense\n"
    updated = (text[:start] + dense + text[end:]).replace(
        "retriever: keyword", "retriever: dense"
    )
    path.write_text(updated, encoding="utf-8")

    def unexpected_import(name: str) -> object:
        raise AssertionError(f"unexpected optional import: {name}")

    monkeypatch.setattr(
        "retrieval_lab.retrievers.dense.import_module", unexpected_import
    )

    runner = EvaluationRunner.from_config(path)

    assert isinstance(runner, EvaluationRunner)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EvaluationConfig(metrics=("recall",)),
        lambda: EvaluationConfig(repetitions=True),
        lambda: ReportConfig(formats=("xml",)),
    ],
)
def test_typed_config_models_reject_unsupported_values(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(ConfigurationError):
        factory()


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda: ChunkerConfig(type="other"), "chunker.type"),  # type: ignore[arg-type]
        (lambda: ChunkerConfig(size=0), "chunker.size"),
        (lambda: ChunkerConfig(size=2, overlap=2), "chunker.overlap"),
        (lambda: ExperimentConfig(name=" "), "experiment.name"),
        (lambda: ExperimentConfig(seed=True), "experiment.seed"),
        (lambda: ExperimentConfig(workspace="cache"), "workspace"),  # type: ignore[arg-type]
        (lambda: CorpusConfig(path=Path("docs"), include="*.md"), "include"),  # type: ignore[arg-type]
        (lambda: CorpusConfig(path=Path("docs"), include=("",)), "include"),
        (lambda: CorpusConfig(path=Path("docs"), chunker=object()), "chunker"),  # type: ignore[arg-type]
        (lambda: DatasetConfig(path=Path("data"), format="csv"), "format"),  # type: ignore[arg-type]
        (
            lambda: DatasetConfig(path=Path("data"), relevance_level="other"),  # type: ignore[arg-type]
            "relevance_level",
        ),
        (lambda: KeywordRetrieverConfig(name=""), "retriever.name"),
        (
            lambda: KeywordRetrieverConfig(name="keyword", type="bm25"),  # type: ignore[arg-type]
            "retriever.type",
        ),
        (lambda: BM25RetrieverConfig(name="bm25", tokenizer="unicode"), "tokenizer"),  # type: ignore[arg-type]
        (lambda: BM25RetrieverConfig(name="bm25", k1=True), "finite"),
        (lambda: BM25RetrieverConfig(name="bm25", k1=float("nan")), "finite"),
        (lambda: BM25RetrieverConfig(name="bm25", k1=0), "retriever.k1"),
        (lambda: BM25RetrieverConfig(name="bm25", b=2), "retriever.b"),
        (lambda: DenseRetrieverConfig(name="dense", model=""), "retriever.model"),
        (
            lambda: DenseRetrieverConfig(name="dense", model_revision=""),
            "model_revision",
        ),
        (
            lambda: DenseRetrieverConfig(name="dense", normalize_embeddings=1),  # type: ignore[arg-type]
            "normalize_embeddings",
        ),
        (lambda: DenseRetrieverConfig(name="dense", batch_size=0), "batch_size"),
        (
            lambda: DenseRetrieverConfig(name="dense", query_prompt=1),  # type: ignore[arg-type]
            "query_prompt",
        ),
        (
            lambda: DenseRetrieverConfig(name="dense", document_prompt=1),  # type: ignore[arg-type]
            "document_prompt",
        ),
        (
            lambda: HybridRetrieverConfig(name="hybrid", sources="bm25"),  # type: ignore[arg-type]
            "sources",
        ),
        (
            lambda: HybridRetrieverConfig(name="hybrid", sources=("bm25",)),
            "at least two",
        ),
        (
            lambda: HybridRetrieverConfig(name="hybrid", sources=("bm25", "bm25")),
            "duplicates",
        ),
        (
            lambda: HybridRetrieverConfig(
                name="hybrid",
                sources=("bm25", "dense"),
                fusion="sum",  # type: ignore[arg-type]
            ),
            "fusion",
        ),
        (
            lambda: HybridRetrieverConfig(
                name="hybrid", sources=("bm25", "dense"), rrf_k=0
            ),
            "rrf_k",
        ),
        (
            lambda: HybridRetrieverConfig(
                name="hybrid", sources=("bm25", "dense"), candidate_k=0
            ),
            "candidate_k",
        ),
        (lambda: EvaluationConfig(top_k=()), "top_k"),
        (lambda: EvaluationConfig(top_k=(1, 1)), "duplicates"),
        (lambda: EvaluationConfig(metrics="recall"), "metrics"),  # type: ignore[arg-type]
        (lambda: EvaluationConfig(concurrency=2), "concurrency"),
        (
            lambda: QualityGateConfig(retriever="keyword", metric="recall@1"),
            "require",
        ),
        (
            lambda: QualityGateConfig(retriever="", metric="recall@1", min_value=0.5),
            "retriever",
        ),
        (
            lambda: QualityGateConfig(retriever="keyword", metric="", min_value=0.5),
            "metric",
        ),
        (
            lambda: QualityGateConfig(
                retriever="keyword", metric="recall@1", max_value=float("inf")
            ),
            "finite",
        ),
        (lambda: ReportConfig(output_dir="reports"), "output_dir"),  # type: ignore[arg-type]
        (lambda: ReportConfig(formats="json"), "formats"),  # type: ignore[arg-type]
        (lambda: ReportConfig(formats=("json", "json")), "duplicates"),
    ],
)
def test_typed_config_model_validation_branches(
    factory: Callable[[], object], expected: str
) -> None:
    with pytest.raises(ConfigurationError, match=expected):
        factory()


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("experiment:\n  seed: 42", "experiment: []", "experiment"),
        ("seed: 42", "seed: true", "experiment.seed"),
        ("path: docs", "path: [docs]", "corpus.path"),
        ('include: ["**/*.txt"]', "include: '*.txt'", "corpus.include"),
        ("type: recursive_characters", "type: words", "chunker.type"),
        ("overlap: 4", "overlap: 32", "chunker.overlap"),
        ("format: native_jsonl", "format: csv", "dataset.format"),
        ("relevance_level: document", "relevance_level: other", "relevance_level"),
        ("top_k: [1, 3]", "top_k: one", "evaluation.top_k"),
        ("top_k: [1, 3]", "top_k: [1, true]", r"top_k\[1\]"),
        ("top_k: [1, 3]", "top_k: [1, 1]", "duplicates"),
        ("repetitions: 1", "repetitions: 2", "repetitions"),
        ("concurrency: 1", "concurrency: 0", "concurrency"),
        ("formats: [json, csv, html]", "formats: json", "report.formats"),
        ("formats: [json, csv, html]", "formats: [json, json]", "duplicates"),
        ("min_value: 0.5", "unknown_threshold: 0.5", "unknown_threshold"),
    ],
)
def test_yaml_loader_validation_branches(
    tmp_path: Path, old: str, new: str, expected: str
) -> None:
    path = _write_fixture(tmp_path)
    original = path.read_text(encoding="utf-8")
    assert old in original
    path.write_text(original.replace(old, new), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=expected):
        load_config(path)


def test_config_rejects_unreadable_and_non_mapping_inputs(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")

    path = tmp_path / "list.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="config"):
        load_config(path)


def test_retrieval_config_rejects_invalid_nested_contracts(tmp_path: Path) -> None:
    corpus = CorpusConfig(path=tmp_path / "docs")
    dataset = DatasetConfig(path=tmp_path / "eval.jsonl")
    evaluation = EvaluationConfig(top_k=(1,))
    keyword = KeywordRetrieverConfig(name="keyword")
    base: dict[str, object] = {
        "schema_version": 1,
        "corpus": corpus,
        "dataset": dataset,
        "retrievers": (keyword,),
        "evaluation": evaluation,
    }
    cases: list[tuple[dict[str, object], str]] = [
        ({"schema_version": True}, "schema_version"),
        ({"corpus": object()}, "corpus"),
        ({"dataset": object()}, "dataset"),
        ({"evaluation": object()}, "evaluation"),
        ({"experiment": object()}, "experiment"),
        ({"report": object()}, "report"),
        ({"retrievers": "keyword"}, "retrievers"),
        ({"retrievers": ()}, "must not be empty"),
        ({"retrievers": (object(),)}, "invalid config"),
        ({"retrievers": (keyword, keyword)}, "unique"),
        (
            {"retrievers": (KeywordRetrieverConfig(name="custom"),)},
            "names must equal",
        ),
        ({"quality_gates": "gate"}, "quality_gates"),
        ({"quality_gates": (object(),)}, "invalid config"),
        ({"source_dir": "relative"}, "source_dir"),
        (
            {
                "retrievers": (
                    keyword,
                    HybridRetrieverConfig(name="hybrid", sources=("hybrid", "keyword")),
                )
            },
            "itself",
        ),
        (
            {
                "retrievers": (
                    keyword,
                    HybridRetrieverConfig(name="hybrid", sources=("keyword", "dense")),
                )
            },
            "unknown source",
        ),
        (
            {
                "quality_gates": (
                    QualityGateConfig(
                        retriever="bm25", metric="recall@1", min_value=0.5
                    ),
                )
            },
            "unknown retriever",
        ),
    ]

    for overrides, expected in cases:
        values = {**base, **overrides}
        with pytest.raises(ConfigurationError, match=expected):
            RetrievalConfig(**values)  # type: ignore[arg-type]
