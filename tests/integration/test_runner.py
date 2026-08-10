from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

import retrieval_lab.runner as runner_module
from retrieval_lab import (
    BaseRetriever,
    BM25Retriever,
    Chunk,
    ConfigurationError,
    CorpusValidationError,
    DatasetValidationError,
    DenseRetriever,
    Document,
    EmbeddingModelMetadata,
    EvaluationError,
    EvaluationQuery,
    EvaluationRunner,
    HybridRetriever,
    KeywordRetriever,
    SearchResult,
)


def test_quick_evaluate_runs_readme_vertical_slice(tmp_path: Path) -> None:
    result = EvaluationRunner.quick_evaluate(
        documents=[
            Document(id="doc-1", text="RAGでは検索品質の評価が重要です。"),
            Document(id="doc-2", text="Pythonではpytestを利用できます。"),
        ],
        queries=[
            EvaluationQuery(
                id="q-1",
                query="RAG 検索品質",
                relevant_document_ids={"doc-1"},
            )
        ],
        strategies=["keyword"],
        top_k=[1],
    )

    assert result.metrics["keyword"].recall_at(1) == 1.0
    assert result.query_results["keyword"][0].retrieved_ids == ("doc-1",)
    assert result.query_results["keyword"][0].warnings
    assert len(result.run_id) == 64

    output = tmp_path / "nested" / "result.json"
    result.save_json(output)
    assert output.read_text(encoding="utf-8") == result.to_json()
    assert json.loads(result.to_json())["run"]["id"] == result.run_id


def test_runner_macro_averages_queries_and_normalizes_cutoffs() -> None:
    result = EvaluationRunner.quick_evaluate(
        documents=[
            Document(id="hit", text="shared relevant"),
            Document(id="miss", text="unrelated"),
        ],
        queries=[
            EvaluationQuery(
                id="hit-query",
                query="shared",
                relevant_document_ids={"hit"},
            ),
            EvaluationQuery(
                id="miss-query",
                query="absent",
                relevant_document_ids={"miss"},
            ),
        ],
        top_k=[3, 1],
    )

    assert tuple(result.metrics["keyword"].metrics_by_cutoff) == (1, 3)
    assert result.metrics["keyword"].recall_at(1) == 0.5
    assert result.metrics["keyword"].recall_at(3) == 0.5


class _CountingRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.search_cutoffs: list[int] = []
        self._chunks: tuple[Chunk, ...] = ()

    @property
    def name(self) -> str:
        return "counting"

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = tuple(chunks)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.search_cutoffs.append(top_k)
        return [
            SearchResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=float(len(self._chunks) - index),
                rank=index + 1,
                metadata=chunk.metadata,
            )
            for index, chunk in enumerate(self._chunks[:top_k])
        ]


class _FailingRetriever(_CountingRetriever):
    @property
    def name(self) -> str:
        return "failing"

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        raise RuntimeError("provider secret must remain chained, not serialized")


def test_runner_searches_once_at_max_k_and_accepts_custom_retriever() -> None:
    retriever = _CountingRetriever()
    runner = EvaluationRunner(
        documents=[Document(id="doc", text="relevant")],
        queries=[
            EvaluationQuery(
                id="query",
                query="anything",
                relevant_document_ids={"doc"},
            )
        ],
        retrievers=[retriever],
        top_k=[1, 3],
    )

    result = runner.run()

    assert retriever.search_cutoffs == [3]
    assert result.metrics["counting"].recall_at(1) == 1.0


def test_runner_uses_injected_monotonic_clock_for_build_and_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([0, 2_000_000, 10_000_000, 13_000_000])
    monkeypatch.setattr(runner_module, "perf_counter_ns", lambda: next(clock))

    result = EvaluationRunner(
        documents=[Document(id="doc", text="relevant")],
        queries=[
            EvaluationQuery(
                id="query",
                query="relevant",
                relevant_document_ids={"doc"},
            )
        ],
        retrievers=[_CountingRetriever()],
        top_k=[1],
    ).run()

    runtime = result.manifest["runtime"]
    assert runtime["build_ms"] == {"counting": 2.0}  # type: ignore[index]
    assert result.latency["counting"].mean_ms == 3.0
    assert result.latency["counting"].p50_ms == 3.0
    assert result.query_results["counting"][0].search_latency_ms == 3.0


def test_timing_changes_do_not_change_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run_with_clock(values: list[int]) -> object:
        clock = iter(values)
        monkeypatch.setattr(runner_module, "perf_counter_ns", lambda: next(clock))
        return EvaluationRunner(
            documents=[Document(id="doc", text="relevant")],
            queries=[
                EvaluationQuery(
                    id="query",
                    query="relevant",
                    relevant_document_ids={"doc"},
                )
            ],
            retrievers=[_CountingRetriever()],
            top_k=[1],
        ).run()

    fast = run_with_clock([0, 1_000_000, 2_000_000, 3_000_000])
    slow = run_with_clock([0, 8_000_000, 9_000_000, 25_000_000])

    assert fast.run_id == slow.run_id  # type: ignore[union-attr]
    assert fast.to_json() != slow.to_json()  # type: ignore[union-attr]


def test_query_failure_fails_run_and_preserves_exception_chain() -> None:
    runner = EvaluationRunner(
        documents=[Document(id="doc", text="relevant")],
        queries=[
            EvaluationQuery(
                id="query",
                query="relevant",
                relevant_document_ids={"doc"},
            )
        ],
        retrievers=[_FailingRetriever()],
        top_k=[1],
    )

    with pytest.raises(EvaluationError, match="failed during evaluation") as error:
        runner.run()

    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.mark.parametrize("top_k", [[], [0], [True], [1, 1]])
def test_runner_rejects_invalid_cutoffs(top_k: list[int]) -> None:
    with pytest.raises(ConfigurationError):
        EvaluationRunner(
            documents=[Document(id="doc", text="text")],
            queries=[
                EvaluationQuery(
                    id="query",
                    query="text",
                    relevant_document_ids={"doc"},
                )
            ],
            top_k=top_k,
        )


def test_runner_rejects_duplicate_documents_and_queries() -> None:
    document = Document(id="doc", text="text")
    query = EvaluationQuery(
        id="query",
        query="text",
        relevant_document_ids={"doc"},
    )
    with pytest.raises(CorpusValidationError, match="unique"):
        EvaluationRunner(documents=[document, document], queries=[query])
    with pytest.raises(DatasetValidationError, match="unique"):
        EvaluationRunner(documents=[document], queries=[query, query])


def test_runner_rejects_unsupported_strategy_and_chunk_only_relevance() -> None:
    document = Document(id="doc", text="text")
    with pytest.raises(ConfigurationError, match="unsupported strategy"):
        EvaluationRunner(
            documents=[document],
            queries=[
                EvaluationQuery(
                    id="query",
                    query="text",
                    relevant_document_ids={"doc"},
                )
            ],
            strategies=["dense"],
        )

    runner = EvaluationRunner(
        documents=[document],
        queries=[
            EvaluationQuery(
                id="query",
                query="text",
                relevant_chunk_ids={"chunk"},
            )
        ],
    )
    with pytest.raises(DatasetValidationError, match="no document relevance"):
        runner.run()


def test_identical_inputs_produce_identical_result_json() -> None:
    documents = [Document(id="doc", text="stable text")]
    queries = [
        EvaluationQuery(
            id="query",
            query="stable",
            relevant_document_ids={"doc"},
        )
    ]

    first = EvaluationRunner.quick_evaluate(documents=documents, queries=queries)
    second = EvaluationRunner.quick_evaluate(documents=documents, queries=queries)

    assert first.run_id == second.run_id
    assert first.manifest["seed"] == second.manifest["seed"] == 42
    assert first.manifest["runtime"] != {}
    assert first.query_results["keyword"][0].search_latency_ms is not None


class _RunnerEmbeddingBackend:
    metadata = EmbeddingModelMetadata(
        model_id="runner-fake",
        requested_revision="requested",
        resolved_revision="resolved",
    )

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> Sequence[Sequence[float]]:
        return [
            [1.0, 0.0]
            if "shared relevant" in text or text in {"query: relevant", "query: shared"}
            else [0.0, 1.0]
            for text in texts
        ]


def test_runner_evaluates_keyword_bm25_and_dense_on_shared_chunks() -> None:
    result = EvaluationRunner(
        documents=[
            Document(id="a", text="shared relevant"),
            Document(id="b", text="shared unrelated"),
        ],
        queries=[
            EvaluationQuery(
                id="query",
                query="shared",
                relevant_document_ids={"a"},
            )
        ],
        retrievers=[
            KeywordRetriever(),
            BM25Retriever(),
            DenseRetriever(backend=_RunnerEmbeddingBackend()),
        ],
        top_k=[1],
    ).run()

    assert set(result.metrics) == {"keyword", "bm25", "dense"}
    assert result.metrics["dense"].recall_at(1) == 1.0
    assert result.manifest["retrievers"] == ["keyword", "bm25", "dense"]
    dense_settings = result.manifest["retriever_settings"]["dense"]
    assert dense_settings["model_id"] == "runner-fake"
    assert dense_settings["resolved_revision"] == "resolved"


def test_runner_evaluates_keyword_bm25_dense_and_hybrid_on_shared_chunks() -> None:
    bm25 = BM25Retriever()
    dense = DenseRetriever(backend=_RunnerEmbeddingBackend())
    hybrid = HybridRetriever([bm25, dense], candidate_k=3)
    result = EvaluationRunner(
        documents=[
            Document(id="a", text="shared relevant"),
            Document(id="b", text="shared unrelated"),
        ],
        queries=[
            EvaluationQuery(
                id="query",
                query="shared",
                relevant_document_ids={"a"},
            )
        ],
        retrievers=[KeywordRetriever(), bm25, dense, hybrid],
        top_k=[1, 2],
    ).run()

    assert set(result.metrics) == {"keyword", "bm25", "dense", "hybrid"}
    assert result.metrics["hybrid"].recall_at(2) == 1.0
    assert result.manifest["retrievers"] == ["keyword", "bm25", "dense", "hybrid"]
    assert result.manifest["chunk_hash"]
    assert result.manifest["retriever_settings"]["hybrid"] == hybrid.settings


def test_dense_settings_change_deterministic_run_id() -> None:
    documents = [Document(id="document", text="relevant")]
    queries = [
        EvaluationQuery(
            id="query",
            query="relevant",
            relevant_document_ids={"document"},
        )
    ]
    first = EvaluationRunner(
        documents=documents,
        queries=queries,
        retrievers=[DenseRetriever(backend=_RunnerEmbeddingBackend())],
        top_k=[1],
    ).run()
    second = EvaluationRunner(
        documents=documents,
        queries=queries,
        retrievers=[
            DenseRetriever(
                backend=_RunnerEmbeddingBackend(),
                normalize_embeddings=False,
            )
        ],
        top_k=[1],
    ).run()

    assert first.run_id != second.run_id
    assert first.manifest["retrievers"] == second.manifest["retrievers"] == ["dense"]
