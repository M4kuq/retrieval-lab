from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from retrieval_lab import (
    BaseRetriever,
    Chunk,
    ConfigurationError,
    CorpusValidationError,
    DatasetValidationError,
    Document,
    EvaluationQuery,
    EvaluationRunner,
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

    assert first.to_json() == second.to_json()
