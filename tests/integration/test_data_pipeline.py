"""End-to-end tests for file inputs and every public evaluation path."""

from pathlib import Path

import pytest

from retrieval_lab import (
    ConfigurationError,
    Document,
    EvaluationDataset,
    EvaluationQuery,
    EvaluationRunner,
    FixedSizeChunker,
    RetrievedQueryResult,
    evaluate_results,
    load_documents,
)


def test_jsonl_dataset_runs_keyword_bm25_and_precomputed_evaluation(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "security.md").write_text(
        "AWS Secrets Managerは機密情報を管理するサービスです。",
        encoding="utf-8",
    )
    (corpus / "storage.md").write_text(
        "Amazon S3はオブジェクトストレージです。",
        encoding="utf-8",
    )
    dataset_path = tmp_path / "evaluation.jsonl"
    dataset_path.write_text(
        '{"query_id":"q-1","query":"AWS 機密情報 管理",'
        '"relevant":[{"id":"security.md","relevance":3}]}\n',
        encoding="utf-8",
    )

    documents = load_documents(corpus)
    dataset = EvaluationDataset.from_jsonl(dataset_path)
    dataset.validate(documents=documents)

    retrieved = EvaluationRunner.from_dataset(
        documents=documents,
        dataset=dataset,
        strategies=["keyword", "bm25"],
        top_k=[1],
    ).run()
    precomputed = evaluate_results(
        dataset=dataset,
        retrieved_results=[RetrievedQueryResult("q-1", ["security.md", "storage.md"])],
        top_k=[1],
    )

    assert retrieved.metrics["keyword"].recall_at(1) == 1.0
    assert retrieved.metrics["bm25"].recall_at(1) == 1.0
    assert precomputed.metrics["precomputed"].recall_at(1) == 1.0
    assert retrieved.manifest["dataset_hash"] == precomputed.manifest["dataset_hash"]
    assert set(retrieved.metrics["bm25"].metrics_by_cutoff[1]) == {
        "ap",
        "hit_rate",
        "mrr",
        "ndcg",
        "precision",
        "recall",
    }


def test_runner_requires_exactly_one_query_source() -> None:
    document = Document("doc", "text")
    query = EvaluationQuery("q", "text", relevant_document_ids={"doc"})
    dataset = EvaluationDataset([query])

    with pytest.raises(ConfigurationError, match="exactly one"):
        EvaluationRunner(documents=[document])
    with pytest.raises(ConfigurationError, match="exactly one"):
        EvaluationRunner(
            documents=[document],
            queries=[query],
            dataset=dataset,
        )


def test_dataset_runner_evaluates_explicit_chunk_relevance() -> None:
    document = Document("doc", "alpha beta")
    chunker = FixedSizeChunker(size=5, overlap=0)
    gold_chunk = chunker.chunk([document])[0]
    dataset = EvaluationDataset(
        [
            EvaluationQuery(
                "q",
                "alpha",
                relevant_chunk_ids={gold_chunk.id},
            )
        ],
        relevance_level="chunk",
        relevance_grades_by_query={"q": {gold_chunk.id: 2}},
    )

    result = EvaluationRunner.from_dataset(
        documents=[document],
        dataset=dataset,
        strategies=["keyword"],
        top_k=[1],
        chunker=chunker,
    ).run()

    assert result.manifest["relevance_level"] == "chunk"
    assert result.query_results["keyword"][0].retrieved_ids == (gold_chunk.id,)
    assert result.metrics["keyword"].recall_at(1) == 1.0
