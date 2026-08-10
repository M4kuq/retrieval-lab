"""Compare BM25, a local deterministic Dense backend, and Hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from retrieval_lab import (
    BM25Retriever,
    DenseRetriever,
    EmbeddingModelMetadata,
    EvaluationDataset,
    EvaluationRunner,
    HybridRetriever,
    load_documents,
)


class LocalEmbeddingBackend:
    """Tiny deterministic embedding backend; it never downloads a model."""

    metadata = EmbeddingModelMetadata(
        model_id="local/tutorial-keyword-v1",
        requested_revision="v1",
        resolved_revision="v1",
    )

    _terms = (
        "検索",
        "ランキング",
        "キャッシュ",
        "再現性",
        "recall",
        "mrr",
        "評価",
        "オフライン",
        "安全性",
    )

    def encode(
        self, texts: Sequence[str], *, batch_size: int
    ) -> Sequence[Sequence[float]]:
        del batch_size
        vectors: list[list[float]] = []
        for text in texts:
            normalized = text.casefold()
            vectors.append(
                [float(normalized.count(term.casefold())) for term in self._terms]
            )
        return vectors


documents = load_documents("examples/japanese/corpus")
dataset = EvaluationDataset.from_jsonl("examples/japanese/qrels.jsonl")
bm25 = BM25Retriever()
dense = DenseRetriever(backend=LocalEmbeddingBackend())
hybrid = HybridRetriever([bm25, dense], candidate_k=3)

result = EvaluationRunner.from_dataset(
    documents=documents,
    dataset=dataset,
    retrievers=(bm25, dense, hybrid),
    top_k=(1, 3),
    seed=42,
).run()

assert set(result.metrics) == {"bm25", "dense", "hybrid"}
assert result.manifest["retriever_settings"]["dense"]["model_id"].startswith("local/")
recall_at_3 = {
    name: result.metrics[name].recall_at(3) for name in ("bm25", "dense", "hybrid")
}
assert set(recall_at_3) == {"bm25", "dense", "hybrid"}
