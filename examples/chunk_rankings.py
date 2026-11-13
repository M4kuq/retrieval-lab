"""Evaluate and fuse chunk rankings produced by existing search systems.

The fixture ``examples/chunked-qrels.jsonl`` is a native evaluation
dataset whose relevance is graded at chunk level, so it is loaded with
``relevance_level="chunk"``. The two systems below stand in for external
chunk search services: their rankings are scored as-is with
``evaluate_results()`` and fused with ``evaluate_fused_results()``,
without the built-in retrievers, a chunker, or a corpus.
"""

from retrieval_lab import (
    EvaluationDataset,
    RetrievedQueryResult,
    evaluate_fused_results,
    evaluate_results,
)

dataset = EvaluationDataset.from_jsonl(
    "examples/chunked-qrels.jsonl", relevance_level="chunk"
)

assert dataset.relevance_level == "chunk"
assert dataset.relevance_grades_by_query["q1"] == {
    "doc-refund#0": 2,
    "doc-refund#1": 1,
}
assert dataset.relevance_grades_by_query["q2"] == {"doc-cache#2": 1}

vector_rankings = (
    RetrievedQueryResult("q1", ("doc-refund#1", "doc-shipping#0", "doc-refund#0")),
    RetrievedQueryResult("q2", ("doc-cache#0", "doc-cache#2")),
)
keyword_rankings = (
    RetrievedQueryResult("q1", ("doc-refund#0",)),
    RetrievedQueryResult("q2", ("doc-cache#2", "doc-cache#1")),
)

vector = evaluate_results(
    dataset=dataset, retrieved_results=vector_rankings, top_k=(1, 3), name="vector"
)
keyword = evaluate_results(
    dataset=dataset, retrieved_results=keyword_rankings, top_k=(1, 3), name="keyword"
)

# Vector ranks a relevant chunk first for q1 only; keyword for both.
assert vector.metrics["vector"].recall_at(1) == 0.25
assert vector.metrics["vector"].recall_at(3) == 1.0
assert keyword.metrics["keyword"].recall_at(1) == 0.75
assert keyword.metrics["keyword"].recall_at(3) == 0.75

fused = evaluate_fused_results(
    dataset=dataset,
    results={"keyword": keyword_rankings, "vector": vector_rankings},
    top_k=(1, 3),
)

assert fused.manifest["relevance_level"] == "chunk"
assert fused.manifest["fusion"] == {"rrf_k": 60, "sources": ["keyword", "vector"]}
assert fused.metrics["hybrid"].recall_at(1) == 0.75
assert fused.metrics["hybrid"].recall_at(3) == 1.0
