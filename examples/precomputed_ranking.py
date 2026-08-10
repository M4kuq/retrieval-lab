"""Evaluate rankings produced by an existing system without retrieval calls."""

from retrieval_lab import EvaluationDataset, RetrievedQueryResult, evaluate_results

dataset = EvaluationDataset.from_jsonl("examples/japanese/qrels.jsonl")
rankings = (
    RetrievedQueryResult("q-retrieval", ("retrieval.md", "evaluation.md")),
    RetrievedQueryResult("q-cache", ("cache.md",)),
    RetrievedQueryResult("q-metrics", ("evaluation.md", "retrieval.md")),
    RetrievedQueryResult("q-offline", ("safety.md", "cache.md")),
)

result = evaluate_results(dataset=dataset, retrieved_results=rankings, top_k=(1, 3))

assert result.metrics["precomputed"].recall_at(3) == 1.0
