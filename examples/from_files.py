"""Run the local file-input example from the repository root."""

from retrieval_lab import EvaluationDataset, EvaluationRunner, load_documents

documents = load_documents("examples/corpus")
dataset = EvaluationDataset.from_jsonl("examples/evaluation.jsonl")
result = EvaluationRunner.from_dataset(
    documents=documents,
    dataset=dataset,
    strategies=["keyword", "bm25"],
    top_k=[1, 3],
).run()

assert result.metrics["keyword"].recall_at(1) == 1.0
assert result.metrics["bm25"].recall_at(1) == 1.0
