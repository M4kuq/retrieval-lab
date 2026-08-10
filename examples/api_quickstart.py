"""Minimal offline Python API example using the Japanese sample corpus."""

from retrieval_lab import EvaluationDataset, EvaluationRunner, load_documents

documents = load_documents("examples/japanese/corpus")
dataset = EvaluationDataset.from_jsonl("examples/japanese/qrels.jsonl")

# quick_evaluate is convenient for the built-in strategies. It uses the query
# relevance IDs from the dataset, while from_dataset below preserves graded
# relevance when that distinction matters to the experiment.
result = EvaluationRunner.quick_evaluate(
    documents=documents,
    queries=dataset.queries,
    strategies=("keyword", "bm25"),
    top_k=(1, 3),
    seed=42,
)

assert set(result.metrics) == {"keyword", "bm25"}
assert result.metrics["bm25"].recall_at(1) >= 0.0

# The graded dataset follows the same runner path when grades are important.
graded_result = EvaluationRunner.from_dataset(
    documents=documents,
    dataset=dataset,
    strategies=("bm25",),
    top_k=(1, 3),
    seed=42,
).run()

assert graded_result.manifest["relevance_level"] == "document"
