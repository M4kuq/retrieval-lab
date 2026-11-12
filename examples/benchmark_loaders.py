"""Load a BEIR-format benchmark, evaluate two strategies, and fuse them."""

from retrieval_lab import (
    EvaluationRunner,
    RetrievedQueryResult,
    evaluate_fused_results,
    load_beir_dataset,
)

benchmark = load_beir_dataset("examples/beir-mini")

# q3 has no positive judgments in the test split, so the loader reports it
# as excluded instead of dropping it silently.
assert benchmark.excluded_query_ids == ("q3",)
assert len(benchmark.documents) == 4
assert len(benchmark.dataset.queries) == 2
assert benchmark.dataset.relevance_grades_by_query["q1"] == {"d1": 2, "d3": 1}

result = EvaluationRunner.from_dataset(
    documents=benchmark.documents,
    dataset=benchmark.dataset,
    strategies=("keyword", "bm25"),
    top_k=(1, 3),
    seed=42,
).run()

assert set(result.metrics) == {"keyword", "bm25"}

# The rankings a run already produced can be fused after the fact; the
# fused run records its provenance and can be diffed against either source.
sources = {
    name: tuple(
        RetrievedQueryResult(evaluation.query_id, evaluation.retrieved_ids)
        for evaluation in result.query_results[name]
    )
    for name in ("keyword", "bm25")
}
fused_result = evaluate_fused_results(
    dataset=benchmark.dataset, results=sources, top_k=(1, 3)
)

assert fused_result.manifest["fusion"] == {
    "rrf_k": 60,
    "sources": ["bm25", "keyword"],
}
assert fused_result.metrics["hybrid"].recall_at(3) == 1.0

# For q1, BM25 alone ranks d3 above d2. Under RRF the two tie, and the
# identifier breaks the tie, so the fused order differs from BM25's own
# ranking and proves both sources fed the fusion.
fused_by_query = {
    evaluation.query_id: evaluation.retrieved_ids
    for evaluation in fused_result.query_results["hybrid"]
}
assert fused_by_query == {"q1": ("d1", "d2", "d3"), "q2": ("d2", "d1")}
