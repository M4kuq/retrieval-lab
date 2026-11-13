"""Load MS MARCO and Natural Questions benchmarks and evaluate both.

The MS MARCO fixture uses the published passage layout
(``collection.tsv``, ``queries.<split>.tsv``, TREC ``qrels.<split>.tsv``);
the Natural Questions fixture is a single simplified-format JSONL file
whose pages become documents keyed by ``example_id``. Both loaders
report queries they cannot evaluate in ``excluded_query_ids`` instead
of dropping them silently.
"""

from retrieval_lab import (
    EvaluationRunner,
    load_msmarco_dataset,
    load_natural_questions_dataset,
)

msmarco = load_msmarco_dataset("examples/msmarco-mini")

# q3 has no qrels rows at all, so the loader reports it as excluded.
assert msmarco.excluded_query_ids == ("q3",)
assert len(msmarco.documents) == 4
assert len(msmarco.dataset.queries) == 2
assert msmarco.dataset.relevance_grades_by_query["q1"] == {"p1": 2, "p2": 1}
assert msmarco.dataset.relevance_grades_by_query["q2"] == {"p4": 1}

result = EvaluationRunner.from_dataset(
    documents=msmarco.documents,
    dataset=msmarco.dataset,
    strategies=("keyword", "bm25"),
    top_k=(1, 3),
    seed=42,
).run()

assert set(result.metrics) == {"keyword", "bm25"}
assert result.metrics["bm25"].recall_at(3) == 1.0

natural_questions = load_natural_questions_dataset("examples/nq-mini.jsonl")

# Example 103 has no long answer (all three fields are -1), so its page
# stays in the corpus but its question is reported as excluded.
assert natural_questions.excluded_query_ids == ("103",)
assert [document.id for document in natural_questions.documents] == [
    "101",
    "102",
    "103",
]
assert len(natural_questions.dataset.queries) == 2
assert natural_questions.dataset.relevance_grades_by_query["101"] == {"101": 1}

nq_result = EvaluationRunner.from_dataset(
    documents=natural_questions.documents,
    dataset=natural_questions.dataset,
    strategies=("keyword", "bm25"),
    top_k=(1, 3),
    seed=42,
).run()

assert set(nq_result.metrics) == {"keyword", "bm25"}
assert nq_result.metrics["bm25"].recall_at(1) == 1.0
