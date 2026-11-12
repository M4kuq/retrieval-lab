"""Resume an interrupted evaluation from a checkpoint.

A stand-in retriever crashes on the second query during the first pass,
so only q1 is recorded. Re-running with the same checkpoint executes
just the missing queries, and the collected rankings are then scored
with evaluate_results().
"""

import contextlib
import tempfile
from pathlib import Path

from retrieval_lab import (
    EvaluationCheckpoint,
    EvaluationDataset,
    EvaluationQuery,
    RetrievedQueryResult,
    evaluate_results,
    run_with_checkpoint,
)

RANKINGS = {
    "q1": ("d1", "d2", "d3"),
    "q2": ("d2", "d1", "d3"),
    "q3": ("d3", "d2", "d1"),
}

dataset = EvaluationDataset(
    queries=(
        EvaluationQuery("q1", "first query", frozenset({"d1"})),
        EvaluationQuery("q2", "second query", frozenset({"d2"})),
        EvaluationQuery("q3", "third query", frozenset({"d3"})),
    )
)

attempts: list[str] = []


def run_query(query: EvaluationQuery) -> RetrievedQueryResult:
    attempts.append(query.id)
    if query.id == "q2" and attempts.count("q2") == 1:
        raise RuntimeError("simulated crash")
    return RetrievedQueryResult(query.id, RANKINGS[query.id])


with tempfile.TemporaryDirectory() as directory:
    checkpoint = EvaluationCheckpoint(
        Path(directory) / "run.checkpoint.jsonl",
        fingerprint="checkpoint-example-v1",
    )
    with contextlib.suppress(RuntimeError):
        run_with_checkpoint(checkpoint, dataset.queries, run_query)
    assert checkpoint.completed_query_ids == frozenset({"q1"})

    rankings = run_with_checkpoint(checkpoint, dataset.queries, run_query)

assert attempts == ["q1", "q2", "q2", "q3"]
assert [ranking.query_id for ranking in rankings] == ["q1", "q2", "q3"]

result = evaluate_results(
    dataset=dataset, retrieved_results=rankings, top_k=(1, 3), name="checkpointed"
)
assert result.metrics["checkpointed"].recall_at(1) == 1.0
print("recall@1:", result.metrics["checkpointed"].recall_at(1))
