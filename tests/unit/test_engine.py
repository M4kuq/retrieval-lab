from __future__ import annotations

import pytest

from retrieval_lab.evaluation.engine import (
    aggregate_metrics,
    content_hash,
    evaluate_cutoff_rankings,
)
from retrieval_lab.exceptions import EvaluationError


def test_evaluate_cutoff_rankings_rejects_missing_cutoff_ranking() -> None:
    with pytest.raises(EvaluationError, match="cutoff ranking is missing"):
        evaluate_cutoff_rankings(
            query_id="q1",
            retrieved_ids=["a"],
            retrieved_ids_by_cutoff={1: ["a"]},
            relevance_grades={"a": 1},
            top_k=(1, 3),
        )


def test_aggregate_metrics_rejects_empty_evaluations() -> None:
    with pytest.raises(EvaluationError, match="at least one query evaluation"):
        aggregate_metrics([], (1,))


@pytest.mark.parametrize("value", [float("nan"), object()])
def test_content_hash_rejects_unserializable_values(value: object) -> None:
    with pytest.raises(EvaluationError, match="could not be hashed"):
        content_hash(value)  # type: ignore[arg-type]
