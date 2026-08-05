"""Evaluation internals for Retrieval Lab."""

from retrieval_lab.evaluation.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from retrieval_lab.evaluation.precomputed import RetrievedQueryResult, evaluate_results

__all__ = [
    "RetrievedQueryResult",
    "average_precision_at_k",
    "evaluate_results",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
