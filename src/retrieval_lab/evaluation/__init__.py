"""Evaluation internals for Retrieval Lab."""

from retrieval_lab.evaluation.latency import (
    LatencyStats,
    nearest_rank_percentile,
)
from retrieval_lab.evaluation.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def __getattr__(name: str) -> object:
    """Load the precomputed entry point without creating a domain import cycle."""

    if name in {"RetrievedQueryResult", "evaluate_results"}:
        from retrieval_lab.evaluation.precomputed import (
            RetrievedQueryResult,
            evaluate_results,
        )

        return {
            "RetrievedQueryResult": RetrievedQueryResult,
            "evaluate_results": evaluate_results,
        }[name]
    raise AttributeError(name)


__all__ = [
    "LatencyStats",
    "RetrievedQueryResult",
    "average_precision_at_k",
    "evaluate_results",
    "hit_rate_at_k",
    "ndcg_at_k",
    "nearest_rank_percentile",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
