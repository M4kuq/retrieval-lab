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
    if name == "evaluate_retrievers":
        from retrieval_lab.retrievers.callable import evaluate_retrievers

        return evaluate_retrievers
    if name in {
        "AsyncRetriever",
        "AsyncCallableRetriever",
        "evaluate_async_retrievers",
    }:
        from retrieval_lab.retrievers.async_callable import (
            AsyncCallableRetriever,
            AsyncRetriever,
            evaluate_async_retrievers,
        )

        return {
            "AsyncCallableRetriever": AsyncCallableRetriever,
            "AsyncRetriever": AsyncRetriever,
            "evaluate_async_retrievers": evaluate_async_retrievers,
        }[name]
    raise AttributeError(name)


__all__ = [
    "AsyncCallableRetriever",
    "AsyncRetriever",
    "LatencyStats",
    "RetrievedQueryResult",
    "average_precision_at_k",
    "evaluate_async_retrievers",
    "evaluate_results",
    "evaluate_retrievers",
    "hit_rate_at_k",
    "ndcg_at_k",
    "nearest_rank_percentile",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
