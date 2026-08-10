"""Evaluate a provider-independent custom Callable Retriever."""

from retrieval_lab import (
    CallableRetriever,
    EvaluationDataset,
    RetrievedItem,
    evaluate_retrievers,
)

dataset = EvaluationDataset.from_jsonl("examples/japanese/qrels.jsonl")

rankings = {
    "q-retrieval": ("retrieval.md", "evaluation.md"),
    "q-cache": ("cache.md",),
    "q-metrics": ("evaluation.md", "retrieval.md"),
    "q-offline": ("safety.md", "cache.md"),
}


def search(query: str, *, top_k: int) -> list[RetrievedItem]:
    """Return a fixed local ranking as a stand-in for an existing service."""

    query_id = next(
        query_record.id
        for query_record in dataset.queries
        if query_record.query == query
    )
    return [
        RetrievedItem(
            identifier,
            parent_document_id=identifier,
            score=float(top_k - position),
        )
        for position, identifier in enumerate(rankings[query_id][:top_k])
    ]


result = evaluate_retrievers(
    dataset=dataset,
    retrievers={"local-service": CallableRetriever("local-service", search)},
    top_k=(1, 3),
)

assert result.metrics["local-service"].recall_at(3) == 1.0
