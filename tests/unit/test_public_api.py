from retrieval_lab import (
    BM25Retriever,
    Document,
    EvaluationDataset,
    EvaluationRunner,
    RetrievalLabError,
    RetrievedQueryResult,
    evaluate_results,
    load_documents,
    validate_dataset,
)
from retrieval_lab.errors import RetrievalLabError as ErrorAlias
from retrieval_lab.models import Document as DocumentAlias


def test_package_root_and_compatibility_modules_share_public_types() -> None:
    assert DocumentAlias is Document
    assert ErrorAlias is RetrievalLabError
    assert EvaluationRunner.__module__ == "retrieval_lab.runner"


def test_new_milestone_apis_are_available_from_the_package_root() -> None:
    assert BM25Retriever.__module__ == "retrieval_lab.retrievers.bm25"
    assert EvaluationDataset.__module__ == "retrieval_lab.datasets"
    assert RetrievedQueryResult.__module__ == "retrieval_lab.evaluation.precomputed"
    assert evaluate_results.__module__ == "retrieval_lab.evaluation.precomputed"
    assert load_documents.__module__ == "retrieval_lab.loaders"
    assert validate_dataset.__module__ == "retrieval_lab.datasets"
