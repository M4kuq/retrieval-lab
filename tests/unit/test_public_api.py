from retrieval_lab import (
    BM25Retriever,
    DenseRetriever,
    Document,
    EmbeddingBackend,
    EmbeddingModelMetadata,
    EvaluationDataset,
    EvaluationRunner,
    ExperimentOutput,
    HybridRetriever,
    InitializedProject,
    OptionalDependencyError,
    RetrievalLabError,
    RetrievedQueryResult,
    ValidationResult,
    evaluate_results,
    initialize_project,
    load_documents,
    run_configured_experiment,
    validate_config_inputs,
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
    assert DenseRetriever.__module__ == "retrieval_lab.retrievers.dense"
    assert HybridRetriever.__module__ == "retrieval_lab.retrievers.hybrid"
    assert EmbeddingBackend.__module__ == "retrieval_lab.retrievers.dense"
    assert EmbeddingModelMetadata.__module__ == "retrieval_lab.retrievers.dense"
    assert OptionalDependencyError.__module__ == "retrieval_lab.exceptions"
    assert EvaluationDataset.__module__ == "retrieval_lab.datasets"
    assert RetrievedQueryResult.__module__ == "retrieval_lab.evaluation.precomputed"
    assert evaluate_results.__module__ == "retrieval_lab.evaluation.precomputed"
    assert load_documents.__module__ == "retrieval_lab.loaders"
    assert validate_dataset.__module__ == "retrieval_lab.datasets"


def test_application_services_are_available_from_the_package_root() -> None:
    assert InitializedProject.__module__ == "retrieval_lab.application"
    assert ValidationResult.__module__ == "retrieval_lab.application"
    assert ExperimentOutput.__module__ == "retrieval_lab.application"
    assert initialize_project.__module__ == "retrieval_lab.application"
    assert validate_config_inputs.__module__ == "retrieval_lab.application"
    assert run_configured_experiment.__module__ == "retrieval_lab.application"
