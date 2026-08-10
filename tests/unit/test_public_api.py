from retrieval_lab import (
    BM25Retriever,
    ComparisonOutput,
    ComparisonRow,
    DenseRetriever,
    Document,
    EmbeddingBackend,
    EmbeddingModelMetadata,
    EvaluationDataset,
    EvaluationRunner,
    ExperimentOutput,
    GateOutput,
    HybridRetriever,
    InitializedProject,
    InspectionOutput,
    OptionalDependencyError,
    QueryEvidence,
    RetrievalLabError,
    RetrievedQueryResult,
    ValidationResult,
    compare_result_files,
    evaluate_configured_quality_gates,
    evaluate_results,
    initialize_project,
    inspect_result,
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
    assert QueryEvidence.__module__ == "retrieval_lab.application"
    assert InspectionOutput.__module__ == "retrieval_lab.application"
    assert ComparisonRow.__module__ == "retrieval_lab.application"
    assert ComparisonOutput.__module__ == "retrieval_lab.application"
    assert GateOutput.__module__ == "retrieval_lab.application"
    assert inspect_result.__module__ == "retrieval_lab.application"
    assert compare_result_files.__module__ == "retrieval_lab.application"
    assert evaluate_configured_quality_gates.__module__ == "retrieval_lab.application"
