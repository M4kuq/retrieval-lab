"""Public Python API for Retrieval Lab."""

from retrieval_lab.chunkers import FixedSizeChunker
from retrieval_lab.datasets import EvaluationDataset, RelevanceLevel, validate_dataset
from retrieval_lab.domain import (
    Chunk,
    Document,
    EvaluationQuery,
    EvaluationResult,
    QueryEvaluation,
    RetrieverMetrics,
    SearchResult,
)
from retrieval_lab.evaluation.precomputed import RetrievedQueryResult, evaluate_results
from retrieval_lab.exceptions import (
    ConfigurationError,
    CorpusValidationError,
    DatasetValidationError,
    EvaluationError,
    IncomparableRunError,
    OptionalDependencyError,
    RetrievalLabError,
    RetrieverContractError,
)
from retrieval_lab.loaders import load_documents
from retrieval_lab.retrievers import (
    BaseRetriever,
    BM25Retriever,
    DenseRetriever,
    EmbeddingBackend,
    EmbeddingModelMetadata,
    HybridRetriever,
    KeywordRetriever,
)
from retrieval_lab.runner import EvaluationRunner

__all__ = [
    "BM25Retriever",
    "BaseRetriever",
    "Chunk",
    "ConfigurationError",
    "CorpusValidationError",
    "DatasetValidationError",
    "DenseRetriever",
    "Document",
    "EmbeddingBackend",
    "EmbeddingModelMetadata",
    "EvaluationDataset",
    "EvaluationError",
    "EvaluationQuery",
    "EvaluationResult",
    "EvaluationRunner",
    "FixedSizeChunker",
    "HybridRetriever",
    "IncomparableRunError",
    "KeywordRetriever",
    "OptionalDependencyError",
    "QueryEvaluation",
    "RelevanceLevel",
    "RetrievalLabError",
    "RetrievedQueryResult",
    "RetrieverContractError",
    "RetrieverMetrics",
    "SearchResult",
    "evaluate_results",
    "load_documents",
    "validate_dataset",
]
