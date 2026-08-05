"""Compatibility alias for the public Retrieval Lab exception hierarchy."""

from retrieval_lab.exceptions import (
    ConfigurationError,
    CorpusValidationError,
    DatasetValidationError,
    EvaluationError,
    IncomparableRunError,
    RetrievalLabError,
    RetrieverContractError,
)

__all__ = [
    "ConfigurationError",
    "CorpusValidationError",
    "DatasetValidationError",
    "EvaluationError",
    "IncomparableRunError",
    "RetrievalLabError",
    "RetrieverContractError",
]
