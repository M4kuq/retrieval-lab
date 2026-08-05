"""Public exception hierarchy for Retrieval Lab."""


class RetrievalLabError(Exception):
    """Base class for errors raised by Retrieval Lab public APIs."""


class ConfigurationError(RetrievalLabError):
    """Raised when an evaluation configuration is invalid."""


class DatasetValidationError(RetrievalLabError):
    """Raised when an evaluation dataset violates its public contract."""


class CorpusValidationError(RetrievalLabError):
    """Raised when a document or chunk violates the corpus contract."""


class RetrieverContractError(RetrievalLabError):
    """Raised when a retriever result violates the ranking contract."""


class EvaluationError(RetrievalLabError):
    """Raised when an evaluation cannot produce a valid result."""


class IncomparableRunError(RetrievalLabError):
    """Raised when two evaluation runs cannot be compared safely."""


__all__ = [
    "ConfigurationError",
    "CorpusValidationError",
    "DatasetValidationError",
    "EvaluationError",
    "IncomparableRunError",
    "RetrievalLabError",
    "RetrieverContractError",
]
