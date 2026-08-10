"""Public exception hierarchy for Retrieval Lab."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval_lab.comparison import ComparabilityIssue


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


class OptionalDependencyError(RetrievalLabError):
    """Raised when an optional Retrieval Lab feature is not installed."""


class EvaluationError(RetrievalLabError):
    """Raised when an evaluation cannot produce a valid result."""


class IncomparableRunError(RetrievalLabError):
    """Raised when two evaluation runs cannot be compared safely."""

    def __init__(
        self,
        message: str,
        *,
        issues: tuple[ComparabilityIssue, ...] = (),
    ) -> None:
        """Create an error retaining all structured comparability issues.

        The type is available only to static type checkers to avoid importing
        comparison models at runtime and creating a module cycle.
        """
        super().__init__(message)
        self.issues: tuple[ComparabilityIssue, ...] = issues


__all__ = [
    "ConfigurationError",
    "CorpusValidationError",
    "DatasetValidationError",
    "EvaluationError",
    "IncomparableRunError",
    "OptionalDependencyError",
    "RetrievalLabError",
    "RetrieverContractError",
]
