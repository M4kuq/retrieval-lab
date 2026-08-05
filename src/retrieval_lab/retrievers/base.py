"""Base contract for synchronous retrievers."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from retrieval_lab.models import Chunk, SearchResult


class BaseRetriever(ABC):
    """Contract implemented by synchronous, index-backed retrievers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable strategy name used in evaluation results."""

    @abstractmethod
    def index(self, chunks: Sequence[Chunk]) -> None:
        """Replace the retriever index with the supplied chunks."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return at most ``top_k`` results in deterministic best-first order."""
