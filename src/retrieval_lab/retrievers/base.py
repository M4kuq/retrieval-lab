"""Base contract for synchronous retrievers."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from retrieval_lab.models import Chunk, JSONValue, SearchResult


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

    @property
    def settings(self) -> Mapping[str, JSONValue]:
        """Return deterministic settings recorded in an evaluation manifest.

        Subclasses may extend this mapping with their algorithm-specific options.
        The default keeps existing custom ``BaseRetriever`` implementations
        source-compatible while ensuring every built-in retriever has a stable
        manifest record.
        """

        return {"name": self.name, "type": self.name}
