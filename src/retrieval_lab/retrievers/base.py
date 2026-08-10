"""Base contract for synchronous retrievers."""

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from retrieval_lab.models import Chunk, JSONValue, SearchResult


def _chunk_payload(chunk: Chunk) -> dict[str, object]:
    """Return the stable logical payload retained by built-in indexes."""

    return {
        "document_id": chunk.document_id,
        "end_offset": chunk.end_offset,
        "id": chunk.id,
        "metadata": dict(chunk.metadata),
        "start_offset": chunk.start_offset,
        "text": chunk.text,
    }


def _serialized_index_size_bytes(value: object) -> int:
    """Return the UTF-8 size of a deterministic logical index payload."""

    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


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
