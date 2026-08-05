"""Fixed-size character chunking."""

from collections.abc import Sequence

from retrieval_lab.chunkers.ids import stable_chunk_id
from retrieval_lab.exceptions import CorpusValidationError
from retrieval_lab.models import Chunk, Document


class FixedSizeChunker:
    """Split documents into deterministic, overlapping character chunks."""

    def __init__(self, size: int = 512, overlap: int = 64) -> None:
        """Configure the character count and overlap for each chunk."""
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise CorpusValidationError(
                "chunk size must be a positive integer; for example, size=512"
            )
        if (
            isinstance(overlap, bool)
            or not isinstance(overlap, int)
            or overlap < 0
            or overlap >= size
        ):
            raise CorpusValidationError(
                "chunk overlap must be an integer satisfying "
                "0 <= overlap < size; for example, overlap=64 with size=512"
            )
        self.size = size
        self.overlap = overlap

    def chunk(self, documents: Sequence[Document]) -> list[Chunk]:
        """Chunk documents while preserving document and offset order."""
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise CorpusValidationError(
                "documents must be a sequence of Document records"
            )
        chunks: list[Chunk] = []
        for position, document in enumerate(documents):
            if not isinstance(document, Document):
                raise CorpusValidationError(
                    f"documents[{position}] must be a Document record"
                )
            chunks.extend(self.chunk_document(document))
        return chunks

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split one document without dropping any characters from its text."""
        if not isinstance(document, Document):
            raise CorpusValidationError("document must be a Document record")
        chunks: list[Chunk] = []
        start = 0
        text_length = len(document.text)

        metadata = dict(document.metadata)
        if document.source is not None:
            metadata.setdefault("source", document.source)

        while start < text_length:
            end = min(start + self.size, text_length)
            chunk_text = document.text[start:end]
            chunks.append(
                Chunk(
                    id=stable_chunk_id(document.id, start, end, chunk_text),
                    document_id=document.id,
                    text=chunk_text,
                    start_offset=start,
                    end_offset=end,
                    metadata=metadata.copy(),
                )
            )
            if end == text_length:
                break
            start = end - self.overlap

        return chunks
