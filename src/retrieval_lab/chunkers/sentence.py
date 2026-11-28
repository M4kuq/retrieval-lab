"""Sentence-boundary chunking."""

from collections.abc import Sequence

from retrieval_lab.chunkers.ids import stable_chunk_id
from retrieval_lab.exceptions import CorpusValidationError
from retrieval_lab.models import Chunk, Document

_TERMINATORS = frozenset("\u3002\uff0e.!?\uff01\uff1f")
_CLOSERS = frozenset("\u300d\u300f\u3011\u3009\u300b\uff09)\"'\u201d\u2019\u00bb")


class SentenceBoundaryChunker:
    """Split documents into chunks that end on sentence boundaries.

    Sentences are packed in order until adding another would exceed
    ``size`` characters. ``overlap`` is a character budget: trailing
    sentences totalling at most ``overlap`` characters are repeated at
    the start of the next chunk. A single sentence longer than ``size``
    is split at ``size`` characters, and trailing whitespace is folded
    into the final chunk instead of forming a chunk of its own, so a
    chunk can exceed ``size`` only by an adjacent whitespace run.
    """

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
        """Split one document at sentence boundaries within ``size``."""
        if not isinstance(document, Document):
            raise CorpusValidationError("document must be a Document record")
        text = document.text
        metadata = dict(document.metadata)
        if document.source is not None:
            metadata.setdefault("source", document.source)

        units, trailing_start = _units(text, self.size)
        spans: list[tuple[int, int]] = []
        index = 0
        while index < len(units):
            if units[index][1] - units[index][0] > self.size:
                spans.append(units[index])
                index += 1
                continue
            end_index = index
            total = 0
            while (
                end_index < len(units)
                and total + (units[end_index][1] - units[end_index][0]) <= self.size
            ):
                total += units[end_index][1] - units[end_index][0]
                end_index += 1
            spans.append((units[index][0], units[end_index - 1][1]))
            if end_index == len(units):
                break
            index = _next_index(units, index, end_index, self.overlap, self.size)
        if trailing_start is not None and spans:
            spans[-1] = (spans[-1][0], len(text))

        return [
            Chunk(
                id=stable_chunk_id(document.id, start, end, text[start:end]),
                document_id=document.id,
                text=text[start:end],
                start_offset=start,
                end_offset=end,
                metadata=metadata.copy(),
            )
            for start, end in spans
        ]


def _next_index(
    units: list[tuple[int, int]],
    start_index: int,
    end_index: int,
    overlap: int,
    size: int,
) -> int:
    carried = 0
    candidate = end_index
    for index in range(end_index - 1, start_index, -1):
        length = units[index][1] - units[index][0]
        if carried + length > overlap:
            break
        carried += length
        candidate = index
    next_length = units[end_index][1] - units[end_index][0]
    while candidate < end_index and carried + next_length > size:
        carried -= units[candidate][1] - units[candidate][0]
        candidate += 1
    return candidate


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char not in _TERMINATORS or _is_decimal_point(text, index):
            index += 1
            continue
        end = index + 1
        while end < len(text) and (
            text[end] in _TERMINATORS and not _is_decimal_point(text, end)
        ):
            end += 1
        while end < len(text) and text[end] in _CLOSERS:
            end += 1
        if (
            char in ".\uff0e"
            and end < len(text)
            and text[end].isascii()
            and text[end].isalnum()
        ):
            index = end
            continue
        spans.append((start, end))
        start = end
        index = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _is_decimal_point(text: str, index: int) -> bool:
    return (
        text[index] in ".\uff0e"
        and 0 < index < len(text) - 1
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def _units(text: str, size: int) -> tuple[list[tuple[int, int]], int | None]:
    spans = _sentence_spans(text)
    trailing_start: int | None = None
    if len(spans) > 1 and not text[spans[-1][0] : spans[-1][1]].strip():
        trailing_start = spans[-1][0]
        spans = spans[:-1]
    units: list[tuple[int, int]] = []
    for start, end in spans:
        piece = start
        while end - piece > size:
            units.append((piece, piece + size))
            piece += size
        units.append((piece, end))
    merged: list[tuple[int, int]] = []
    pending: tuple[int, int] | None = None
    for unit in units:
        if not text[unit[0] : unit[1]].strip():
            if merged:
                merged[-1] = (merged[-1][0], unit[1])
            elif pending is None:
                pending = unit
            else:
                pending = (pending[0], unit[1])
            continue
        if pending is not None:
            unit = (pending[0], unit[1])
            pending = None
        merged.append(unit)
    return merged, trailing_start
