"""Tests for deterministic fixed-size chunking."""

from collections.abc import Sequence
from typing import cast

import pytest

from retrieval_lab.chunkers import FixedSizeChunker
from retrieval_lab.exceptions import CorpusValidationError
from retrieval_lab.models import Document


def test_short_document_produces_one_chunk() -> None:
    document = Document(id="doc-1", text="short", source="manual.md")

    chunks = FixedSizeChunker(size=10, overlap=2).chunk([document])

    assert [(chunk.text, chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        ("short", 0, 5)
    ]
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].metadata == {"source": "manual.md"}


def test_exact_size_does_not_produce_overlap_only_tail() -> None:
    chunks = FixedSizeChunker(size=5, overlap=2).chunk(
        [Document(id="doc-1", text="abcde")]
    )

    assert [(chunk.text, chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        ("abcde", 0, 5)
    ]


def test_overlap_covers_all_text_with_character_offsets() -> None:
    document = Document(id="doc-1", text="abcdefghij")

    chunks = FixedSizeChunker(size=5, overlap=2).chunk_document(document)

    assert [(chunk.text, chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        ("abcde", 0, 5),
        ("defgh", 3, 8),
        ("ghij", 6, 10),
    ]
    assert chunks[0].text[3:] == chunks[1].text[:2]
    assert chunks[1].text[3:] == chunks[2].text[:2]


def test_multiple_documents_preserve_order_and_copy_metadata() -> None:
    first_metadata = {"source": "explicit", "nested": {"value": 1}}
    documents = [
        Document(
            id="doc-b",
            text="abcdef",
            metadata=first_metadata,
            source="fallback.md",
        ),
        Document(id="doc-a", text="xyz", metadata={"kind": "note"}),
    ]

    chunks = FixedSizeChunker(size=4, overlap=1).chunk(documents)

    assert [chunk.document_id for chunk in chunks] == ["doc-b", "doc-b", "doc-a"]
    assert [chunk.start_offset for chunk in chunks] == [0, 3, 0]
    assert chunks[0].metadata["source"] == "explicit"
    assert chunks[2].metadata == {"kind": "note"}
    assert chunks[0].metadata is not chunks[1].metadata


def test_unicode_offsets_count_characters() -> None:
    chunks = FixedSizeChunker(size=2, overlap=1).chunk(
        [Document(id="日本語", text="検索品質")]
    )

    assert [(chunk.text, chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        ("検索", 0, 2),
        ("索品", 1, 3),
        ("品質", 2, 4),
    ]


def test_chunk_ids_are_deterministic_and_input_sensitive() -> None:
    chunker = FixedSizeChunker(size=4, overlap=1)
    original = Document(id="doc-1", text="abcdef")

    first = chunker.chunk([original])
    second = chunker.chunk([original])
    changed_text = chunker.chunk([Document(id="doc-1", text="abcxef")])
    changed_document = chunker.chunk([Document(id="doc-2", text="abcdef")])

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert first[0].id != changed_text[0].id
    assert first[0].id != changed_document[0].id
    assert all(len(chunk.id) == 24 for chunk in first)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (True, 0),
        (4, -1),
        (4, 4),
        (4, 5),
        (4, True),
    ],
)
def test_invalid_configuration_raises_library_error(size: int, overlap: int) -> None:
    with pytest.raises(CorpusValidationError):
        FixedSizeChunker(size=size, overlap=overlap)


def test_chunking_rejects_invalid_document_inputs_with_library_error() -> None:
    chunker = FixedSizeChunker(size=4, overlap=1)

    with pytest.raises(CorpusValidationError, match="sequence"):
        chunker.chunk(cast(Sequence[Document], "not documents"))
    with pytest.raises(CorpusValidationError, match=r"documents\[0\]"):
        chunker.chunk(cast(Sequence[Document], [object()]))
    with pytest.raises(CorpusValidationError, match="document must"):
        chunker.chunk_document(cast(Document, object()))
