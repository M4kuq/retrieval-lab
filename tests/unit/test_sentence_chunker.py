"""Tests for sentence-boundary chunking."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from retrieval_lab import (
    Chunk,
    ChunkerConfig,
    Document,
    EvaluationQuery,
    EvaluationResult,
    EvaluationRunner,
    FixedSizeChunker,
    SentenceBoundaryChunker,
    load_config,
)
from retrieval_lab.exceptions import ConfigurationError, CorpusValidationError


def _tuples(chunks: Sequence[Chunk]) -> list[tuple[str, int, int]]:
    return [(chunk.text, chunk.start_offset, chunk.end_offset) for chunk in chunks]


def test_japanese_sentences_pack_without_splitting() -> None:
    document = Document(id="doc-1", text="あ。い。う。え。", source="manual.md")

    chunks = SentenceBoundaryChunker(size=5, overlap=0).chunk([document])

    assert _tuples(chunks) == [("あ。い。", 0, 4), ("う。え。", 4, 8)]
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].metadata == {"source": "manual.md"}


def test_english_sentences_pack_in_order() -> None:
    chunks = SentenceBoundaryChunker(size=9, overlap=0).chunk(
        [Document(id="doc-1", text="One. Two. Three.")]
    )

    assert _tuples(chunks) == [("One. Two.", 0, 9), (" Three.", 9, 16)]


def test_overlap_repeats_trailing_sentence() -> None:
    chunks = SentenceBoundaryChunker(size=4, overlap=2).chunk(
        [Document(id="doc-1", text="あ。い。う。え。")]
    )

    assert _tuples(chunks) == [
        ("あ。い。", 0, 4),
        ("い。う。", 2, 6),
        ("う。え。", 4, 8),
    ]


def test_mixed_terminators_and_closing_quote() -> None:
    text = 'He said "Go." Then left! Really? Yes\uff0e'
    chunks = SentenceBoundaryChunker(size=13, overlap=0).chunk(
        [Document(id="doc-1", text=text)]
    )

    assert _tuples(chunks) == [
        ('He said "Go."', 0, 13),
        (" Then left!", 13, 24),
        (" Really? Yes\uff0e", 24, 37),
    ]


def test_decimal_point_is_not_a_sentence_boundary() -> None:
    chunks = SentenceBoundaryChunker(size=17, overlap=0).chunk(
        [Document(id="doc-1", text="Pi is 3.14 today. Done.")]
    )

    assert _tuples(chunks) == [("Pi is 3.14 today.", 0, 17), (" Done.", 17, 23)]


def test_oversized_sentence_is_split_at_size() -> None:
    chunks = SentenceBoundaryChunker(size=4, overlap=0).chunk(
        [Document(id="doc-1", text="abcdefg。")]
    )

    assert _tuples(chunks) == [("abcd", 0, 4), ("efg。", 4, 8)]


def test_text_without_terminator_is_one_sentence() -> None:
    chunks = SentenceBoundaryChunker(size=4, overlap=0).chunk(
        [Document(id="doc-1", text="abcdefghij")]
    )

    assert _tuples(chunks) == [("abcd", 0, 4), ("efgh", 4, 8), ("ij", 8, 10)]


def test_trailing_whitespace_is_folded_into_final_chunk() -> None:
    chunks = SentenceBoundaryChunker(size=5, overlap=0).chunk(
        [Document(id="doc-1", text="One. Two.   ")]
    )

    assert _tuples(chunks) == [("One.", 0, 4), (" Two.   ", 4, 12)]


def test_whitespace_only_text_produces_no_chunks() -> None:
    document = object.__new__(Document)
    object.__setattr__(document, "id", "doc-1")
    object.__setattr__(document, "text", "   ")
    object.__setattr__(document, "metadata", {})
    object.__setattr__(document, "source", None)

    assert SentenceBoundaryChunker(size=4, overlap=1).chunk([document]) == []


def test_multiple_documents_preserve_order_and_copy_metadata() -> None:
    documents = [
        Document(id="doc-b", text="A. B.", metadata={"kind": "note"}),
        Document(id="doc-a", text="あ。", source="fallback.md"),
    ]

    chunks = SentenceBoundaryChunker(size=3, overlap=0).chunk(documents)

    assert [chunk.document_id for chunk in chunks] == ["doc-b", "doc-b", "doc-a"]
    assert chunks[0].metadata == {"kind": "note"}
    assert chunks[0].metadata is not chunks[1].metadata
    assert chunks[2].metadata == {"source": "fallback.md"}


def test_chunk_ids_are_deterministic_and_input_sensitive() -> None:
    chunker = SentenceBoundaryChunker(size=4, overlap=2)
    original = Document(id="doc-1", text="あ。い。う。")

    first = chunker.chunk([original])
    second = chunker.chunk([original])
    changed = chunker.chunk([Document(id="doc-1", text="あ。い。え。")])

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert first[-1].id != changed[-1].id
    assert all(len(chunk.id) == 24 for chunk in first)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(4, 0), (4, 2), (7, 3), (10, 0), (512, 64)],
)
def test_chunking_covers_text_and_respects_size(size: int, overlap: int) -> None:
    text = "検索は重要です。評価も重要です! Is 3.5 enough? Yes. 次に進みます。"
    document = Document(id="doc-1", text=text)

    chunks = SentenceBoundaryChunker(size=size, overlap=overlap).chunk_document(
        document
    )

    covered = [False] * len(text)
    for chunk in chunks:
        assert chunk.text == text[chunk.start_offset : chunk.end_offset]
        assert chunk.text.strip()
        assert len(chunk.text) <= size
        for offset in range(chunk.start_offset, chunk.end_offset):
            covered[offset] = True
    assert all(covered)
    assert len({chunk.id for chunk in chunks}) == len(chunks)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-1, 0), (True, 0), (4, -1), (4, 4), (4, 5), (4, True)],
)
def test_invalid_configuration_raises_library_error(size: int, overlap: int) -> None:
    with pytest.raises(CorpusValidationError):
        SentenceBoundaryChunker(size=size, overlap=overlap)


def test_chunking_rejects_invalid_document_inputs() -> None:
    chunker = SentenceBoundaryChunker(size=4, overlap=1)

    with pytest.raises(CorpusValidationError, match="sequence"):
        chunker.chunk(cast(Sequence[Document], "not documents"))
    with pytest.raises(CorpusValidationError, match=r"documents\[0\]"):
        chunker.chunk(cast(Sequence[Document], [object()]))
    with pytest.raises(CorpusValidationError, match="document must"):
        chunker.chunk_document(cast(Document, object()))


def test_whitespace_tail_of_oversized_sentence_terminates() -> None:
    text = "One. abcdefg\n"
    chunks = SentenceBoundaryChunker(size=4, overlap=0).chunk(
        [Document(id="doc-1", text=text)]
    )

    covered = [False] * len(text)
    for chunk in chunks:
        assert chunk.text == text[chunk.start_offset : chunk.end_offset]
        assert chunk.text.strip()
        for offset in range(chunk.start_offset, chunk.end_offset):
            covered[offset] = True
    assert all(covered)


def test_leading_whitespace_longer_than_size_terminates() -> None:
    text = "     abc. x."
    chunks = SentenceBoundaryChunker(size=4, overlap=0).chunk(
        [Document(id="doc-1", text=text)]
    )

    assert chunks
    assert chunks[0].start_offset == 0
    assert chunks[-1].end_offset == len(text)
    for chunk in chunks:
        assert chunk.text == text[chunk.start_offset : chunk.end_offset]
        assert chunk.text.strip()


def test_overlap_never_produces_a_chunk_without_new_text() -> None:
    chunks = SentenceBoundaryChunker(size=5, overlap=2).chunk(
        [Document(id="doc-1", text="あ。い。うえお。")]
    )

    assert _tuples(chunks) == [("あ。い。", 0, 4), ("うえお。", 4, 8)]


def test_closing_brackets_attach_to_their_sentence() -> None:
    chunks = SentenceBoundaryChunker(size=5, overlap=0).chunk(
        [Document(id="doc-1", text="『次。』その後。")]
    )

    assert _tuples(chunks) == [("『次。』", 0, 4), ("その後。", 4, 8)]


def test_period_followed_by_a_letter_is_not_a_boundary() -> None:
    chunks = SentenceBoundaryChunker(size=14, overlap=0).chunk(
        [Document(id="doc-1", text="See a.txt now. Done.")]
    )

    assert _tuples(chunks) == [("See a.txt now.", 0, 14), (" Done.", 14, 20)]


def test_fullwidth_period_splits_japanese_without_spaces() -> None:
    chunks = SentenceBoundaryChunker(size=3, overlap=0).chunk(
        [Document(id="doc-1", text="検索\uff0e評価\uff0e")]
    )

    assert _tuples(chunks) == [("検索\uff0e", 0, 3), ("評価\uff0e", 3, 6)]


def test_overlap_carry_is_kept_when_the_next_sentence_fits() -> None:
    chunks = SentenceBoundaryChunker(size=6, overlap=2).chunk(
        [Document(id="doc-1", text="あ。い。う。えおか。")]
    )

    assert _tuples(chunks) == [("あ。い。う。", 0, 6), ("う。えおか。", 4, 10)]


def test_fixed_chunker_runner_records_recursive_characters_type() -> None:
    result = EvaluationRunner(
        documents=[Document(id="doc-1", text="RAG is useful.")],
        queries=[
            EvaluationQuery(id="q-1", query="RAG", relevant_document_ids={"doc-1"})
        ],
        chunker=FixedSizeChunker(size=17, overlap=0),
        top_k=(1,),
    ).run()

    assert result.manifest["chunking"] == {
        "overlap": 0,
        "size": 17,
        "type": "recursive_characters",
    }


def test_identical_chunks_still_distinguish_chunker_type_in_run_id() -> None:
    def run(
        chunker: FixedSizeChunker | SentenceBoundaryChunker,
    ) -> EvaluationResult:
        return EvaluationRunner(
            documents=[Document(id="doc-1", text="RAG is useful.")],
            queries=[
                EvaluationQuery(id="q-1", query="RAG", relevant_document_ids={"doc-1"})
            ],
            chunker=chunker,
            top_k=(1,),
        ).run()

    fixed = run(FixedSizeChunker(size=17, overlap=0))
    sentence = run(SentenceBoundaryChunker(size=17, overlap=0))

    assert fixed.manifest["chunk_count"] == sentence.manifest["chunk_count"] == 1
    assert fixed.manifest["chunk_hash"] != sentence.manifest["chunk_hash"]
    assert fixed.run_id != sentence.run_id


def test_chunker_config_accepts_sentence_boundary() -> None:
    config = ChunkerConfig(type="sentence_boundary", size=32, overlap=4)

    assert config.type == "sentence_boundary"
    with pytest.raises(ConfigurationError, match=r"chunker\.type"):
        ChunkerConfig(type="other")  # type: ignore[arg-type]


def test_yaml_config_loads_sentence_boundary_and_runner_uses_it(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text(
        "one relevant. two more.", encoding="utf-8"
    )
    (tmp_path / "eval.jsonl").write_text(
        '{"query_id":"q1","query":"one","relevant":[{"id":"a.txt","relevance":1}]}\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "corpus:",
                "  path: docs",
                "  chunker:",
                "    type: sentence_boundary",
                "    size: 14",
                "    overlap: 0",
                "dataset:",
                "  path: eval.jsonl",
                "retrievers:",
                "  - name: keyword",
                "    type: keyword",
                "evaluation:",
                "  top_k: [1]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.corpus.chunker.type == "sentence_boundary"
    result = EvaluationRunner.from_config(config).run()
    assert result.manifest["chunk_count"] == 2
    assert result.manifest["chunking"] == {
        "overlap": 0,
        "size": 14,
        "type": "sentence_boundary",
    }


def test_runner_accepts_sentence_boundary_chunker_directly() -> None:
    result = EvaluationRunner(
        documents=[Document(id="doc-1", text="RAG is useful. It helps search.")],
        queries=[
            EvaluationQuery(id="q-1", query="RAG", relevant_document_ids={"doc-1"})
        ],
        chunker=SentenceBoundaryChunker(size=17, overlap=0),
        top_k=(1,),
    ).run()

    assert result.manifest["chunk_count"] == 2
