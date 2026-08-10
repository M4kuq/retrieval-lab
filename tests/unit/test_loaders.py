from pathlib import Path

import pytest

from retrieval_lab.exceptions import CorpusValidationError
from retrieval_lab.loaders import load_documents


def test_load_single_text_uses_stable_relative_path_id_and_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "manual.txt"
    source.write_text("本文\r\n続き", encoding="utf-8", newline="")

    first = load_documents(source)
    second = load_documents(source)

    assert first == second
    assert first[0].id == "manual.txt"
    assert first[0].source == "manual.txt"
    assert first[0].text == "本文\n続き"


def test_load_directory_recurses_and_orders_by_posix_relative_source(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "nested" / "b.markdown").write_text("b", encoding="utf-8")
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "ignored.pdf").write_bytes(b"not loaded")

    documents = load_documents(tmp_path)

    assert [document.id for document in documents] == [
        "a.md",
        "nested/b.markdown",
        "z.txt",
    ]
    assert [document.source for document in documents] == [
        "a.md",
        "nested/b.markdown",
        "z.txt",
    ]


def test_load_text_normalizes_unicode_to_nfc(tmp_path: Path) -> None:
    source = tmp_path / "unicode.md"
    source.write_text("cafe\u0301", encoding="utf-8")

    document = load_documents(source)[0]

    assert document.text == "café"


def test_load_corpus_jsonl_preserves_ids_metadata_order_and_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corpus.jsonl"
    source.write_text(
        '{"id":"doc-2","text":"二番目","metadata":{"section":2}}\n'
        '{"id":"doc-1","text":"一番目"}\n',
        encoding="utf-8",
    )

    documents = load_documents(source)

    assert [document.id for document in documents] == ["doc-2", "doc-1"]
    assert documents[0].metadata == {"section": 2}
    assert documents[1].metadata == {}
    assert all(document.source == "corpus.jsonl" for document in documents)


def test_load_directory_detects_duplicate_ids_across_jsonl_files(
    tmp_path: Path,
) -> None:
    record = '{"id":"duplicate","text":"text"}\n'
    (tmp_path / "a.jsonl").write_text(record, encoding="utf-8")
    (tmp_path / "b.jsonl").write_text(record, encoding="utf-8")

    with pytest.raises(CorpusValidationError, match="duplicate document ID"):
        load_documents(tmp_path)


def test_load_jsonl_detects_duplicate_ids_after_nfc_normalization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corpus.jsonl"
    source.write_text(
        '{"id":"doc-e\u0301","text":"one"}\n{"id":"doc-é","text":"two"}\n',
        encoding="utf-8",
    )

    with pytest.raises(CorpusValidationError, match=r":2: duplicate document ID"):
        load_documents(source)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "must not be empty"),
        ("{broken}\n", "invalid JSON"),
        ("[]\n", "must be a JSON object"),
        ('{"id":"doc-1"}\n', "missing=['text']"),
        ('{"id":"doc-1","text":"x","extra":1}\n', "unknown=['extra']"),
        ('{"id":"doc-1","text":" "}\n', "text must be a non-empty string"),
        ('{"id":"doc-1","text":"x","metadata":[]}\n', "metadata"),
        ('{"id":"doc-1","id":"doc-2","text":"x"}\n', "duplicate JSON key"),
    ],
)
def test_load_jsonl_wraps_invalid_records_with_path_and_line(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(content, encoding="utf-8")

    with pytest.raises(CorpusValidationError) as captured:
        load_documents(source)

    assert f"{source}:1:" in str(captured.value)
    assert message in str(captured.value)


def test_load_rejects_empty_text_file(tmp_path: Path) -> None:
    source = tmp_path / "empty.md"
    source.write_text(" \n", encoding="utf-8")

    with pytest.raises(CorpusValidationError, match="document text must not be empty"):
        load_documents(source)


def test_load_rejects_non_utf8_with_line(tmp_path: Path) -> None:
    source = tmp_path / "bad.txt"
    source.write_bytes(b"valid\n\xff")

    with pytest.raises(CorpusValidationError, match=r":2:.*UTF-8"):
        load_documents(source)


def test_include_filters_files_before_reading_excluded_invalid_utf8(
    tmp_path: Path,
) -> None:
    (tmp_path / "included.txt").write_text("valid", encoding="utf-8")
    (tmp_path / "excluded.txt").write_bytes(b"invalid \xff UTF-8")

    documents = load_documents(tmp_path, include=("included.txt",))

    assert [document.source for document in documents] == ["included.txt"]


def test_include_no_match_reports_actionable_error_for_file_and_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "single.txt"
    source.write_text("valid", encoding="utf-8")
    with pytest.raises(CorpusValidationError, match="include matched no documents"):
        load_documents(source, include=("missing.txt",))

    with pytest.raises(CorpusValidationError, match="include matched no documents"):
        load_documents(tmp_path, include=("missing.txt",))


def test_load_rejects_unsupported_single_file(tmp_path: Path) -> None:
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"pdf")

    with pytest.raises(CorpusValidationError, match="unsupported corpus file type"):
        load_documents(source)


def test_load_rejects_directory_without_supported_files(tmp_path: Path) -> None:
    (tmp_path / "manual.pdf").write_bytes(b"pdf")

    with pytest.raises(CorpusValidationError, match="no supported files"):
        load_documents(tmp_path)


def test_load_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(CorpusValidationError, match="does not exist"):
        load_documents(tmp_path / "missing")
