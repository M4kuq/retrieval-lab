from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from retrieval_lab.datasets import EvaluationDataset, validate_dataset
from retrieval_lab.domain import Chunk, Document, EvaluationQuery
from retrieval_lab.exceptions import CorpusValidationError, DatasetValidationError


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_from_jsonl_preserves_grades_answer_metadata_and_is_immutable(
    tmp_path: Path,
) -> None:
    source = _write(
        tmp_path / "evaluation.jsonl",
        '{"query_id":"q-1","query":"再設定方法は?",'
        '"relevant":[{"id":"doc-2","relevance":1},'
        '{"id":"doc-1","relevance":3}],'
        '"reference_answer":"設定画面から行う。",'
        '"metadata":{"category":"how-to"}}\n',
    )

    dataset = EvaluationDataset.from_jsonl(source)

    assert isinstance(dataset.queries, tuple)
    assert dataset.relevance_level == "document"
    assert dataset.queries[0].relevant_document_ids == frozenset({"doc-1", "doc-2"})
    assert dataset.queries[0].metadata == {"category": "how-to"}
    assert dataset.relevance_for("q-1") == {"doc-1": 3, "doc-2": 1}
    assert dataset.reference_answer_for("q-1") == "設定画面から行う。"
    with pytest.raises(TypeError):
        dataset.relevance_grades_by_query["q-1"]["doc-1"] = 9  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        dataset.relevance_level = "chunk"  # type: ignore[misc]


def test_from_jsonl_supports_chunk_relevance_and_optional_answer(
    tmp_path: Path,
) -> None:
    source = _write(
        tmp_path / "evaluation.jsonl",
        '{"query_id":"q-1","query":"query",'
        '"relevant":[{"id":"chunk-1","relevance":2}]}\n',
    )

    dataset = EvaluationDataset.from_jsonl(source, relevance_level="chunk")

    assert dataset.queries[0].relevant_chunk_ids == frozenset({"chunk-1"})
    assert dataset.queries[0].relevant_document_ids == frozenset()
    assert dataset.reference_answer_for("q-1") is None


def test_constructor_adds_default_binary_grades() -> None:
    query = EvaluationQuery("q-1", "query", relevant_document_ids={"doc-1"})

    dataset = EvaluationDataset([query])

    assert dataset.relevance_grades_by_query == {"q-1": {"doc-1": 1}}


def test_constructor_rejects_duplicate_query_ids() -> None:
    query = EvaluationQuery("q-1", "query", relevant_document_ids={"doc-1"})

    with pytest.raises(DatasetValidationError, match="unique"):
        EvaluationDataset([query, query])


def test_constructor_requires_relevance_at_selected_level() -> None:
    query = EvaluationQuery("q-1", "query", relevant_chunk_ids={"chunk-1"})

    with pytest.raises(DatasetValidationError, match="document relevance"):
        EvaluationDataset([query], relevance_level="document")


@pytest.mark.parametrize("level", ["documents", "", 1, None])
def test_constructor_rejects_invalid_relevance_level(level: object) -> None:
    query = EvaluationQuery("q-1", "query", relevant_document_ids={"doc-1"})

    with pytest.raises(DatasetValidationError, match="relevance_level"):
        EvaluationDataset([query], relevance_level=level)  # type: ignore[arg-type]


def test_constructor_requires_grade_mapping_to_match_positive_ids() -> None:
    query = EvaluationQuery("q-1", "query", relevant_document_ids={"doc-1"})

    with pytest.raises(DatasetValidationError, match="exactly match"):
        EvaluationDataset(
            [query],
            relevance_grades_by_query={"q-1": {"other": 1}},
        )


@pytest.mark.parametrize("grade", [True, False, 0, -1, 1.5, "2"])
def test_constructor_rejects_invalid_relevance_grade(grade: object) -> None:
    query = EvaluationQuery("q-1", "query", relevant_document_ids={"doc-1"})

    with pytest.raises(DatasetValidationError, match="integer >= 1"):
        EvaluationDataset(
            [query],
            relevance_grades_by_query={"q-1": {"doc-1": grade}},  # type: ignore[dict-item]
        )


def test_document_level_validation_accepts_complete_corpus() -> None:
    query = EvaluationQuery("q-1", "query", relevant_document_ids={"doc-1"})
    dataset = EvaluationDataset([query])

    assert dataset.validate(documents=[Document("doc-1", "text")]) is None


def test_document_level_validation_reports_all_missing_gold_ids() -> None:
    query = EvaluationQuery(
        "q-1",
        "query",
        relevant_document_ids={"missing-1", "missing-2"},
    )
    dataset = EvaluationDataset([query])

    with pytest.raises(DatasetValidationError, match="missing-1, missing-2"):
        validate_dataset(dataset, documents=[Document("doc-1", "text")])


def test_chunk_level_validation_uses_chunk_ids() -> None:
    query = EvaluationQuery("q-1", "query", relevant_chunk_ids={"chunk-1"})
    dataset = EvaluationDataset([query], relevance_level="chunk")
    chunk = Chunk("chunk-1", "doc-1", "text", 0, 4)

    assert dataset.validate(chunks=[chunk]) is None


def test_validation_rejects_empty_and_duplicate_corpus_records() -> None:
    query = EvaluationQuery("q-1", "query", relevant_document_ids={"doc-1"})
    dataset = EvaluationDataset([query])
    document = Document("doc-1", "text")

    with pytest.raises(CorpusValidationError, match="must not be empty"):
        dataset.validate(documents=[])
    with pytest.raises(CorpusValidationError, match="duplicate document ID"):
        dataset.validate(documents=[document, document])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "must not be empty"),
        ("{broken}\n", "invalid JSON"),
        ("[]\n", "must be a JSON object"),
        (
            '{"query_id":"q","query":"x","relevant":[],"extra":1}\n',
            "unknown=['extra']",
        ),
        (
            '{"query_id":"q","query":"x","relevant":[]}\n',
            "non-empty array",
        ),
        (
            '{"query_id":"q","query":"x","relevant":[{"id":"doc","relevance":true}]}\n',
            "booleans",
        ),
        (
            '{"query_id":"q","query":"x","relevant":'
            '[{"id":"doc","relevance":1},{"id":"doc","relevance":2}]}\n',
            "duplicate relevant ID",
        ),
    ],
)
def test_from_jsonl_wraps_schema_errors_with_path_and_line(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    source = _write(tmp_path / "bad.jsonl", content)

    with pytest.raises(DatasetValidationError) as captured:
        EvaluationDataset.from_jsonl(source)

    assert f"{source}:1:" in str(captured.value)
    assert message in str(captured.value)


def test_from_jsonl_rejects_duplicate_query_id_on_later_line(tmp_path: Path) -> None:
    record = (
        '{"query_id":"q-1","query":"query","relevant":[{"id":"doc-1","relevance":1}]}\n'
    )
    source = _write(tmp_path / "duplicate.jsonl", record + record)

    with pytest.raises(DatasetValidationError, match=r":2: duplicate query_id"):
        EvaluationDataset.from_jsonl(source)


def test_from_jsonl_rejects_non_utf8_with_source_line(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_bytes(b'{"query_id":"q"}\n\xff')

    with pytest.raises(DatasetValidationError, match=r":2:.*UTF-8"):
        EvaluationDataset.from_jsonl(source)


def test_from_jsonl_normalizes_identifiers_and_text_to_nfc(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "evaluation.jsonl",
        '{"query_id":"q-e\u0301","query":"cafe\u0301",'
        '"relevant":[{"id":"doc-e\u0301","relevance":1}]}\n',
    )

    dataset = EvaluationDataset.from_jsonl(source)

    assert dataset.queries[0].id == "q-é"
    assert dataset.queries[0].query == "café"
    assert dataset.queries[0].relevant_document_ids == {"doc-é"}


def test_unknown_query_lookup_uses_package_exception() -> None:
    query = EvaluationQuery("q-1", "query", relevant_document_ids={"doc-1"})
    dataset = EvaluationDataset([query])

    with pytest.raises(DatasetValidationError, match="unknown query_id"):
        dataset.relevance_for("missing")
    with pytest.raises(DatasetValidationError, match="unknown query_id"):
        dataset.reference_answer_for("missing")
