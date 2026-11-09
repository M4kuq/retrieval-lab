from pathlib import Path

import pytest

from retrieval_lab import (
    DatasetDraft,
    DatasetProvenance,
    DraftQuery,
    EvaluationDataset,
    EvaluationQuery,
)
from retrieval_lab.exceptions import DatasetValidationError


def test_synthetic_unreviewed_provenance_is_experimental() -> None:
    provenance = DatasetProvenance(
        origin="synthetic",
        review_status="unreviewed",
        generator="test-generator",
    )

    assert not provenance.human_reviewed
    assert provenance.reliability == "experimental"
    assert provenance.to_dict()["human_reviewed"] is False


def test_reviewed_human_provenance_is_reviewed() -> None:
    provenance = DatasetProvenance(origin="human", review_status="reviewed")

    assert provenance.human_reviewed
    assert provenance.reliability == "reviewed"


def test_draft_allows_incomplete_queries_but_blocks_finalize() -> None:
    draft = DatasetDraft([DraftQuery("q-1", "query")])

    assert not draft.complete
    assert draft.pending_query_ids == ("q-1",)
    with pytest.raises(DatasetValidationError, match="pending relevance"):
        draft.finalize()


def test_draft_query_relevance_editing_is_immutable() -> None:
    original = DraftQuery("q-1", "query")

    updated = original.with_relevance("doc-1", 3)
    removed = updated.without_relevance("doc-1")

    assert original.relevance == {}
    assert updated.relevance == {"doc-1": 3}
    assert removed.relevance == {}


def test_finalize_builds_document_level_evaluation_dataset() -> None:
    draft = DatasetDraft(
        [
            DraftQuery(
                "q-1",
                "query",
                relevance={"doc-1": 3, "doc-2": 1},
                reference_answer="answer",
                metadata={"category": "test"},
            )
        ],
        provenance=DatasetProvenance(origin="human", review_status="reviewed"),
    )

    dataset = draft.finalize()

    assert isinstance(dataset, EvaluationDataset)
    assert dataset.queries[0].relevant_document_ids == {"doc-1", "doc-2"}
    assert dataset.relevance_for("q-1") == {"doc-1": 3, "doc-2": 1}
    assert dataset.reference_answer_for("q-1") == "answer"


def test_finalize_builds_chunk_level_evaluation_dataset() -> None:
    draft = DatasetDraft(
        [DraftQuery("q-1", "query", relevance={"chunk-1": 2})],
        relevance_level="chunk",
    )

    dataset = draft.finalize()

    assert dataset.relevance_level == "chunk"
    assert dataset.queries[0].relevant_chunk_ids == {"chunk-1"}
    assert dataset.queries[0].relevant_document_ids == frozenset()


def test_replace_query_and_add_query_return_new_drafts() -> None:
    original = DatasetDraft([DraftQuery("q-1", "query")])
    replaced = original.replace_query(
        DraftQuery("q-1", "query", relevance={"doc-1": 1})
    )
    added = replaced.add_query(DraftQuery("q-2", "second"))

    assert original.pending_query_ids == ("q-1",)
    assert replaced.pending_query_ids == ()
    assert [query.id for query in added.queries] == ["q-1", "q-2"]


def test_to_jsonl_is_deterministic_and_compatible_with_loader(tmp_path: Path) -> None:
    draft = DatasetDraft(
        [
            DraftQuery(
                "q-1",
                "query",
                relevance={"doc-2": 1, "doc-1": 3},
                reference_answer="answer",
                metadata={"b": 2, "a": 1},
            )
        ]
    )
    content = draft.to_jsonl()
    path = tmp_path / "evaluation.jsonl"
    path.write_text(content, encoding="utf-8")

    loaded = EvaluationDataset.from_jsonl(path)

    assert content == draft.to_jsonl()
    assert loaded.relevance_for("q-1") == {"doc-1": 3, "doc-2": 1}
    assert loaded.reference_answer_for("q-1") == "answer"


def test_save_bundle_records_provenance_and_pending_queries(tmp_path: Path) -> None:
    draft = DatasetDraft(
        [DraftQuery("q-1", "query")],
        provenance=DatasetProvenance(
            origin="synthetic",
            generator="fake-generator",
        ),
    )

    dataset_path, manifest_path = draft.save_bundle(tmp_path / "bundle")

    assert dataset_path.read_text(encoding="utf-8") == draft.to_jsonl()
    manifest = manifest_path.read_text(encoding="utf-8")
    assert '"origin": "synthetic"' in manifest
    assert '"reliability": "experimental"' in manifest
    assert '"q-1"' in manifest


def test_from_dataset_requires_explicit_provenance_for_trust() -> None:
    dataset = EvaluationDataset(
        [EvaluationQuery("q-1", "query", relevant_document_ids={"doc-1"})]
    )

    draft = DatasetDraft.from_dataset(dataset)

    assert draft.complete
    assert draft.provenance.origin == "unknown"
    assert draft.provenance.reliability == "unverified"


@pytest.mark.parametrize(
    ("origin", "review_status"),
    [("ai", "reviewed"), ("human", "done")],
)
def test_provenance_rejects_unknown_enums(origin: str, review_status: str) -> None:
    with pytest.raises(DatasetValidationError):
        DatasetProvenance(
            origin=origin,  # type: ignore[arg-type]
            review_status=review_status,  # type: ignore[arg-type]
        )


def test_draft_rejects_duplicate_query_ids() -> None:
    query = DraftQuery("q-1", "query")

    with pytest.raises(DatasetValidationError, match="unique"):
        DatasetDraft([query, query])


@pytest.mark.parametrize("field_name", ["generator", "notes"])
@pytest.mark.parametrize("value", ["", "   ", 123])
def test_provenance_rejects_invalid_optional_text(
    field_name: str, value: object
) -> None:
    with pytest.raises(DatasetValidationError, match="non-empty string or None"):
        DatasetProvenance(**{field_name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("id_value", "query_value"),
    [("", "query"), ("   ", "query"), ("q-1", ""), (123, "query")],
)
def test_draft_query_rejects_invalid_id_or_query(
    id_value: object, query_value: object
) -> None:
    with pytest.raises(DatasetValidationError, match="must be non-empty"):
        DraftQuery(id_value, query_value)  # type: ignore[arg-type]


def test_draft_query_rejects_non_mapping_relevance() -> None:
    with pytest.raises(DatasetValidationError, match="relevance must be a mapping"):
        DraftQuery("q-1", "query", relevance=[("doc-1", 1)])  # type: ignore[arg-type]


@pytest.mark.parametrize("key", ["", "   ", 123])
def test_draft_query_rejects_invalid_relevance_keys(key: object) -> None:
    with pytest.raises(DatasetValidationError, match="keys must be non-empty"):
        DraftQuery("q-1", "query", relevance={key: 1})  # type: ignore[dict-item]


@pytest.mark.parametrize("grade", [True, 0, -1, "1", 1.5])
def test_draft_query_rejects_invalid_relevance_grades(grade: object) -> None:
    with pytest.raises(DatasetValidationError, match="grades must be integers >= 1"):
        DraftQuery("q-1", "query", relevance={"doc-1": grade})  # type: ignore[dict-item]


@pytest.mark.parametrize("answer", ["", "   ", 123])
def test_draft_query_rejects_invalid_reference_answer(answer: object) -> None:
    with pytest.raises(DatasetValidationError, match="reference_answer must be"):
        DraftQuery("q-1", "query", reference_answer=answer)  # type: ignore[arg-type]


def test_draft_rejects_invalid_relevance_level() -> None:
    with pytest.raises(DatasetValidationError, match="relevance_level must be"):
        DatasetDraft([], relevance_level="passage")  # type: ignore[arg-type]


@pytest.mark.parametrize("queries", ["q-1", b"q-1", 123, None])
def test_draft_rejects_non_sequence_queries(queries: object) -> None:
    with pytest.raises(DatasetValidationError, match="queries must be a sequence"):
        DatasetDraft(queries)  # type: ignore[arg-type]


def test_draft_rejects_non_draft_query_elements() -> None:
    with pytest.raises(
        DatasetValidationError, match=r"queries\[0\] must be a DraftQuery"
    ):
        DatasetDraft([object()])  # type: ignore[list-item]


def test_draft_rejects_invalid_provenance_type() -> None:
    with pytest.raises(DatasetValidationError, match="provenance must be"):
        DatasetDraft([], provenance=object())  # type: ignore[arg-type]


def test_replace_query_rejects_unknown_id() -> None:
    draft = DatasetDraft([DraftQuery("q-1", "query")])

    with pytest.raises(DatasetValidationError, match="unknown draft query ID"):
        draft.replace_query(DraftQuery("q-2", "other"))


def test_finalize_rejects_empty_draft() -> None:
    with pytest.raises(DatasetValidationError, match="cannot finalize an empty"):
        DatasetDraft().finalize()


def test_save_bundle_rejects_uncreatable_directory(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")
    draft = DatasetDraft([DraftQuery("q-1", "query")])

    with pytest.raises(DatasetValidationError, match="cannot create dataset bundle"):
        draft.save_bundle(blocker / "bundle")


def test_from_dataset_rejects_non_dataset() -> None:
    with pytest.raises(DatasetValidationError, match="must be an EvaluationDataset"):
        DatasetDraft.from_dataset(object())  # type: ignore[arg-type]


def test_save_bundle_reports_write_failure(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "evaluation.jsonl").mkdir()
    draft = DatasetDraft([DraftQuery("q-1", "query")])

    with pytest.raises(
        DatasetValidationError, match="cannot write dataset artifact"
    ) as raised:
        draft.save_bundle(bundle)
    assert isinstance(raised.value.__cause__, OSError)
