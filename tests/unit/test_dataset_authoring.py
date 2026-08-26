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
