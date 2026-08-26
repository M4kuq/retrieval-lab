import pytest

from retrieval_lab import (
    CallableSyntheticGenerator,
    Document,
    DraftQuery,
    generate_synthetic_draft,
    mark_draft_in_review,
    mark_draft_reviewed,
)
from retrieval_lab.exceptions import DatasetValidationError


def _generator(documents, count, seed):
    return [
        DraftQuery(
            f"q-{seed}-{index}",
            f"question {index}",
            relevance={documents[index % len(documents)].id: 1},
        )
        for index in range(count)
    ]


def test_generate_synthetic_draft_forces_experimental_unreviewed_provenance() -> None:
    generator = CallableSyntheticGenerator("deterministic-test", _generator)

    draft = generate_synthetic_draft(
        [Document("doc-1", "text")],
        generator=generator,
        count=2,
        seed=7,
    )

    assert draft.complete
    assert draft.provenance.origin == "synthetic"
    assert draft.provenance.review_status == "unreviewed"
    assert draft.provenance.reliability == "experimental"
    assert draft.provenance.generator == "deterministic-test"
    assert [query.id for query in draft.queries] == ["q-7-0", "q-7-1"]


def test_mark_reviewed_preserves_synthetic_origin() -> None:
    draft = generate_synthetic_draft(
        [Document("doc-1", "text")],
        generator=CallableSyntheticGenerator("test", _generator),
        count=1,
    )

    in_review = mark_draft_in_review(draft, notes="checking relevance")
    reviewed = mark_draft_reviewed(in_review, notes="checked by human")

    assert in_review.provenance.review_status == "in_review"
    assert reviewed.provenance.origin == "synthetic"
    assert reviewed.provenance.review_status == "reviewed"
    assert reviewed.provenance.reliability == "reviewed"
    assert reviewed.provenance.notes == "checked by human"


def test_generator_must_return_requested_count() -> None:
    generator = CallableSyntheticGenerator("short", lambda docs, count, seed: [])

    with pytest.raises(DatasetValidationError, match="expected 1"):
        generate_synthetic_draft(
            [Document("doc-1", "text")],
            generator=generator,
            count=1,
        )


def test_generated_relevance_must_reference_source_documents() -> None:
    generator = CallableSyntheticGenerator(
        "invalid",
        lambda docs, count, seed: [
            DraftQuery("q-1", "query", relevance={"missing": 1})
        ],
    )

    with pytest.raises(DatasetValidationError, match="unknown documents"):
        generate_synthetic_draft(
            [Document("doc-1", "text")],
            generator=generator,
            count=1,
        )


def test_generated_queries_must_include_positive_relevance() -> None:
    generator = CallableSyntheticGenerator(
        "incomplete",
        lambda docs, count, seed: [DraftQuery("q-1", "query")],
    )

    with pytest.raises(DatasetValidationError, match="positive relevance"):
        generate_synthetic_draft(
            [Document("doc-1", "text")],
            generator=generator,
            count=1,
        )


def test_generator_exception_is_wrapped() -> None:
    def failing(documents, count, seed):
        raise RuntimeError("provider failed")

    with pytest.raises(DatasetValidationError, match="provider failed") as captured:
        generate_synthetic_draft(
            [Document("doc-1", "text")],
            generator=CallableSyntheticGenerator("failing", failing),
            count=1,
        )

    assert isinstance(captured.value.__cause__, RuntimeError)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_count_must_be_positive_integer(count: object) -> None:
    with pytest.raises(DatasetValidationError, match="count"):
        generate_synthetic_draft(
            [Document("doc-1", "text")],
            generator=CallableSyntheticGenerator("test", _generator),
            count=count,  # type: ignore[arg-type]
        )


def test_source_document_ids_must_be_unique() -> None:
    document = Document("doc-1", "text")

    with pytest.raises(DatasetValidationError, match="unique"):
        generate_synthetic_draft(
            [document, document],
            generator=CallableSyntheticGenerator("test", _generator),
            count=1,
        )
