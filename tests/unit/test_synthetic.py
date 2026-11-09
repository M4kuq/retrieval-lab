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


@pytest.mark.parametrize("name", ["", "   ", 42])
def test_callable_generator_name_must_be_non_empty_string(name: object) -> None:
    with pytest.raises(DatasetValidationError, match="name must be a non-empty"):
        CallableSyntheticGenerator(name, _generator)  # type: ignore[arg-type]


def test_callable_generator_callback_must_be_callable() -> None:
    with pytest.raises(DatasetValidationError, match="callback must be callable"):
        CallableSyntheticGenerator("test", "not-callable")  # type: ignore[arg-type]


@pytest.mark.parametrize("documents", ["doc-1", 42])
def test_documents_must_be_a_sequence(documents: object) -> None:
    with pytest.raises(DatasetValidationError, match="sequence of Document"):
        generate_synthetic_draft(
            documents,  # type: ignore[arg-type]
            generator=CallableSyntheticGenerator("test", _generator),
            count=1,
        )


def test_documents_must_not_be_empty() -> None:
    with pytest.raises(DatasetValidationError, match="must not be empty"):
        generate_synthetic_draft(
            [],
            generator=CallableSyntheticGenerator("test", _generator),
            count=1,
        )


def test_documents_must_be_document_values() -> None:
    with pytest.raises(DatasetValidationError, match=r"documents\[1\] must be"):
        generate_synthetic_draft(
            [Document("doc-1", "text"), "not-a-document"],  # type: ignore[list-item]
            generator=CallableSyntheticGenerator("test", _generator),
            count=1,
        )


def test_generator_output_must_be_a_sequence() -> None:
    generator = CallableSyntheticGenerator("text", lambda docs, count, seed: "nope")

    with pytest.raises(DatasetValidationError, match="output must be a sequence"):
        generate_synthetic_draft(
            [Document("doc-1", "text")],
            generator=generator,
            count=1,
        )


def test_generator_output_items_must_be_draft_queries() -> None:
    generator = CallableSyntheticGenerator("junk", lambda docs, count, seed: ["x"])

    with pytest.raises(DatasetValidationError, match=r"output\[0\] must be"):
        generate_synthetic_draft(
            [Document("doc-1", "text")],
            generator=generator,
            count=1,
        )


def test_generated_query_ids_must_be_unique() -> None:
    duplicate = DraftQuery("q-1", "query", relevance={"doc-1": 1})
    generator = CallableSyntheticGenerator(
        "duplicates", lambda docs, count, seed: [duplicate, duplicate]
    )

    with pytest.raises(DatasetValidationError, match="query IDs must be unique"):
        generate_synthetic_draft(
            [Document("doc-1", "text")],
            generator=generator,
            count=2,
        )


def test_generator_dataset_validation_error_is_not_wrapped() -> None:
    def invalid(documents, count, seed):
        raise DatasetValidationError("inner validation problem")

    with pytest.raises(DatasetValidationError) as captured:
        generate_synthetic_draft(
            [Document("doc-1", "text")],
            generator=CallableSyntheticGenerator("invalid", invalid),
            count=1,
        )

    assert str(captured.value) == "inner validation problem"


class _NamelessGenerator:
    def generate(self, documents, *, count, seed):
        return []


class _BlankNameGenerator:
    name = ""

    def generate(self, documents, *, count, seed):
        return []


class _NonCallableGenerate:
    name = "broken"
    generate = None


@pytest.mark.parametrize(
    "generator", [_NamelessGenerator(), _BlankNameGenerator(), _NonCallableGenerate()]
)
def test_generator_must_provide_name_and_generate(generator: object) -> None:
    with pytest.raises(DatasetValidationError, match=r"non-empty name and .*generate"):
        generate_synthetic_draft(
            [Document("doc-1", "text")],
            generator=generator,  # type: ignore[arg-type]
            count=1,
        )


@pytest.mark.parametrize("marker", [mark_draft_in_review, mark_draft_reviewed])
def test_mark_draft_requires_a_dataset_draft(marker: object) -> None:
    with pytest.raises(DatasetValidationError, match="draft must be a DatasetDraft"):
        marker("not-a-draft")  # type: ignore[operator]


def test_mark_draft_without_notes_keeps_existing_notes() -> None:
    draft = generate_synthetic_draft(
        [Document("doc-1", "text")],
        generator=CallableSyntheticGenerator("test", _generator),
        count=1,
    )

    in_review = mark_draft_in_review(draft)
    reviewed = mark_draft_reviewed(draft)

    assert in_review.provenance.notes == draft.provenance.notes
    assert reviewed.provenance.notes == draft.provenance.notes


@pytest.mark.parametrize("notes", ["", "   ", 42])
def test_review_notes_must_be_non_empty_or_none(notes: object) -> None:
    draft = generate_synthetic_draft(
        [Document("doc-1", "text")],
        generator=CallableSyntheticGenerator("test", _generator),
        count=1,
    )

    with pytest.raises(DatasetValidationError, match="notes must be non-empty"):
        mark_draft_reviewed(draft, notes=notes)  # type: ignore[arg-type]
