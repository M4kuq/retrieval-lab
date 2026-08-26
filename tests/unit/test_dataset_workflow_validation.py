import json
from pathlib import Path

import pytest

from retrieval_lab import DatasetDraft, DraftQuery
from retrieval_lab.dataset_workflow import (
    dataset_draft_status,
    finalize_dataset_bundle,
    load_dataset_draft,
    review_dataset_query,
)
from retrieval_lab.exceptions import DatasetValidationError


def _bundle(tmp_path: Path, *, complete: bool = False) -> Path:
    bundle = tmp_path / "bundle"
    relevance = {"doc.md": 1} if complete else {}
    DatasetDraft([DraftQuery("q-1", "query", relevance=relevance)]).save_bundle(bundle)
    return bundle


def _manifest(bundle: Path) -> dict[str, object]:
    return json.loads((bundle / "dataset-manifest.json").read_text(encoding="utf-8"))


def _write_manifest(bundle: Path, value: object) -> None:
    (bundle / "dataset-manifest.json").write_text(json.dumps(value), encoding="utf-8")


def _write_dataset(bundle: Path, content: str) -> None:
    (bundle / "evaluation.jsonl").write_text(content, encoding="utf-8")


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "doc.md").write_text("document", encoding="utf-8")
    return corpus


def test_manifest_query_count_must_match_jsonl(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = _manifest(bundle)
    manifest["query_count"] = 2
    _write_manifest(bundle, manifest)

    with pytest.raises(DatasetValidationError, match="query_count does not match"):
        load_dataset_draft(bundle)


def test_manifest_pending_ids_must_match_jsonl(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = _manifest(bundle)
    manifest["pending_query_ids"] = []
    _write_manifest(bundle, manifest)

    with pytest.raises(DatasetValidationError, match="pending_query_ids do not match"):
        load_dataset_draft(bundle)


def test_review_rejects_chunk_level_draft(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    DatasetDraft([DraftQuery("q-1", "query")], relevance_level="chunk").save_bundle(
        bundle
    )

    with pytest.raises(DatasetValidationError, match="document-level"):
        review_dataset_query(
            bundle,
            query_id="q-1",
            relevance={"doc.md": 1},
            corpus=_corpus(tmp_path),
        )


def test_review_rejects_unknown_query_id(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    with pytest.raises(DatasetValidationError, match="unknown draft query"):
        review_dataset_query(
            bundle,
            query_id="missing",
            relevance={"doc.md": 1},
            corpus=_corpus(tmp_path),
        )


def test_review_notes_update_without_completing_review(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    status = review_dataset_query(
        bundle,
        query_id="q-1",
        relevance={"doc.md": 1},
        corpus=_corpus(tmp_path),
        notes="checked once",
    )

    assert status.review_status == "unreviewed"
    assert load_dataset_draft(bundle).provenance.notes == "checked once"


def test_finalize_completed_bundle_returns_status(tmp_path: Path) -> None:
    status = finalize_dataset_bundle(_bundle(tmp_path, complete=True))

    assert status.complete_query_count == 1
    assert status.pending_query_ids == ()


@pytest.mark.parametrize("value", [None, 42, object()])
def test_bundle_path_rejects_non_path_values(tmp_path: Path, value: object) -> None:
    with pytest.raises(DatasetValidationError, match="string or Path"):
        dataset_draft_status(value)  # type: ignore[arg-type]


def test_bundle_path_must_be_directory(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_text("x", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="not a directory"):
        dataset_draft_status(path)


def test_empty_draft_jsonl_can_be_loaded_when_manifest_matches(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    _write_dataset(bundle, "")
    manifest = _manifest(bundle)
    manifest["query_count"] = 0
    manifest["pending_query_ids"] = []
    _write_manifest(bundle, manifest)

    loaded = load_dataset_draft(bundle)

    assert loaded.queries == ()


def test_blank_jsonl_line_is_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    _write_dataset(bundle, "\n")

    with pytest.raises(DatasetValidationError, match="blank lines"):
        load_dataset_draft(bundle)


def test_duplicate_query_ids_in_jsonl_are_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    record = '{"query_id":"q-1","query":"query","relevant":[]}\n'
    _write_dataset(bundle, record + record)
    manifest = _manifest(bundle)
    manifest["query_count"] = 2
    manifest["pending_query_ids"] = ["q-1"]
    _write_manifest(bundle, manifest)

    with pytest.raises(DatasetValidationError, match="duplicate query_id"):
        load_dataset_draft(bundle)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"query_id":"q-1","query":"query"}\n', "record fields"),
        (
            '{"query_id":"q-1","query":"query","relevant":{}}\n',
            "relevant must be an array",
        ),
        (
            '{"query_id":"q-1","query":"query","relevant":[{}]}\n',
            "contain only id and relevance",
        ),
        (
            '{"query_id":"q-1","query":"query","relevant":'
            '[{"id":"doc.md","relevance":0}]}\n',
            "integer >= 1",
        ),
        (
            '{"query_id":"q-1","query":"query","relevant":'
            '[{"id":"doc.md","relevance":1},{"id":"doc.md","relevance":2}]}\n',
            "duplicate relevant ID",
        ),
        (
            '{"query_id":"","query":"query","relevant":[]}\n',
            "query_id must be a non-empty string",
        ),
        (
            '{"query_id":"q-1","query":"query","relevant":[],"metadata":[]}\n',
            "metadata",
        ),
    ],
)
def test_invalid_draft_records_are_rejected(
    tmp_path: Path, content: str, message: str
) -> None:
    bundle = _bundle(tmp_path)
    _write_dataset(bundle, content)

    with pytest.raises(DatasetValidationError, match=message):
        load_dataset_draft(bundle)


def test_reference_answer_is_loaded_from_draft_record(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    _write_dataset(
        bundle,
        '{"query_id":"q-1","query":"query","relevant":[],"reference_answer":"a"}\n',
    )

    loaded = load_dataset_draft(bundle)

    assert loaded.queries[0].reference_answer == "a"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("relevance_level", "page", "relevance_level"),
        ("query_count", -1, "query_count"),
        ("query_count", True, "query_count"),
        ("pending_query_ids", [1], "pending_query_ids"),
        ("pending_query_ids", ["q-1", "q-1"], "duplicates"),
        ("provenance", [], "provenance must be an object"),
    ],
)
def test_invalid_manifest_values_are_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    bundle = _bundle(tmp_path)
    manifest = _manifest(bundle)
    manifest[field] = value
    _write_manifest(bundle, manifest)

    with pytest.raises(DatasetValidationError, match=message):
        load_dataset_draft(bundle)


def test_manifest_field_set_is_strict(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = _manifest(bundle)
    manifest["extra"] = True
    _write_manifest(bundle, manifest)

    with pytest.raises(DatasetValidationError, match="manifest fields"):
        load_dataset_draft(bundle)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("origin", 1, "origin and review_status"),
        ("review_status", 1, "origin and review_status"),
        ("generator", 1, "generator"),
        ("notes", 1, "notes"),
        ("human_reviewed", True, "human_reviewed conflicts"),
    ],
)
def test_invalid_provenance_manifest_values_are_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    bundle = _bundle(tmp_path)
    manifest = _manifest(bundle)
    provenance = manifest["provenance"]
    assert isinstance(provenance, dict)
    provenance[field] = value
    _write_manifest(bundle, manifest)

    with pytest.raises(DatasetValidationError, match=message):
        load_dataset_draft(bundle)


def test_provenance_field_set_is_strict(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = _manifest(bundle)
    provenance = manifest["provenance"]
    assert isinstance(provenance, dict)
    provenance["extra"] = True
    _write_manifest(bundle, manifest)

    with pytest.raises(DatasetValidationError, match="provenance fields"):
        load_dataset_draft(bundle)


@pytest.mark.parametrize(
    ("relevance", "message"),
    [
        ({}, "non-empty mapping"),
        ({"": 1}, "IDs must be non-empty"),
        ({"doc.md": 0}, "grades must be integers"),
        ({"doc.md": True}, "grades must be integers"),
    ],
)
def test_review_relevance_validation(
    tmp_path: Path, relevance: dict[str, int], message: str
) -> None:
    bundle = _bundle(tmp_path)

    with pytest.raises(DatasetValidationError, match=message):
        review_dataset_query(
            bundle,
            query_id="q-1",
            relevance=relevance,
            corpus=_corpus(tmp_path),
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{", "invalid dataset manifest JSON"),
        ("[]", "dataset manifest must be a JSON object"),
        ('{"a":1,"a":2}', "duplicate JSON key"),
    ],
)
def test_invalid_manifest_json_is_rejected(
    tmp_path: Path, content: str, message: str
) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "dataset-manifest.json").write_text(content, encoding="utf-8")

    with pytest.raises(DatasetValidationError, match=message):
        load_dataset_draft(bundle)


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "dataset-manifest.json").unlink()

    with pytest.raises(DatasetValidationError, match="not a regular file"):
        load_dataset_draft(bundle)


def test_manifest_must_be_utf8(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "dataset-manifest.json").write_bytes(b"\xff")

    with pytest.raises(DatasetValidationError, match="valid UTF-8"):
        load_dataset_draft(bundle)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{broken}\n", "invalid JSON"),
        ("[]\n", "record must be a JSON object"),
        (
            '{"query_id":"q-1","query":"query","relevant":[],"query_id":"q-2"}\n',
            "duplicate JSON key",
        ),
    ],
)
def test_invalid_dataset_json_is_rejected(
    tmp_path: Path, content: str, message: str
) -> None:
    bundle = _bundle(tmp_path)
    _write_dataset(bundle, content)

    with pytest.raises(DatasetValidationError, match=message):
        load_dataset_draft(bundle)
