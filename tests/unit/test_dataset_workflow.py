import json
from pathlib import Path

import pytest

from retrieval_lab import (
    DatasetDraft,
    DatasetProvenance,
    DraftQuery,
)
from retrieval_lab.dataset_workflow import (
    dataset_draft_status,
    finalize_dataset_bundle,
    load_dataset_draft,
    review_dataset_query,
)
from retrieval_lab.exceptions import DatasetValidationError


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "doc-1.md").write_text("first document", encoding="utf-8")
    (root / "doc-2.md").write_text("second document", encoding="utf-8")
    return root


def test_load_dataset_draft_round_trips_incomplete_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    original = DatasetDraft(
        [DraftQuery("q-1", "query")],
        provenance=DatasetProvenance(origin="synthetic", generator="generator"),
    )
    original.save_bundle(bundle)

    loaded = load_dataset_draft(bundle)

    assert loaded.queries == original.queries
    assert loaded.pending_query_ids == ("q-1",)
    assert loaded.provenance.origin == "synthetic"
    assert loaded.provenance.reliability == "experimental"


def test_dataset_draft_status_is_machine_readable(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    DatasetDraft(
        [
            DraftQuery("q-1", "first", relevance={"doc-1.md": 1}),
            DraftQuery("q-2", "second"),
        ]
    ).save_bundle(bundle)

    status = dataset_draft_status(bundle)
    payload = json.loads(status.to_json())

    assert status.query_count == 2
    assert status.complete_query_count == 1
    assert status.pending_query_ids == ("q-2",)
    assert payload["pending_query_ids"] == ["q-2"]


def test_review_dataset_query_validates_and_persists_relevance(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    corpus = _corpus(tmp_path)
    DatasetDraft([DraftQuery("q-1", "query")]).save_bundle(bundle)

    status = review_dataset_query(
        bundle,
        query_id="q-1",
        relevance={"doc-1.md": 3},
        corpus=corpus,
    )

    loaded = load_dataset_draft(bundle)
    assert status.pending_query_ids == ()
    assert loaded.queries[0].relevance == {"doc-1.md": 3}


def test_review_rejects_unknown_corpus_id(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    corpus = _corpus(tmp_path)
    DatasetDraft([DraftQuery("q-1", "query")]).save_bundle(bundle)

    with pytest.raises(DatasetValidationError, match="unknown documents"):
        review_dataset_query(
            bundle,
            query_id="q-1",
            relevance={"missing": 1},
            corpus=corpus,
        )


def test_complete_review_requires_all_queries_complete(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    corpus = _corpus(tmp_path)
    DatasetDraft(
        [DraftQuery("q-1", "first"), DraftQuery("q-2", "second")]
    ).save_bundle(bundle)

    with pytest.raises(DatasetValidationError, match="pending"):
        review_dataset_query(
            bundle,
            query_id="q-1",
            relevance={"doc-1.md": 1},
            corpus=corpus,
            complete_review=True,
        )


def test_complete_review_preserves_origin_and_updates_status(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    corpus = _corpus(tmp_path)
    DatasetDraft(
        [DraftQuery("q-1", "query")],
        provenance=DatasetProvenance(origin="synthetic", generator="generator"),
    ).save_bundle(bundle)

    status = review_dataset_query(
        bundle,
        query_id="q-1",
        relevance={"doc-2.md": 1},
        corpus=corpus,
        complete_review=True,
        notes="human checked",
    )

    loaded = load_dataset_draft(bundle)
    assert status.review_status == "reviewed"
    assert status.reliability == "reviewed"
    assert loaded.provenance.origin == "synthetic"
    assert loaded.provenance.notes == "human checked"


def test_finalize_dataset_bundle_rejects_pending_queries(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    DatasetDraft([DraftQuery("q-1", "query")]).save_bundle(bundle)

    with pytest.raises(DatasetValidationError, match="pending relevance"):
        finalize_dataset_bundle(bundle)


def test_manifest_derived_fields_cannot_be_forged(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    DatasetDraft(
        [DraftQuery("q-1", "query")],
        provenance=DatasetProvenance(origin="synthetic", generator="generator"),
    ).save_bundle(bundle)
    manifest_path = bundle / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["reliability"] = "reviewed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="reliability conflicts"):
        load_dataset_draft(bundle)
