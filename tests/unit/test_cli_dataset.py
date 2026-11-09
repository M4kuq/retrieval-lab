import argparse
import json
import re
from pathlib import Path

import pytest

from retrieval_lab import (
    DatasetDraft,
    DatasetValidationError,
    DraftQuery,
    load_dataset_draft,
)
from retrieval_lab.cli import dataset as dataset_cli
from retrieval_lab.cli.app import main
from retrieval_lab.cli.dataset import run_dataset_command


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    DatasetDraft([DraftQuery("q-1", "which document is relevant?")]).save_bundle(bundle)
    return bundle


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc-1.md").write_text("relevant content", encoding="utf-8")
    return corpus


def _review_args(
    bundle: Path, corpus: Path, relevant: list[str] | None
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_command="review",
        bundle=bundle,
        corpus=corpus,
        query_id="q-1",
        relevant=relevant,
        complete_review=False,
        notes=None,
        json_output=False,
    )


def test_dataset_status_cli_emits_json(tmp_path: Path, capsys) -> None:
    bundle = _bundle(tmp_path)

    status = main(["dataset", "status", str(bundle), "--json"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query_count"] == 1
    assert payload["pending_query_ids"] == ["q-1"]


def test_dataset_review_cli_persists_noninteractive_relevance(
    tmp_path: Path,
    capsys,
) -> None:
    bundle = _bundle(tmp_path)
    corpus = _corpus(tmp_path)

    status = main(
        [
            "dataset",
            "review",
            str(bundle),
            "--corpus",
            str(corpus),
            "--query-id",
            "q-1",
            "--relevant",
            "doc-1.md:2",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pending_query_ids"] == []
    assert load_dataset_draft(bundle).queries[0].relevance == {"doc-1.md": 2}


def test_dataset_review_cli_supports_interactive_selection(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    bundle = _bundle(tmp_path)
    corpus = _corpus(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "doc-1.md")

    status = main(
        [
            "dataset",
            "review",
            str(bundle),
            "--corpus",
            str(corpus),
            "--query-id",
            "q-1",
        ]
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "query[q-1]" in output
    assert "doc-1.md" in output
    assert load_dataset_draft(bundle).queries[0].relevance == {"doc-1.md": 1}


def test_dataset_finalize_cli_returns_input_error_for_pending_draft(
    tmp_path: Path,
    capsys,
) -> None:
    bundle = _bundle(tmp_path)

    status = main(["dataset", "finalize", str(bundle)])

    assert status == 2
    assert "configuration or input error" in capsys.readouterr().err


def test_dataset_status_cli_lists_pending_queries_in_text_output(
    tmp_path: Path, capsys
) -> None:
    bundle = _bundle(tmp_path)

    status = main(["dataset", "status", str(bundle)])

    assert status == 0
    output = capsys.readouterr().out
    assert "queries: 0/1 complete" in output
    assert "pending: q-1" in output


def test_dataset_finalize_cli_succeeds_for_a_completed_draft(
    tmp_path: Path,
    capsys,
) -> None:
    bundle = _bundle(tmp_path)
    corpus = _corpus(tmp_path)
    reviewed = main(
        [
            "dataset",
            "review",
            str(bundle),
            "--corpus",
            str(corpus),
            "--query-id",
            "q-1",
            "--relevant",
            "doc-1.md",
            "--complete-review",
        ]
    )
    assert reviewed == 0
    capsys.readouterr()

    status = main(["dataset", "finalize", str(bundle), "--json"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pending_query_ids"] == []
    assert payload["complete_query_count"] == 1
    assert payload["review_status"] == "reviewed"


def test_dataset_command_rejects_unknown_commands() -> None:
    args = argparse.Namespace(dataset_command="bogus")

    with pytest.raises(DatasetValidationError, match="unknown dataset command"):
        run_dataset_command(args)


def test_dataset_review_prompt_rejects_unknown_query_ids(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    corpus = _corpus(tmp_path)
    args = _review_args(bundle, corpus, None)
    args.query_id = "missing"

    with pytest.raises(DatasetValidationError, match="unknown draft query ID"):
        run_dataset_command(args)


def test_dataset_review_prompt_requires_at_least_one_document(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = _bundle(tmp_path)
    corpus = _corpus(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "  ")

    with pytest.raises(
        DatasetValidationError, match="at least one relevant document is required"
    ):
        run_dataset_command(_review_args(bundle, corpus, None))


@pytest.mark.parametrize(
    ("relevant", "message"),
    [
        ([" "], "--relevant values must be non-empty"),
        ([":2"], "--relevant must use ID or ID:GRADE syntax"),
        (["doc-1.md:"], "--relevant must use ID or ID:GRADE syntax"),
        (["doc-1.md:abc"], "--relevant grade must be an integer"),
        (["doc-1.md:0"], "--relevant grade must be >= 1"),
        (["doc-1.md", "doc-1.md:2"], "duplicate --relevant document ID"),
        ([], "at least one relevant document is required"),
    ],
)
def test_dataset_review_rejects_invalid_relevance_specs(
    tmp_path: Path, relevant: list[str], message: str
) -> None:
    bundle = _bundle(tmp_path)
    corpus = _corpus(tmp_path)

    with pytest.raises(DatasetValidationError, match=re.escape(message)):
        run_dataset_command(_review_args(bundle, corpus, relevant))


def test_dataset_relevance_specs_allow_colons_in_document_ids() -> None:
    assert dataset_cli._parse_relevance_specs(["a:b:2"]) == {"a:b": 2}
    assert dataset_cli._parse_relevance_specs(["a:b"]) == {"a:b": 1}


def test_dataset_review_cli_returns_input_error_for_invalid_grade(
    tmp_path: Path,
    capsys,
) -> None:
    bundle = _bundle(tmp_path)
    corpus = _corpus(tmp_path)

    status = main(
        [
            "dataset",
            "review",
            str(bundle),
            "--corpus",
            str(corpus),
            "--query-id",
            "q-1",
            "--relevant",
            "doc-1.md:0",
        ]
    )

    assert status == 2
    assert "configuration or input error" in capsys.readouterr().err
