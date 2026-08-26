import json
from pathlib import Path

from retrieval_lab import DatasetDraft, DraftQuery, load_dataset_draft
from retrieval_lab.cli.app import main


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    DatasetDraft([DraftQuery("q-1", "which document is relevant?")]).save_bundle(bundle)
    return bundle


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc-1.md").write_text("relevant content", encoding="utf-8")
    return corpus


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
