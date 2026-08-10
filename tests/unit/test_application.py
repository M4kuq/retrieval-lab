from __future__ import annotations

from pathlib import Path

import pytest

from retrieval_lab.application import (
    ExperimentOutput,
    InitializedProject,
    initialize_project,
    run_configured_experiment,
    validate_config_inputs,
)
from retrieval_lab.exceptions import ConfigurationError, DatasetValidationError


def test_initialize_project_creates_a_runnable_unicode_template(
    tmp_path: Path,
) -> None:
    project = initialize_project(tmp_path / "実験")

    assert isinstance(project, InitializedProject)
    assert {path.relative_to(project.target).as_posix() for path in project.files} == {
        "retrieval-lab.yaml",
        "corpus/example.md",
        "evaluation.jsonl",
    }
    assert "schema_version: 1" in (project.target / "retrieval-lab.yaml").read_text(
        encoding="utf-8"
    )
    assert "サンプル" in (project.target / "corpus/example.md").read_text(
        encoding="utf-8"
    )
    assert (
        (project.target / "evaluation.jsonl").read_text(encoding="utf-8").endswith("\n")
    )


def test_initialize_preserves_existing_and_unrelated_files(tmp_path: Path) -> None:
    project = initialize_project(tmp_path / "project")
    unrelated = project.target / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")
    config = project.target / "retrieval-lab.yaml"
    config.write_text("custom", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        initialize_project(project.target)
    initialize_project(project.target, force=True)

    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert "schema_version: 1" in config.read_text(encoding="utf-8")


def test_initialize_rejects_file_target(tmp_path: Path) -> None:
    target = tmp_path / "file"
    target.write_text("not a project", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        initialize_project(target)


def test_initialize_rejects_symlinked_targets_and_template_parents(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked_target = tmp_path / "linked-target"
    linked_target.symlink_to(real, target_is_directory=True)
    with pytest.raises(ConfigurationError):
        initialize_project(linked_target)

    project = tmp_path / "project"
    project.mkdir()
    linked_corpus = project / "corpus"
    linked_corpus.symlink_to(real, target_is_directory=True)
    with pytest.raises(ConfigurationError):
        initialize_project(project, force=True)


def test_initialize_allows_a_user_selected_symlinked_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    project = initialize_project(linked_parent / "project")

    assert (project.target / "retrieval-lab.yaml").is_file()


def test_run_rejects_file_or_symlink_output_directory(tmp_path: Path) -> None:
    project = initialize_project(tmp_path / "project")
    config = project.target / "retrieval-lab.yaml"
    output_file = project.target / "output-file"
    output_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        run_configured_experiment(config, output_dir=output_file, formats=("json",))

    linked_output = project.target / "linked-output"
    linked_output.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ConfigurationError):
        run_configured_experiment(config, output_dir=linked_output, formats=("json",))


def test_validate_inputs_does_not_run_search(tmp_path: Path) -> None:
    project = initialize_project(tmp_path / "project")
    validated = validate_config_inputs(project.target / "retrieval-lab.yaml")

    assert validated.document_count == 1
    assert validated.query_count == 1
    assert validated.retriever_names == ("keyword", "bm25")
    assert validated.config.schema_version == 1


def test_validate_inputs_checks_chunk_level_gold_ids(tmp_path: Path) -> None:
    project = initialize_project(tmp_path / "project")
    config_path = project.target / "retrieval-lab.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "relevance_level: document", "relevance_level: chunk"
        ),
        encoding="utf-8",
    )
    (project.target / "evaluation.jsonl").write_text(
        '{"query_id":"q","query":"検索","relevant":[{"id":"missing","relevance":1}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError):
        validate_config_inputs(config_path)


def test_run_configured_experiment_writes_selected_formats(tmp_path: Path) -> None:
    project = initialize_project(tmp_path / "project")
    output = run_configured_experiment(
        project.target / "retrieval-lab.yaml",
        output_dir=project.target / "out",
        formats=("html", "json", "json"),
    )

    assert isinstance(output, ExperimentOutput)
    assert output.formats == ("json", "html")
    assert [path.name for path in output.paths] == ["result.json", "report.html"]
    assert (project.target / "out/result.json").exists()
    assert (project.target / "out/report.html").exists()
    assert not (project.target / "out/summary.csv").exists()


def test_run_uses_configured_formats_and_replaces_owned_outputs(tmp_path: Path) -> None:
    project = initialize_project(tmp_path / "project")
    config_path = project.target / "retrieval-lab.yaml"

    first = run_configured_experiment(config_path)
    (project.target / "reports/result.json").write_text("stale", encoding="utf-8")
    second = run_configured_experiment(config_path)

    assert first.formats == ("json", "csv", "html")
    assert [path.name for path in second.paths] == [
        "result.json",
        "summary.csv",
        "per_query.csv",
        "report.html",
    ]
    assert (
        (project.target / "reports/result.json")
        .read_text(encoding="utf-8")
        .startswith("{")
    )


def test_run_rejects_empty_or_unknown_format(tmp_path: Path) -> None:
    project = initialize_project(tmp_path / "project")
    config_path = project.target / "retrieval-lab.yaml"
    with pytest.raises(ConfigurationError):
        run_configured_experiment(config_path, formats=())
    with pytest.raises(ConfigurationError):
        run_configured_experiment(config_path, formats=("pdf",))


def test_application_rejects_invalid_boundary_inputs(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        initialize_project("")
    with pytest.raises(ConfigurationError):
        initialize_project(tmp_path / "bad-force", force=1)  # type: ignore[arg-type]

    project = initialize_project(tmp_path / "project")
    config_path = project.target / "retrieval-lab.yaml"
    with pytest.raises(ConfigurationError):
        run_configured_experiment(config_path, formats="json")  # type: ignore[arg-type]

    no_reports = config_path.read_text(encoding="utf-8").replace(
        "formats: [json, csv, html]", "formats: []"
    )
    config_path.write_text(no_reports, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        run_configured_experiment(config_path)


def test_force_does_not_write_over_a_known_directory(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "retrieval-lab.yaml").mkdir()
    with pytest.raises(ConfigurationError):
        initialize_project(project, force=True)
