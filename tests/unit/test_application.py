from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieval_lab.application import (
    ExperimentOutput,
    InitializedProject,
    compare_result_files,
    evaluate_configured_quality_gates,
    initialize_project,
    inspect_result,
    run_configured_experiment,
    validate_config_inputs,
)
from retrieval_lab.evaluation.engine import content_hash
from retrieval_lab.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    IncomparableRunError,
)

_CONFIGURED_RUN_IDENTITY_FIELDS = {
    "chunk_hash",
    "dataset_hash",
    "index_hashes",
    "metric_version",
    "quality_gate_policy_hash",
    "relevance_level",
    "retrievers",
    "retriever_settings",
    "seed",
    "top_k",
}


def _recompute_configured_run_id(payload: dict[str, object]) -> None:
    run = payload["run"]
    assert isinstance(run, dict)
    manifest = run["manifest"]
    assert isinstance(manifest, dict)
    run["id"] = content_hash(
        {key: manifest[key] for key in sorted(_CONFIGURED_RUN_IDENTITY_FIELDS)}  # type: ignore[arg-type]
    )


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


def test_inspect_compare_and_gate_application_services(tmp_path: Path) -> None:
    project = initialize_project(tmp_path / "project")
    output = run_configured_experiment(project.target / "retrieval-lab.yaml")
    result_path = output.paths[0]

    inspection = inspect_result(result_path, query_id="q-example")
    assert inspection.result.run_id == output.result.run_id
    assert [row.retriever for row in inspection.evidence] == ["bm25", "keyword"]
    inspection_payload = json.loads(inspection.to_json())
    assert inspection_payload["query"]["query_id"] == "q-example"
    assert inspection_payload["query"]["evidence"][0]["retrieved_ids_by_cutoff"] == {
        "1": ["example.md"],
        "3": ["example.md"],
    }

    comparison = compare_result_files(result_path, result_path)
    assert comparison.comparison.baseline_run_id == output.result.run_id
    assert comparison.rows
    assert any(row.direction == "lower_is_better" for row in comparison.rows)
    assert json.loads(comparison.to_json())["common_retrievers"] == [
        "bm25",
        "keyword",
    ]

    passing = evaluate_configured_quality_gates(
        project.target / "retrieval-lab.yaml", result_path
    )
    assert passing.report.passed

    config_path = project.target / "retrieval-lab.yaml"
    failing_gate = (
        "quality_gates:\n  - retriever: bm25\n    metric: recall@1\n    min_value: 2.0"
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "quality_gates: []",
            failing_gate,
        ),
        encoding="utf-8",
    )
    failing = evaluate_configured_quality_gates(config_path, result_path)
    assert not failing.report.passed
    assert json.loads(failing.to_json())["passed"] is False


def test_gate_application_restores_embedded_candidate_gates(tmp_path: Path) -> None:
    project = initialize_project(tmp_path / "project")
    config_path = project.target / "retrieval-lab.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "quality_gates: []",
            "quality_gates:\n  - retriever: bm25\n    metric: recall@1\n"
            "    min_value: 0.0",
        ),
        encoding="utf-8",
    )
    output = run_configured_experiment(config_path)

    restored = evaluate_configured_quality_gates(output.paths[0])
    restored_by_keyword = evaluate_configured_quality_gates(
        candidate_path=output.paths[0]
    )

    assert restored.report.passed
    assert restored_by_keyword.report.passed
    payload = json.loads(output.paths[0].read_text(encoding="utf-8"))
    payload["run"]["manifest"]["config"]["quality_gates"] = []
    output.paths[0].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="embedded quality gates"):
        evaluate_configured_quality_gates(output.paths[0])


def test_embedded_gate_rejects_invalid_manifest_and_policy_tampering(
    tmp_path: Path,
) -> None:
    project = initialize_project(tmp_path / "project")
    config_path = project.target / "retrieval-lab.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "quality_gates: []",
            "quality_gates:\n  - retriever: bm25\n    metric: recall@1\n"
            "    min_value: 0.0",
        ),
        encoding="utf-8",
    )
    output = run_configured_experiment(config_path)
    original = json.loads(output.paths[0].read_text(encoding="utf-8"))

    invalid_manifest = json.loads(json.dumps(original))
    invalid_manifest["run"]["manifest"]["top_k"] = []
    output.paths[0].write_text(json.dumps(invalid_manifest), encoding="utf-8")
    with pytest.raises(IncomparableRunError, match="producer manifest"):
        evaluate_configured_quality_gates(output.paths[0])

    changed_policy = json.loads(json.dumps(original))
    changed_policy["run"]["manifest"]["config"]["quality_gates"][0]["min_value"] = 1.0
    output.paths[0].write_text(json.dumps(changed_policy), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="policy"):
        evaluate_configured_quality_gates(output.paths[0])


def test_embedded_gate_policy_participates_in_configured_run_id(tmp_path: Path) -> None:
    project = initialize_project(tmp_path / "project")
    config_path = project.target / "retrieval-lab.yaml"
    base = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        base.replace(
            "quality_gates: []",
            "quality_gates:\n  - retriever: bm25\n    metric: recall@1\n"
            "    min_value: 0.0",
        ),
        encoding="utf-8",
    )
    first = run_configured_experiment(config_path).result
    config_path.write_text(
        base.replace(
            "quality_gates: []",
            "quality_gates:\n  - retriever: bm25\n    metric: recall@1\n"
            "    min_value: 1.0",
        ),
        encoding="utf-8",
    )
    second = run_configured_experiment(config_path).result

    assert first.run_id != second.run_id
    assert (
        first.manifest["quality_gate_policy_hash"]
        != second.manifest["quality_gate_policy_hash"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", -1),
        ("chunk_hash", 7),
        ("retrievers", []),
        ("retriever_settings", []),
        ("index_hashes", []),
    ],
)
def test_embedded_gate_rejects_forged_producer_identity_shapes(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    project = initialize_project(tmp_path / "project")
    config_path = project.target / "retrieval-lab.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "quality_gates: []",
            "quality_gates:\n  - retriever: bm25\n    metric: recall@1\n"
            "    min_value: 0.0",
        ),
        encoding="utf-8",
    )
    output = run_configured_experiment(config_path)
    payload = json.loads(output.paths[0].read_text(encoding="utf-8"))
    payload["run"]["manifest"][field] = value
    _recompute_configured_run_id(payload)
    output.paths[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="candidate result manifest"):
        evaluate_configured_quality_gates(output.paths[0])


def test_artifact_only_gate_uses_matching_baseline_policy_as_authority(
    tmp_path: Path,
) -> None:
    project = initialize_project(tmp_path / "project")
    config_path = project.target / "retrieval-lab.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "quality_gates: []",
            "quality_gates:\n  - retriever: bm25\n    metric: recall@1\n"
            "    min_value: 1.1",
        ),
        encoding="utf-8",
    )
    output = run_configured_experiment(config_path)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(output.result.to_json(), encoding="utf-8")
    candidate_path = tmp_path / "candidate.json"
    payload = json.loads(output.result.to_json())
    gates = payload["run"]["manifest"]["config"]["quality_gates"]
    gates[0]["min_value"] = 0.0
    payload["run"]["manifest"]["quality_gate_policy_hash"] = content_hash(gates)
    _recompute_configured_run_id(payload)
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")

    assert evaluate_configured_quality_gates(candidate_path).report.passed
    with pytest.raises(ConfigurationError, match="policies differ"):
        evaluate_configured_quality_gates(
            candidate_path,
            baseline_path=baseline_path,
        )


def test_compare_output_exposes_experimental_variable_differences(
    tmp_path: Path,
) -> None:
    first = initialize_project(tmp_path / "first")
    second = initialize_project(tmp_path / "second")
    first_output = run_configured_experiment(first.target / "retrieval-lab.yaml")
    second_output = run_configured_experiment(second.target / "retrieval-lab.yaml")

    comparison = compare_result_files(first_output.paths[0], second_output.paths[0])
    payload = comparison.to_dict()

    assert payload["variable_differences"]
    assert any(item["field"] == "runtime" for item in payload["variable_differences"])
    assert all(
        set(item) == {"field", "reason"} for item in payload["variable_differences"]
    )


def test_inspection_rejects_unknown_query_and_allows_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    project = initialize_project(tmp_path / "project")
    output = run_configured_experiment(project.target / "retrieval-lab.yaml")
    result_path = output.paths[0]
    linked_parent = tmp_path / "linked-reports"
    linked_parent.symlink_to(result_path.parent, target_is_directory=True)

    inspected = inspect_result(linked_parent / result_path.name)
    assert inspected.result.run_id == output.result.run_id
    with pytest.raises(ConfigurationError):
        inspect_result(result_path, query_id="missing-query")
