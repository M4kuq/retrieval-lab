from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieval_lab.application import (
    ExperimentOutput,
    InitializedProject,
    InspectionOutput,
    QueryEvidence,
    compare_result_files,
    evaluate_configured_quality_gates,
    initialize_project,
    inspect_result,
    run_configured_experiment,
    validate_config_inputs,
)
from retrieval_lab.domain.results import EvaluationResult
from retrieval_lab.evaluation.engine import content_hash
from retrieval_lab.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    EvaluationError,
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


def _gated_output(tmp_path: Path) -> ExperimentOutput:
    project = initialize_project(tmp_path / "project")
    config_path = project.target / "retrieval-lab.yaml"
    template = config_path.read_text(encoding="utf-8")
    assert "quality_gates: []" in template
    config_path.write_text(
        template.replace(
            "quality_gates: []",
            "quality_gates:\n  - retriever: bm25\n    metric: recall@1\n"
            "    min_value: 0.0",
        ),
        encoding="utf-8",
    )
    return run_configured_experiment(config_path)


def _load_payload(output: ExperimentOutput) -> dict[str, object]:
    return json.loads(output.paths[0].read_text(encoding="utf-8"))


def _rewrite_result(output: ExperimentOutput, payload: dict[str, object]) -> Path:
    output.paths[0].write_text(json.dumps(payload), encoding="utf-8")
    return output.paths[0]


def _manifest(payload: dict[str, object]) -> dict[str, object]:
    run = payload["run"]
    assert isinstance(run, dict)
    manifest = run["manifest"]
    assert isinstance(manifest, dict)
    return manifest


def _first_gate(payload: dict[str, object]) -> dict[str, object]:
    config = _manifest(payload)["config"]
    assert isinstance(config, dict)
    gates = config["quality_gates"]
    assert isinstance(gates, list)
    gate = gates[0]
    assert isinstance(gate, dict)
    return gate


def test_initialize_rejects_missing_or_invalid_target_values() -> None:
    with pytest.raises(ConfigurationError, match="must be provided"):
        initialize_project(None)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="must be a valid path"):
        initialize_project(123)  # type: ignore[arg-type]


def test_initialize_wraps_inaccessible_project_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_exists(self: Path) -> bool:
        raise OSError("denied")

    monkeypatch.setattr(Path, "exists", broken_exists)

    with pytest.raises(ConfigurationError, match="project target is not accessible"):
        initialize_project(tmp_path / "project")


def test_initialize_wraps_project_directory_creation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_mkdir(
        self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
    ) -> None:
        raise OSError("denied")

    monkeypatch.setattr(Path, "mkdir", broken_mkdir)

    with pytest.raises(ConfigurationError, match="could not create project directory"):
        initialize_project(tmp_path / "project")


def test_initialize_wraps_inaccessible_template_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_exists = Path.exists

    def flaky_exists(self: Path) -> bool:
        if self.name == "evaluation.jsonl":
            raise OSError("denied")
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", flaky_exists)

    with pytest.raises(ConfigurationError, match="template path is not accessible"):
        initialize_project(tmp_path / "project")


def test_initialize_wraps_template_write_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_write_text(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        raise OSError("denied")

    monkeypatch.setattr(Path, "write_text", broken_write_text)

    with pytest.raises(ConfigurationError, match="could not write project template"):
        initialize_project(tmp_path / "project")


def test_path_checks_wrap_symlink_probe_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_is_symlink(self: Path) -> bool:
        raise OSError("denied")

    monkeypatch.setattr(Path, "is_symlink", broken_is_symlink)

    with pytest.raises(ConfigurationError, match=r"^path is not accessible$"):
        initialize_project(tmp_path / "project")


def test_run_requires_an_output_directory_for_selected_formats(
    tmp_path: Path,
) -> None:
    project = initialize_project(tmp_path / "project")
    config_path = project.target / "retrieval-lab.yaml"
    template = config_path.read_text(encoding="utf-8")
    assert "  output_dir: reports\n" in template
    config_path.write_text(
        template.replace("  output_dir: reports\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="an output directory is required"):
        run_configured_experiment(config_path, formats=("json",))


def test_run_csv_only_skips_the_json_report(tmp_path: Path) -> None:
    project = initialize_project(tmp_path / "project")
    output_dir = project.target / "out"

    output = run_configured_experiment(
        project.target / "retrieval-lab.yaml",
        output_dir=output_dir,
        formats=("csv",),
    )

    assert [path.name for path in output.paths] == ["summary.csv", "per_query.csv"]
    assert not (output_dir / "result.json").exists()
    assert not (output_dir / "report.html").exists()


def test_run_wraps_report_save_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = initialize_project(tmp_path / "project")

    def broken_save_json(self: EvaluationResult, path: Path) -> Path:
        raise EvaluationError("disk full")

    monkeypatch.setattr(EvaluationResult, "save_json", broken_save_json)

    with pytest.raises(
        EvaluationError, match="could not save experiment reports"
    ) as exc_info:
        run_configured_experiment(
            project.target / "retrieval-lab.yaml",
            output_dir=project.target / "out",
            formats=("json",),
        )
    assert isinstance(exc_info.value.__cause__, EvaluationError)


def test_run_rejects_non_sequence_formats(tmp_path: Path) -> None:
    project = initialize_project(tmp_path / "project")

    with pytest.raises(ConfigurationError, match="formats must be a sequence"):
        run_configured_experiment(
            project.target / "retrieval-lab.yaml",
            formats=123,  # type: ignore[arg-type]
        )


def test_run_wraps_inaccessible_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = initialize_project(tmp_path / "project")
    original_exists = Path.exists

    def flaky_exists(self: Path) -> bool:
        if self.name == "out":
            raise OSError("denied")
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", flaky_exists)

    with pytest.raises(ConfigurationError, match="output directory is not accessible"):
        run_configured_experiment(
            project.target / "retrieval-lab.yaml",
            output_dir=project.target / "out",
            formats=("json",),
        )


def test_inspect_rejects_blank_or_non_string_query_ids(tmp_path: Path) -> None:
    project = initialize_project(tmp_path / "project")
    output = run_configured_experiment(project.target / "retrieval-lab.yaml")

    with pytest.raises(ConfigurationError, match="query_id must be a non-empty string"):
        inspect_result(output.paths[0], query_id=" ")
    with pytest.raises(ConfigurationError, match="query_id must be a non-empty string"):
        inspect_result(output.paths[0], query_id=123)  # type: ignore[arg-type]


def test_inspect_wraps_inaccessible_result_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = initialize_project(tmp_path / "project")
    output = run_configured_experiment(project.target / "retrieval-lab.yaml")

    def broken_exists(self: Path) -> bool:
        raise OSError("denied")

    monkeypatch.setattr(Path, "exists", broken_exists)

    with pytest.raises(ConfigurationError, match="result path is not accessible"):
        inspect_result(output.paths[0])


def test_gate_evaluation_requires_a_candidate_path() -> None:
    with pytest.raises(
        ConfigurationError, match="candidate result path must be provided"
    ):
        evaluate_configured_quality_gates()


def test_embedded_gates_require_an_embedded_config(tmp_path: Path) -> None:
    output = _gated_output(tmp_path)
    payload = _load_payload(output)
    _manifest(payload).pop("config")

    with pytest.raises(ConfigurationError, match="does not contain embedded"):
        evaluate_configured_quality_gates(_rewrite_result(output, payload))


def test_embedded_gates_reject_malformed_entries(tmp_path: Path) -> None:
    output = _gated_output(tmp_path)
    payload = _load_payload(output)
    config = _manifest(payload)["config"]
    assert isinstance(config, dict)
    config["quality_gates"] = [{"retriever": "bm25"}]

    with pytest.raises(
        ConfigurationError, match=r"quality_gates\[0\] is invalid"
    ) as exc_info:
        evaluate_configured_quality_gates(_rewrite_result(output, payload))
    assert exc_info.value.__cause__ is None


def test_embedded_gates_reject_non_string_fields(tmp_path: Path) -> None:
    output = _gated_output(tmp_path)
    payload = _load_payload(output)
    _first_gate(payload)["retriever"] = 123

    with pytest.raises(
        ConfigurationError, match=r"quality_gates\[0\] is invalid"
    ) as exc_info:
        evaluate_configured_quality_gates(_rewrite_result(output, payload))
    assert "candidate embedded quality_gates[0].retriever must be a string" in str(
        exc_info.value.__cause__
    )


def test_embedded_gates_reject_non_numeric_thresholds(tmp_path: Path) -> None:
    output = _gated_output(tmp_path)
    payload = _load_payload(output)
    _first_gate(payload)["min_value"] = True

    with pytest.raises(
        ConfigurationError, match=r"quality_gates\[0\] is invalid"
    ) as exc_info:
        evaluate_configured_quality_gates(_rewrite_result(output, payload))
    assert "must be numeric" in str(exc_info.value.__cause__)


def test_embedded_gates_reject_thresholds_that_overflow_floats(
    tmp_path: Path,
) -> None:
    output = _gated_output(tmp_path)
    payload = _load_payload(output)
    _first_gate(payload)["min_value"] = 10**400

    with pytest.raises(
        ConfigurationError, match=r"quality_gates\[0\] is invalid"
    ) as exc_info:
        evaluate_configured_quality_gates(_rewrite_result(output, payload))
    assert "must be finite" in str(exc_info.value.__cause__)


def test_embedded_candidate_requires_all_identity_fields(tmp_path: Path) -> None:
    output = _gated_output(tmp_path)
    payload = _load_payload(output)
    _manifest(payload).pop("seed")

    with pytest.raises(ConfigurationError, match="missing configured run identity"):
        evaluate_configured_quality_gates(_rewrite_result(output, payload))


def test_embedded_candidate_rejects_a_non_hash_run_id(tmp_path: Path) -> None:
    output = _gated_output(tmp_path)
    payload = _load_payload(output)
    run = payload["run"]
    assert isinstance(run, dict)
    run["id"] = "not-a-hash"

    with pytest.raises(ConfigurationError, match="run ID must be a SHA-256 hash"):
        evaluate_configured_quality_gates(_rewrite_result(output, payload))


def test_embedded_candidate_rejects_a_mismatched_run_id(tmp_path: Path) -> None:
    output = _gated_output(tmp_path)
    payload = _load_payload(output)
    run = payload["run"]
    assert isinstance(run, dict)
    run["id"] = "e" * 64

    with pytest.raises(ConfigurationError, match="run identity does not match"):
        evaluate_configured_quality_gates(_rewrite_result(output, payload))


def test_query_evidence_omits_cutoffs_when_not_recorded() -> None:
    evidence = QueryEvidence(
        retriever="keyword",
        query_id="q",
        retrieved_ids=("doc",),
        metrics=(),
        search_latency_ms=None,
        warnings=(),
    )

    assert "retrieved_ids_by_cutoff" not in evidence.to_dict()


def test_inspection_output_omits_the_query_without_a_query_id(
    tmp_path: Path,
) -> None:
    project = initialize_project(tmp_path / "project")
    output = run_configured_experiment(project.target / "retrieval-lab.yaml")
    inspection = InspectionOutput(result=output.result, query_id=None, evidence=())

    assert "query" not in inspection.to_dict()


def test_inspection_output_rejects_non_finite_evidence_metrics(
    tmp_path: Path,
) -> None:
    project = initialize_project(tmp_path / "project")
    output = run_configured_experiment(project.target / "retrieval-lab.yaml")
    inspection = InspectionOutput(
        result=output.result,
        query_id="q",
        evidence=(
            QueryEvidence(
                retriever="bm25",
                query_id="q",
                retrieved_ids=(),
                metrics=(("recall@1", float("nan")),),
                search_latency_ms=None,
                warnings=(),
            ),
        ),
    )

    with pytest.raises(EvaluationError, match="could not be serialized"):
        inspection.to_json()
