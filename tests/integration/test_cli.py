from __future__ import annotations

import json
from pathlib import Path

import pytest

import retrieval_lab.cli.app as cli_app
from retrieval_lab import load_result
from retrieval_lab.cli.app import main
from retrieval_lab.exceptions import (
    EvaluationError,
    OptionalDependencyError,
    RetrieverContractError,
)


def test_help_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0
    assert "offline Retrieval Lab" in capsys.readouterr().out


def test_init_validate_and_run_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "日本語プロジェクト"
    assert main(["init", str(project)]) == 0
    assert "Initialized project" in capsys.readouterr().out

    config = project / "retrieval-lab.yaml"
    assert main(["validate", "--config", str(config)]) == 0
    assert "Configuration valid" in capsys.readouterr().out

    output_dir = project / "artifacts"
    assert (
        main(
            [
                "run",
                "-c",
                str(config),
                "-o",
                str(output_dir),
                "-f",
                "json",
                "-f",
                "csv",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "Evaluation complete" in captured.out
    assert captured.err == ""
    assert (output_dir / "result.json").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "per_query.csv").exists()
    assert not (output_dir / "report.html").exists()


def test_cli_input_errors_are_short_and_do_not_expose_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_path = tmp_path / "secret-token-config.yaml"
    status = main(["validate", "-c", str(secret_path)])
    captured = capsys.readouterr()

    assert status == 2
    assert captured.out == ""
    assert "configuration or input error" in captured.err
    assert str(tmp_path) not in captured.err
    assert "Traceback" not in captured.err


def test_cli_rejects_unknown_command_and_format() -> None:
    with pytest.raises(SystemExit) as unknown:
        main(["unknown"])
    assert unknown.value.code == 2
    with pytest.raises(SystemExit) as bad_format:
        main(["run", "-c", "config.yaml", "-f", "pdf"])
    assert bad_format.value.code == 2


def test_inspect_compare_and_gate_exit_codes_and_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    config = project / "retrieval-lab.yaml"
    assert main(["run", "-c", str(config), "-f", "json"]) == 0
    capsys.readouterr()
    result = project / "reports/result.json"

    assert main(["inspect", str(result), "--query-id", "q-example", "--json"]) == 0
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["command"] == "inspect"
    assert inspect_payload["query"]["query_id"] == "q-example"
    assert inspect_payload["query"]["evidence"][0]["retrieved_ids_by_cutoff"]

    assert main(["inspect", str(result), "--query-id", "q-example"]) == 0
    inspect_text = capsys.readouterr().out
    assert "query_evidence: q-example" in inspect_text
    assert "retrieved_ids:" in inspect_text
    assert "retrieved_ids@1:" in inspect_text

    assert main(["compare", str(result), str(result), "--json"]) == 0
    compare_payload = json.loads(capsys.readouterr().out)
    assert compare_payload["command"] == "compare"
    assert compare_payload["metrics"]

    assert main(["compare", str(result), str(result)]) == 0
    compare_text = capsys.readouterr().out
    assert "baseline_run_id:" in compare_text
    assert "direction=lower_is_better" in compare_text

    assert main(["gate", "-c", str(config), str(result), "--json"]) == 0
    gate_payload = json.loads(capsys.readouterr().out)
    assert gate_payload["command"] == "gate"
    assert gate_payload["passed"] is True

    assert main(["inspect", str(result), "--query-id", "missing"]) == 2
    error = capsys.readouterr()
    assert error.out == ""
    assert "Traceback" not in error.err


def test_compare_human_output_reports_experimental_differences(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert main(["init", str(first)]) == 0
    capsys.readouterr()
    assert main(["init", str(second)]) == 0
    capsys.readouterr()
    assert main(["run", "-c", str(first / "retrieval-lab.yaml"), "-f", "json"]) == 0
    capsys.readouterr()
    assert main(["run", "-c", str(second / "retrieval-lab.yaml"), "-f", "json"]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "compare",
                str(first / "reports/result.json"),
                str(second / "reports/result.json"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "experimental_difference: runtime:" in output


def test_compare_json_omits_experimental_manifest_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    config = project / "retrieval-lab.yaml"
    assert main(["run", "-c", str(config), "-f", "json"]) == 0
    capsys.readouterr()

    source = project / "reports/result.json"
    baseline_payload = json.loads(source.read_text(encoding="utf-8"))
    candidate_payload = json.loads(source.read_text(encoding="utf-8"))
    baseline_payload["run"]["manifest"]["config"] = {
        "path": "/private/baseline",
        "token": "token-value-baseline",
    }
    candidate_payload["run"]["manifest"]["config"] = {
        "path": "/private/candidate",
        "token": "token-value-candidate",
    }
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(baseline_payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(candidate_payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    assert main(["compare", str(baseline), str(candidate), "--json"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    rows = payload["variable_differences"]
    assert rows
    assert all(set(row) == {"field", "reason"} for row in rows)
    assert any(row["field"] == "config" for row in rows)
    for sensitive in (
        "/private/baseline",
        "/private/candidate",
        "token-value-baseline",
        "token-value-candidate",
    ):
        assert sensitive not in output


def test_gate_failure_is_exit_one_and_debug_is_opt_in(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    config = project / "retrieval-lab.yaml"
    failing_gate = (
        "quality_gates:\n  - retriever: bm25\n    metric: recall@1\n    min_value: 2.0"
    )
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "quality_gates: []",
            failing_gate,
        ),
        encoding="utf-8",
    )
    assert main(["run", "-c", str(config), "-f", "json"]) == 0
    capsys.readouterr()
    result = project / "reports/result.json"

    drop_config = config.read_text(encoding="utf-8").replace(
        "min_value: 2.0", "max_absolute_drop: 0.1"
    )
    config.write_text(drop_config, encoding="utf-8")
    assert main(["gate", "-c", str(config), str(result)]) == 2
    capsys.readouterr()
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "max_absolute_drop: 0.1", "min_value: 2.0"
        ),
        encoding="utf-8",
    )

    assert main(["gate", "-c", str(config), str(result)]) == 1
    failure = capsys.readouterr()
    assert "FAIL" in failure.out
    assert failure.err == ""

    def crash(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("retrieval_lab.cli.app.inspect_result", crash)
    assert main(["inspect", str(result)]) == 3
    normal = capsys.readouterr()
    assert "Traceback" not in normal.err
    assert main(["inspect", str(result), "--debug"]) == 3
    debug = capsys.readouterr()
    assert "Traceback" in debug.err


def test_gate_accepts_embedded_candidate_configuration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    config = project / "retrieval-lab.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "quality_gates: []",
            "quality_gates:\n  - retriever: bm25\n    metric: recall@1\n"
            "    min_value: 0.0",
        ),
        encoding="utf-8",
    )
    assert main(["run", "-c", str(config), "-f", "json"]) == 0
    capsys.readouterr()
    result = project / "reports/result.json"

    assert main(["gate", str(result), "--baseline", str(result)]) == 0
    assert "PASS" in capsys.readouterr().out

    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["run"]["manifest"]["config"]["quality_gates"] = []
    result.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["gate", str(result)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "retrieval-lab: configuration or input error\n"


def test_compare_incomparable_and_malformed_inputs_return_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    result = project / "missing.json"
    assert main(["compare", str(result), str(result)]) == 2
    error = capsys.readouterr()
    assert "Traceback" not in error.err

    malformed = project / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    assert main(["inspect", str(malformed)]) == 2
    malformed_error = capsys.readouterr()
    assert malformed_error.out == ""
    assert "Traceback" not in malformed_error.err

    unknown_schema = project / "unknown-schema.json"
    unknown_schema.write_text('{"schema_version": 99}\n', encoding="utf-8")
    assert main(["inspect", str(unknown_schema)]) == 2
    schema_error = capsys.readouterr()
    assert schema_error.out == ""
    assert "Traceback" not in schema_error.err

    second = tmp_path / "second"
    assert main(["init", str(second)]) == 0
    capsys.readouterr()
    second_config = second / "retrieval-lab.yaml"
    second_config.write_text(
        second_config.read_text(encoding="utf-8").replace(
            "top_k: [1, 3]", "top_k: [1]"
        ),
        encoding="utf-8",
    )
    assert main(["run", "-c", str(second_config), "-f", "json"]) == 0
    capsys.readouterr()

    first = tmp_path / "first"
    assert main(["init", str(first)]) == 0
    capsys.readouterr()
    assert main(["run", "-c", str(first / "retrieval-lab.yaml"), "-f", "json"]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "compare",
                str(first / "reports/result.json"),
                str(second / "reports/result.json"),
            ]
        )
        == 2
    )
    incomparable = capsys.readouterr()
    assert "Traceback" not in incomparable.err


def test_cli_default_run_writes_loadable_canonical_reports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    assert main(["init", str(project)]) == 0
    capsys.readouterr()

    assert main(["run", "-c", str(project / "retrieval-lab.yaml")]) == 0
    captured = capsys.readouterr()

    result = load_result(project / "reports/result.json")
    assert result.schema_version == 1
    assert tuple(result.metrics) == ("bm25", "keyword")
    assert (project / "reports/summary.csv").is_file()
    assert (project / "reports/per_query.csv").is_file()
    assert (project / "reports/report.html").is_file()
    assert "result.json" in captured.out
    assert captured.err == ""


def test_cli_malformed_config_is_exit_two_without_details(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "private-config.yaml"
    config.write_text("not: [valid", encoding="utf-8")

    assert main(["validate", "-c", str(config)]) == 2
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == "retrieval-lab: configuration or input error\n"
    assert str(config) not in captured.err


def test_cli_unexpected_runtime_error_is_exit_three_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("secret runtime detail")

    monkeypatch.setattr(cli_app, "run_configured_experiment", fail)

    status = main(["run", "-c", str(tmp_path / "config.yaml")])
    captured = capsys.readouterr()

    assert status == 3
    assert captured.out == ""
    assert captured.err == "retrieval-lab: unexpected runtime error\n"
    assert "secret" not in captured.err
    assert "Traceback" not in captured.err

    status = main(["run", "-c", str(tmp_path / "config.yaml"), "--debug"])
    debug = capsys.readouterr()
    assert status == 3
    assert "Traceback" in debug.err


@pytest.mark.parametrize(
    "error",
    [
        OptionalDependencyError("missing optional dependency"),
        EvaluationError("evaluation failed"),
        RetrieverContractError("retriever failed"),
    ],
)
def test_cli_known_run_failures_are_exit_three(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise error

    monkeypatch.setattr(cli_app, "run_configured_experiment", fail)

    assert main(["run", "-c", str(tmp_path / "config.yaml")]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "retrieval-lab: evaluation error\n"
