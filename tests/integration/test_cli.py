from __future__ import annotations

from pathlib import Path

import pytest

import retrieval_lab.cli.app as cli_app
from retrieval_lab import load_result
from retrieval_lab.cli.app import main


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
