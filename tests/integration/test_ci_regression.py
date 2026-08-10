"""Offline contracts for the published CI regression-gate fixtures."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from retrieval_lab import evaluate_configured_quality_gates, load_config, load_result

ROOT = Path(__file__).resolve().parents[2]
CI_EXAMPLES = ROOT / "examples" / "ci"


def _run_gate(candidate: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_path, environment.get("PYTHONPATH", "")) if item
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "retrieval_lab.cli.app",
            "gate",
            "--config",
            str(CI_EXAMPLES / "retrieval-lab.yaml"),
            "--baseline",
            str(CI_EXAMPLES / "baseline.json"),
            str(CI_EXAMPLES / candidate),
            "--json",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_ci_fixtures_are_strict_results_and_config_is_normalized() -> None:
    results = {
        name: load_result(CI_EXAMPLES / name)
        for name in (
            "baseline.json",
            "candidate-improved.json",
            "candidate-regressed.json",
        )
    }

    assert results["baseline.json"].schema_version == 1
    assert results["baseline.json"].manifest["query_ids"] == ["q-ci-1"]
    assert results["candidate-improved.json"].manifest["dataset_hash"] == (
        "ci-dataset-v1"
    )
    assert results["candidate-regressed.json"].metrics["bm25"].recall_at(1) == 0.6

    config = load_config(CI_EXAMPLES / "retrieval-lab.yaml")
    assert config.quality_gates[0].max_absolute_drop == 0.1
    assert config.quality_gates[0].max_relative_drop == 0.2
    assert config.normalized_settings()["corpus"]["path"] == "corpus"
    assert config.normalized_settings()["dataset"]["path"] == "dataset.jsonl"


def test_cli_and_application_api_have_matching_gate_decisions() -> None:
    improved = evaluate_configured_quality_gates(
        CI_EXAMPLES / "retrieval-lab.yaml",
        CI_EXAMPLES / "candidate-improved.json",
        baseline_path=CI_EXAMPLES / "baseline.json",
    )
    regressed = evaluate_configured_quality_gates(
        CI_EXAMPLES / "retrieval-lab.yaml",
        CI_EXAMPLES / "candidate-regressed.json",
        baseline_path=CI_EXAMPLES / "baseline.json",
    )
    assert improved.report.passed
    assert not regressed.report.passed
    assert all(check.passed for check in improved.report.results[0].checks)
    assert all(not check.passed for check in regressed.report.results[0].checks)

    improved_process = _run_gate("candidate-improved.json")
    regressed_process = _run_gate("candidate-regressed.json")
    assert improved_process.returncode == 0, improved_process.stderr
    assert regressed_process.returncode == 1, regressed_process.stderr
    improved_payload = json.loads(improved_process.stdout)
    regressed_payload = json.loads(regressed_process.stdout)
    assert improved_payload["passed"] is improved.report.passed
    assert regressed_payload["passed"] is regressed.report.passed
    assert len(improved_payload["quality_gates"][0]["checks"]) == 2
    assert len(regressed_payload["quality_gates"][0]["checks"]) == 2
    assert regressed_process.stderr == ""


def test_ci_workflows_are_safe_and_pin_read_only_commands() -> None:
    workflow_paths = (
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / "examples" / "github-actions" / "retrieval-quality-gate.yml",
    )
    workflows = {}
    for path in workflow_paths:
        with path.open(encoding="utf-8") as stream:
            parsed = yaml.safe_load(stream)
        assert isinstance(parsed, dict)
        workflows[path] = parsed
        permissions = parsed.get("permissions")
        assert permissions == {"contents": "read"}
        assert all(
            not isinstance(value, str) or "secrets." not in value
            for value in _run_values(parsed)
        )

    example = workflows[workflow_paths[1]]
    example_jobs = example["jobs"]
    assert isinstance(example_jobs, dict)
    example_steps = example_jobs["retrieval-quality-gate"]["steps"]
    example_runs = _run_values(example_steps)
    assert any("actions/checkout@v4" in value for value in _uses_values(example_steps))
    assert any("setup-uv" in value for value in _uses_values(example_steps))
    assert any("uv sync" in value for value in example_runs)
    assert any("retrieval-lab gate" in value for value in example_runs)
    assert any("--baseline" in value for value in example_runs)

    active = workflows[workflow_paths[0]]
    jobs = active["jobs"]
    assert isinstance(jobs, dict)
    assert {"quality-gate", "wheel-smoke", "install-matrix"} <= set(jobs)
    quality_runs = _run_values(jobs["quality-gate"]["steps"])
    assert any("candidate-improved.json" in value for value in quality_runs)
    assert any("candidate-regressed.json" in value for value in quality_runs)
    assert any('test "$status" -eq 0' in value for value in quality_runs)
    assert any('test "$status" -eq 1' in value for value in quality_runs)
    wheel_runs = _run_values(jobs["wheel-smoke"]["steps"])
    assert any("uv build --wheel" in value for value in wheel_runs)
    assert any("retrieval-lab init" in value for value in wheel_runs)
    assert any("retrieval-lab validate" in value for value in wheel_runs)
    assert any("retrieval-lab run" in value for value in wheel_runs)
    assert any("retrieval-lab inspect" in value for value in wheel_runs)
    matrix = jobs["install-matrix"]["strategy"]["matrix"]
    assert matrix["extra"] == ["core", "dense"]
    install_runs = _run_values(jobs["install-matrix"]["steps"])
    assert any("DenseRetriever" in value for value in install_runs)
    assert any("sentence_transformers" in value for value in install_runs)


def _run_values(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        values: list[str] = []
        run = value.get("run")
        if isinstance(run, str):
            values.append(run)
        for child in value.values():
            values.extend(_run_values(child))
        return tuple(values)
    if isinstance(value, list):
        return tuple(item for child in value for item in _run_values(child))
    return ()


def _uses_values(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        values: list[str] = []
        uses = value.get("uses")
        if isinstance(uses, str):
            values.append(uses)
        for child in value.values():
            values.extend(_uses_values(child))
        return tuple(values)
    if isinstance(value, list):
        return tuple(item for child in value for item in _uses_values(child))
    return ()
