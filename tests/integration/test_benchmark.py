"""Integration coverage for the local benchmark harness."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import sysconfig
from copy import deepcopy
from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest

from retrieval_lab.exceptions import EvaluationError

# ``benchmarks`` is intentionally a repository-local harness, not a package
# dependency. Make it importable when pytest starts with only ``src`` on PATH.
REPOSITORY_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

_harness = importlib.import_module("benchmarks.harness")
_benchmark_cli = importlib.import_module("benchmarks.run")
BenchmarkSpec = _harness.BenchmarkSpec
generate_synthetic_data = _harness.generate_synthetic_data
load_benchmark = _harness.load_benchmark
run_benchmark = _harness.run_benchmark
save_benchmark = _harness.save_benchmark
main = _benchmark_cli.main


def test_synthetic_tiers_are_deterministic_and_bounded() -> None:
    small = BenchmarkSpec(size="small", seed=0)
    first = generate_synthetic_data(small)
    second = generate_synthetic_data(small)
    medium = generate_synthetic_data(BenchmarkSpec(size="medium"))

    assert first.documents == second.documents
    assert first.dataset.queries == second.dataset.queries
    assert len(first.documents) == 24
    assert len(first.dataset.queries) == 8
    assert len(medium.documents) == 160
    assert len(medium.dataset.queries) == 40
    assert all(document.text.startswith("synthetic ") for document in first.documents)


def test_benchmark_cold_warm_report_is_strict_and_reproducible(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "benchmark.json"
    payload = run_benchmark(BenchmarkSpec(size="small", seed=42))
    destination = save_benchmark(payload, output)
    loaded = load_benchmark(destination)

    assert destination == output.absolute()
    assert loaded["schema_version"] == 1
    benchmark = loaded["benchmark"]
    assert benchmark["dataset_id"] == "retrieval-lab.synthetic"
    assert benchmark["dataset_version"] == "1"
    assert benchmark["document_count"] == 24
    assert benchmark["query_count"] == 8
    assert benchmark["retrievers"] == ["keyword", "bm25"]
    assert benchmark["seed"] == 42
    assert benchmark["top_k"] == [1, 3, 5]
    assert benchmark["repetitions"] == 1
    assert loaded["comparisons"] == {"metrics_equal": True, "run_id_equal": True}

    runs = loaded["runs"]
    assert runs["cold"]["cache"] == "cold"
    assert runs["warm"]["cache"] == "warm"
    assert runs["cold"]["run_id"] == runs["warm"]["run_id"]
    assert runs["cold"]["metrics"] == runs["warm"]["metrics"]
    assert runs["cold"]["duration_ms"] >= 0.0
    assert runs["warm"]["duration_ms"] >= 0.0
    assert {event["status"] for event in runs["cold"]["cache_events"]} == {"miss"}
    assert {event["status"] for event in runs["warm"]["cache_events"]} == {"hit"}
    assert set(runs["cold"]["latency"]) == {"keyword", "bm25"}
    assert set(runs["cold"]["metrics"]) == {"keyword", "bm25"}
    assert loaded["environment"]["python"]
    assert loaded["environment"]["retrieval_lab_version"] == importlib_metadata.version(
        "retrieval-lab-sdk"
    )
    assert loaded["started_at_utc"].endswith("Z")
    assert loaded["finished_at_utc"].endswith("Z")
    assert loaded["duration_ms"] >= 0.0

    serialized = output.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "synthetic query" not in serialized
    assert "synthetic document" not in serialized
    assert "marker-42" not in serialized
    assert "NaN" not in serialized
    assert "Infinity" not in serialized


def test_benchmark_cli_writes_report_and_rejects_invalid_repetitions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "cli" / "result.json"
    assert main(["--size", "small", "--seed", "0", "--output", str(output)]) == 0
    message = capsys.readouterr()
    assert message.err == ""
    assert message.out == "benchmark report written: result.json\n"
    assert load_benchmark(output)["benchmark"]["seed"] == 0

    assert main(["--repetitions", "2", "--output", str(tmp_path / "bad.json")]) == 2
    assert "configuration or input error" in capsys.readouterr().err


def test_benchmark_output_rejects_symlink_and_non_finite_values(
    tmp_path: Path,
) -> None:
    payload = run_benchmark(BenchmarkSpec())
    with pytest.raises(EvaluationError, match="NaN"):
        save_benchmark({"value": float("nan")}, tmp_path / "nan.json")

    if not hasattr(Path, "symlink_to"):
        pytest.skip("symlinks are unavailable")
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "link"
    try:
        symlink.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(EvaluationError, match="symlink"):
        save_benchmark(payload, symlink / "benchmark.json")


def test_benchmark_loader_rejects_duplicate_keys_and_non_finite_constants(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}\n')
    with pytest.raises(EvaluationError, match="strict JSON"):
        load_benchmark(duplicate)

    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text('{"schema_version": NaN}\n')
    with pytest.raises(EvaluationError, match="strict JSON"):
        load_benchmark(non_finite)

    # The writer uses canonical sorted JSON, making a second load/save stable.
    payload = run_benchmark(BenchmarkSpec())
    output = tmp_path / "stable.json"
    save_benchmark(payload, output)
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == 1


def test_benchmark_loader_rejects_overflowed_numbers(tmp_path: Path) -> None:
    overflowed = tmp_path / "overflowed.json"
    overflowed.write_text(
        '{"schema_version": 1, "duration_ms": 1e400}\n', encoding="utf-8"
    )

    with pytest.raises(EvaluationError, match="infinity"):
        load_benchmark(overflowed)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["benchmark"].__setitem__("top_k", "1,3,5"),
        lambda payload: payload["comparisons"].__setitem__("metrics_equal", 1),
        lambda payload: payload["environment"].__setitem__("os", []),
        lambda payload: payload["runs"]["cold"].__setitem__("metrics", []),
        lambda payload: payload["runs"]["cold"]["latency"]["keyword"].__setitem__(
            "warnings", "warning"
        ),
        lambda payload: payload["runs"]["cold"]["cache_events"][0].__setitem__(
            "duration_ms", "slow"
        ),
    ],
)
def test_benchmark_loader_rejects_malformed_nested_report_fields(
    tmp_path: Path, mutate: object
) -> None:
    payload = deepcopy(run_benchmark(BenchmarkSpec()))
    mutate(payload)  # type: ignore[operator]
    output = tmp_path / "malformed.json"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationError):
        load_benchmark(output)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["runs"]["cold"].__setitem__("cache", "warm"),
        lambda payload: payload["comparisons"].__setitem__("run_id_equal", False),
        lambda payload: payload["comparisons"].__setitem__("metrics_equal", False),
        lambda payload: payload["runs"]["cold"]["build_ms"].pop("keyword"),
        lambda payload: payload["runs"]["cold"]["latency"].pop("keyword"),
        lambda payload: payload["runs"]["cold"]["index_sizes_bytes"].pop("keyword"),
        lambda payload: payload["runs"]["cold"]["metrics"]["keyword"].pop("recall@1"),
        lambda payload: payload["benchmark"].__setitem__("document_count", 999),
        lambda payload: payload["benchmark"].__setitem__("query_count", 999),
        lambda payload: payload["benchmark"].__setitem__("relevance_level", "chunk"),
        lambda payload: payload["benchmark"].__setitem__("dataset_id", "other"),
        lambda payload: payload["benchmark"].__setitem__("dataset_version", "2"),
        lambda payload: payload["benchmark"].__setitem__("retrievers", ["bm25"]),
        lambda payload: (
            payload["runs"]["cold"]["metrics"]["keyword"].__setitem__("recall@1", 42.0),
            payload["runs"]["warm"]["metrics"]["keyword"].__setitem__("recall@1", 42.0),
        ),
    ],
)
def test_benchmark_saver_rejects_contradictory_producer_invariants(
    tmp_path: Path, mutate: object
) -> None:
    payload = deepcopy(run_benchmark(BenchmarkSpec()))
    mutate(payload)  # type: ignore[operator]

    with pytest.raises(EvaluationError):
        save_benchmark(payload, tmp_path / "contradictory.json")


@pytest.mark.parametrize("field", ["run_id_equal", "metrics_equal"])
def test_benchmark_loader_rejects_false_or_mismatched_equality(
    tmp_path: Path, field: str
) -> None:
    payload = deepcopy(run_benchmark(BenchmarkSpec()))
    if field == "run_id_equal":
        payload["runs"]["warm"]["run_id"] = "different"
    else:
        payload["runs"]["warm"]["metrics"]["keyword"]["recall@1"] = 0.0
    payload["comparisons"][field] = False
    output = tmp_path / f"mismatch-{field}.json"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationError):
        load_benchmark(output)


def test_benchmark_additive_fields_round_trip(tmp_path: Path) -> None:
    payload = run_benchmark(BenchmarkSpec())
    payload["extra_root"] = True
    payload["benchmark"]["extra_benchmark"] = "kept"
    payload["environment"]["extra_environment"] = 1
    payload["environment"]["os"]["extra_os"] = "kept"
    payload["runs"]["cold"]["extra_run"] = None
    payload["runs"]["cold"]["latency"]["keyword"]["extra_latency"] = []
    payload["runs"]["cold"]["cache_events"][0]["extra_event"] = False
    output = tmp_path / "additive.json"

    save_benchmark(payload, output)
    loaded = load_benchmark(output)

    assert loaded["extra_root"] is True
    assert loaded["benchmark"]["extra_benchmark"] == "kept"
    assert loaded["runs"]["cold"]["cache_events"][0]["extra_event"] is False


def test_direct_benchmark_script_bootstraps_an_uninstalled_checkout(
    tmp_path: Path,
) -> None:
    output = tmp_path / "direct.json"
    isolated_python = getattr(sys, "_base_executable", sys.executable)
    environment = os.environ.copy()
    # Keep installed third-party dependencies such as PyYAML available without
    # processing the editable install's ``.pth`` file.  The checkout bootstrap
    # must therefore remain responsible for making ``retrieval_lab`` importable.
    environment["PYTHONPATH"] = os.pathsep.join(
        (sysconfig.get_path("purelib"), str(tmp_path / "external-only"))
    )
    completed = subprocess.run(
        [
            isolated_python,
            str(REPOSITORY_ROOT / "benchmarks" / "run.py"),
            "--size",
            "small",
            "--seed",
            "0",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "benchmark report written: direct.json\n"
    assert completed.stderr == ""
    assert load_benchmark(output)["benchmark"]["seed"] == 0
