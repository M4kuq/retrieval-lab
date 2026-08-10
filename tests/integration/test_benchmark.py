"""Integration coverage for the local benchmark harness."""

from __future__ import annotations

import importlib
import json
import sys
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
        "retrieval-lab"
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
