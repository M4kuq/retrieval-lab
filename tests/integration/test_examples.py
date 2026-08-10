"""Keep repository examples executable as public API contracts."""

import runpy
from pathlib import Path

from retrieval_lab import load_result


def test_file_input_example_runs_without_network_access() -> None:
    namespace = runpy.run_path("examples/from_files.py")

    assert namespace["result"].metrics["bm25"].recall_at(1) == 1.0


def test_readme_result_report_example_runs_without_network_access(
    tmp_path: Path,
) -> None:
    namespace = runpy.run_path("examples/from_files.py")
    result = namespace["result"]
    result_path = tmp_path / "artifacts" / "result.json"
    result.save_json(result_path)
    loaded = load_result(result_path)
    loaded.save_csv(tmp_path / "artifacts" / "reports")
    loaded.save_html(tmp_path / "artifacts" / "reports" / "report.html")

    assert loaded.run_id == result.run_id
    assert "run_id:" in loaded.summary()
    assert (tmp_path / "artifacts" / "reports" / "summary.csv").exists()
    assert (tmp_path / "artifacts" / "reports" / "per_query.csv").exists()
