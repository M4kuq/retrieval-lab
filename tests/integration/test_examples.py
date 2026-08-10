"""Keep repository examples and tutorial artifacts executable contracts."""

import json
import runpy
import subprocess
import sys
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


def test_japanese_python_examples_run_without_network_access() -> None:
    root = Path(__file__).parents[2]
    for relative_path in (
        "examples/api_quickstart.py",
        "examples/dense_hybrid_comparison.py",
        "examples/custom_callable.py",
        "examples/precomputed_ranking.py",
    ):
        namespace = runpy.run_path(str(root / relative_path))
        assert namespace["result"].schema_version == 1


def test_japanese_yaml_is_executable_through_config_api() -> None:
    from retrieval_lab import EvaluationRunner, load_config

    root = Path(__file__).parents[2]
    config_path = root / "examples" / "japanese" / "retrieval-lab.yaml"
    config = load_config(config_path)
    result = EvaluationRunner.from_config(config).run()

    assert set(result.metrics) == {"keyword", "bm25"}
    assert result.manifest["config"]["dataset"]["path"] == "qrels.jsonl"


def test_tutorial_notebook_has_sections_and_executes_top_to_bottom(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    notebook_path = root / "notebooks" / "japanese_bm25_tutorial.ipynb"
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    validator_namespace = runpy.run_path(
        str(root / "tools" / "generate_tutorial_notebook.py")
    )
    validator_namespace["validate_notebook"](payload)

    copied_path = tmp_path / notebook_path.name
    copied_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    cells = payload["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code_cells) == 4

    try:
        import nbclient
        import nbformat
    except ImportError:
        namespace: dict[str, object] = {}
        for cell in code_cells:
            source = "".join(cell["source"])
            exec(compile(source, str(notebook_path), "exec"), namespace, namespace)
    else:
        executed = nbformat.read(copied_path, as_version=4)
        nbclient.NotebookClient(
            executed,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        ).execute()


def test_tutorial_generator_check_command() -> None:
    root = Path(__file__).parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "tools/generate_tutorial_notebook.py",
            "notebooks/japanese_bm25_tutorial.ipynb",
            "--check",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "valid tutorial notebook" in completed.stdout
