"""Keep repository examples and tutorial artifacts executable contracts."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_comparison_statistics_example_runs_without_network_access() -> None:
    namespace = runpy.run_path("examples/comparison_statistics.py")

    bootstrap = namespace["bootstrap"]
    assert bootstrap.observed_delta == 1.0
    assert (bootstrap.ci_lower, bootstrap.ci_upper) == (1.0, 1.0)
    assert bootstrap.significant is True
    assert namespace["comparison"].comparability.comparable is True


def test_standard_loaders_example_runs_without_network_access() -> None:
    namespace = runpy.run_path("examples/standard_loaders.py")

    msmarco = namespace["msmarco"]
    assert msmarco.excluded_query_ids == ("q3",)
    assert msmarco.dataset.relevance_grades_by_query["q1"] == {"p1": 2, "p2": 1}
    assert namespace["result"].metrics["bm25"].recall_at(3) == 1.0
    natural_questions = namespace["natural_questions"]
    assert natural_questions.excluded_query_ids == ("103",)
    assert natural_questions.dataset.relevance_grades_by_query["101"] == {"101": 1}
    assert namespace["nq_result"].metrics["bm25"].recall_at(1) == 1.0


def test_checkpoint_resume_example_runs_without_network_access() -> None:
    namespace = runpy.run_path("examples/checkpoint_resume.py")

    # The first pass crashes on q2, so the resume re-runs only q2 and q3.
    assert namespace["attempts"] == ["q1", "q2", "q2", "q3"]
    assert namespace["result"].metrics["checkpointed"].recall_at(1) == 1.0


def test_benchmark_loader_example_runs_without_network_access() -> None:
    namespace = runpy.run_path("examples/benchmark_loaders.py")

    assert namespace["result"].metrics["bm25"].recall_at(3) == 1.0
    # nDCG depends on the loaded grades (q1: d1=2, d3=1); flattening the
    # grades in the loader or the runner would change this value.
    bm25_q1 = next(
        evaluation
        for evaluation in namespace["result"].query_results["bm25"]
        if evaluation.query_id == "q1"
    )
    assert bm25_q1.metrics_by_cutoff[3]["ndcg"] == pytest.approx(0.58688267143572)
    fused = namespace["fused_result"]
    assert fused.manifest["fusion"] == {"rrf_k": 60, "sources": ["bm25", "keyword"]}
    assert fused.metrics["hybrid"].recall_at(3) == 1.0
    # The fused q1 order (d1, d2, d3) matches neither source: keyword
    # gives (d1, d3, d2) and BM25 gives (d2, d3, d1), so a fusion that
    # ignored either source would fail this assertion.
    fused_by_query = {
        evaluation.query_id: evaluation.retrieved_ids
        for evaluation in fused.query_results["hybrid"]
    }
    assert fused_by_query == {"q1": ("d1", "d2", "d3"), "q2": ("d4",)}


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
