"""Keep repository examples executable as public API contracts."""

import runpy


def test_file_input_example_runs_without_network_access() -> None:
    namespace = runpy.run_path("examples/from_files.py")

    assert namespace["result"].metrics["bm25"].recall_at(1) == 1.0
