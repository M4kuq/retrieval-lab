"""Generate and structurally validate the offline tutorial notebook.

The repository does not require Jupyter as a runtime dependency. When
``nbformat`` is installed, it supplies the notebook cell scaffold; otherwise a
small standard-library fallback writes the same v4 JSON shape. Execution is
performed by the integration test with nbclient when available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol


class _NotebookBuilder(Protocol):
    def new_markdown_cell(self, source: str) -> object: ...

    def new_code_cell(self, source: str) -> object: ...

    def new_notebook(
        self, *, cells: list[object], metadata: dict[str, object]
    ) -> object: ...


SECTION_HEADINGS = ("## Goal", "## Setup", "## Steps", "## Checks", "## Next Steps")


def _fallback_markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def _fallback_code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.rstrip("\n").splitlines(True),
    }


def build_notebook() -> dict[str, object]:
    """Return the deterministic tutorial notebook payload."""

    markdown = [
        "# Retrieval Lab: 日本語 BM25 tutorial\n\n",
        "## Goal\n\n"
        "ローカルの日本語 corpus と qrels を使い、BM25 の Recall を確認します。\n"
        "この notebook はネットワーク、実モデル、秘密情報を使いません。\n",
        "## Setup\n\n"
        "入力は `examples/japanese/corpus` と `examples/japanese/qrels.jsonl` です。\n"
        "実行時の current working directory はリポジトリルートを想定し、\n"
        "見つからない場合は親ディレクトリを探します。文書 relevance を評価します。\n",
        "## Steps\n\n### 1. Load the local inputs\n",
        "### 2. Run BM25\n\n"
        "すべての query を一度検索し、top_k は 1 と 3 に固定します。\n",
        "## Checks\n\n"
        "結果の retriever 名、query 数、Recall の範囲を小さな assert で確認します。\n",
        "## Next Steps\n\n"
        "Dense/Hybrid、Callable Retriever、precomputed ranking の例は\n"
        "`examples/` と `docs/tutorial.md` を参照してください。\n",
    ]
    code = [
        "from pathlib import Path\n\n"
        "from retrieval_lab import EvaluationDataset, EvaluationRunner, "
        "load_documents\n\n"
        "project_root = next(\n"
        "    (\n"
        "        candidate\n"
        "        for candidate in (Path.cwd(), *Path.cwd().parents)\n"
        '        if (candidate / "src" / "retrieval_lab").is_dir()\n'
        "    ),\n"
        "    None,\n"
        ")\n"
        "assert project_root is not None, "
        '"run from the repository or a child directory"\n'
        'corpus_path = project_root / "examples" / "japanese" / "corpus"\n'
        'qrels_path = project_root / "examples" / "japanese" / "qrels.jsonl"\n'
        "assert corpus_path.is_dir() and qrels_path.is_file()\n",
        "documents = load_documents(corpus_path)\n"
        "dataset = EvaluationDataset.from_jsonl(qrels_path)\n"
        'print(f"documents={len(documents)}, queries={len(dataset.queries)}")\n',
        "result = EvaluationRunner.from_dataset(\n"
        "    documents=documents,\n"
        "    dataset=dataset,\n"
        '    strategies=("bm25",),\n'
        "    top_k=(1, 3),\n"
        "    seed=42,\n"
        ").run()\n"
        "metrics_preview = {\n"
        '    f"Recall@{cutoff}": round(result.metrics["bm25"].recall_at(cutoff), 3)\n'
        "    for cutoff in (1, 3)\n"
        "}\n"
        "print(metrics_preview)\n",
        'assert result.manifest["retrievers"] == ["bm25"]\n'
        'assert len(result.query_results["bm25"]) == 4\n'
        "assert all(0.0 <= value <= 1.0 for value in metrics_preview.values())\n"
        'print("checks=ok")\n',
    ]

    try:
        import nbformat

        builder = nbformat.v4
        cells = [
            builder.new_markdown_cell(markdown[0]),
            builder.new_markdown_cell(markdown[1]),
            builder.new_markdown_cell(markdown[2]),
            builder.new_markdown_cell(markdown[3]),
            builder.new_code_cell(code[0].rstrip("\n")),
            builder.new_code_cell(code[1].rstrip("\n")),
            builder.new_markdown_cell(markdown[4]),
            builder.new_markdown_cell(markdown[5]),
            builder.new_code_cell(code[2].rstrip("\n")),
            builder.new_code_cell(code[3].rstrip("\n")),
            builder.new_markdown_cell(markdown[6]),
        ]
        notebook = builder.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
            },
        )
        return dict(notebook)
    except ImportError:
        cells_fallback: list[dict[str, object]] = []
        for index, source in enumerate(markdown):
            cells_fallback.append(_fallback_markdown(source))
            if index == 3:
                cells_fallback.extend(
                    _fallback_code(source + "\n") for source in code[:2]
                )
            elif index == 5:
                cells_fallback.extend(
                    _fallback_code(source + "\n") for source in code[2:]
                )
        return {
            "cells": cells_fallback,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }


def validate_notebook(payload: object) -> None:
    """Validate the tutorial structure without importing notebook packages."""

    if not isinstance(payload, dict) or payload.get("nbformat") != 4:
        raise ValueError("tutorial notebook must be nbformat v4")
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("tutorial notebook must contain cells")
    markdown = "\n".join(
        str(cell.get("source", ""))
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "markdown"
    )
    positions = [markdown.find(heading) for heading in SECTION_HEADINGS]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise ValueError("tutorial sections are missing or out of order")
    code_cells = [
        cell
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    ]
    if len(code_cells) != 4:
        raise ValueError("tutorial notebook must contain four code cells")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("notebooks/japanese_bm25_tutorial.ipynb"),
    )
    parser.add_argument(
        "--check", action="store_true", help="validate an existing notebook"
    )
    args = parser.parse_args()
    if args.check:
        validate_notebook(json.loads(args.output.read_text(encoding="utf-8")))
        print(f"valid tutorial notebook: {args.output}")
        return 0
    notebook = build_notebook()
    validate_notebook(notebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"generated tutorial notebook: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
