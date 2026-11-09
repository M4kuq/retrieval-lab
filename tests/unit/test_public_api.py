from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType

import retrieval_lab
from retrieval_lab import (
    BM25Retriever,
    ComparisonOutput,
    ComparisonRow,
    DenseRetriever,
    Document,
    EmbeddingBackend,
    EmbeddingModelMetadata,
    EvaluationDataset,
    EvaluationRunner,
    ExperimentOutput,
    GateOutput,
    HybridRetriever,
    InitializedProject,
    InspectionOutput,
    KeywordRetriever,
    OptionalDependencyError,
    QueryEvidence,
    RetrievalLabError,
    RetrievedQueryResult,
    ValidationResult,
    compare_result_files,
    evaluate_configured_quality_gates,
    evaluate_results,
    initialize_project,
    inspect_result,
    load_documents,
    run_configured_experiment,
    validate_config_inputs,
    validate_dataset,
)
from retrieval_lab.errors import RetrievalLabError as ErrorAlias
from retrieval_lab.models import Document as DocumentAlias

ROOT = Path(__file__).resolve().parents[2]
API_DOCUMENT = ROOT / "docs" / "api-minimum.md"


def test_package_root_and_compatibility_modules_share_public_types() -> None:
    assert DocumentAlias is Document
    assert ErrorAlias is RetrievalLabError
    assert EvaluationRunner.__module__ == "retrieval_lab.runner"


def test_new_milestone_apis_are_available_from_the_package_root() -> None:
    assert BM25Retriever.__module__ == "retrieval_lab.retrievers.bm25"
    assert DenseRetriever.__module__ == "retrieval_lab.retrievers.dense"
    assert HybridRetriever.__module__ == "retrieval_lab.retrievers.hybrid"
    assert EmbeddingBackend.__module__ == "retrieval_lab.retrievers.dense"
    assert EmbeddingModelMetadata.__module__ == "retrieval_lab.retrievers.dense"
    assert OptionalDependencyError.__module__ == "retrieval_lab.exceptions"
    assert EvaluationDataset.__module__ == "retrieval_lab.datasets"
    assert RetrievedQueryResult.__module__ == "retrieval_lab.evaluation.precomputed"
    assert evaluate_results.__module__ == "retrieval_lab.evaluation.precomputed"
    assert load_documents.__module__ == "retrieval_lab.loaders"
    assert validate_dataset.__module__ == "retrieval_lab.datasets"


def test_hybrid_constructor_smoke_uses_public_package_imports() -> None:
    hybrid = HybridRetriever(
        [KeywordRetriever(), BM25Retriever()], rrf_k=7, candidate_k=9
    )

    assert hybrid.settings["rrf_k"] == 7
    assert hybrid.settings["candidate_k"] == 9
    assert [entry["name"] for entry in hybrid.settings["sources"]] == [
        "bm25",
        "keyword",
    ]


def test_application_services_are_available_from_the_package_root() -> None:
    assert InitializedProject.__module__ == "retrieval_lab.application"
    assert ValidationResult.__module__ == "retrieval_lab.application"
    assert ExperimentOutput.__module__ == "retrieval_lab.application"
    assert initialize_project.__module__ == "retrieval_lab.application"
    assert validate_config_inputs.__module__ == "retrieval_lab.application"
    assert run_configured_experiment.__module__ == "retrieval_lab.application"
    assert QueryEvidence.__module__ == "retrieval_lab.application"
    assert InspectionOutput.__module__ == "retrieval_lab.application"
    assert ComparisonRow.__module__ == "retrieval_lab.application"
    assert ComparisonOutput.__module__ == "retrieval_lab.application"
    assert GateOutput.__module__ == "retrieval_lab.application"
    assert inspect_result.__module__ == "retrieval_lab.application"
    assert compare_result_files.__module__ == "retrieval_lab.application"
    assert evaluate_configured_quality_gates.__module__ == "retrieval_lab.application"


def test_public_all_is_sorted_and_duplicate_free() -> None:
    names = list(retrieval_lab.__all__)
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_every_export_is_importable_from_the_package_root() -> None:
    for name in retrieval_lab.__all__:
        assert hasattr(retrieval_lab, name), name


def test_package_root_exposes_no_undocumented_names() -> None:
    exposed = {name for name in dir(retrieval_lab) if not name.startswith("_")}
    submodules = {
        name for name in exposed if isinstance(getattr(retrieval_lab, name), ModuleType)
    }
    assert exposed - submodules == set(retrieval_lab.__all__)


def test_public_imports_document_lists_exactly_the_exports() -> None:
    text = API_DOCUMENT.read_text(encoding="utf-8")
    start = text.index("## Public imports")
    end = text.index("## ", start + len("## Public imports"))
    documented: set[str] = set()
    in_bullet = False
    for line in text[start:end].splitlines():
        if line.startswith("- "):
            in_bullet = True
        elif line and not line[0].isspace():
            in_bullet = False
        if in_bullet:
            documented.update(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", line))
    assert documented == set(retrieval_lab.__all__)


def test_distribution_is_typed() -> None:
    package_dir = Path(retrieval_lab.__file__).parent
    assert (package_dir / "py.typed").is_file()
