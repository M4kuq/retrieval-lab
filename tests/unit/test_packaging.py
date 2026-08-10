"""Release-candidate metadata, documentation, and artifact contracts."""

from __future__ import annotations

import email
import json
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

import yaml

from retrieval_lab import Document, EvaluationQuery, EvaluationRunner

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_URLS = {
    "Homepage": "https://github.com/M4kuq/retrieval-lab",
    "Documentation": "https://m4kuq.github.io/retrieval-lab/",
    "Repository": "https://github.com/M4kuq/retrieval-lab",
    "Issues": "https://github.com/M4kuq/retrieval-lab/issues",
    "Changelog": "https://github.com/M4kuq/retrieval-lab/blob/main/CHANGELOG.md",
}


def test_release_metadata_and_manifest_are_explicit() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert project["name"] == "retrieval-lab-sdk"
    assert project["version"] == "0.1.0rc1"
    assert project["readme"] == "docs/index.md"
    assert "Development Status :: 3 - Alpha" in project["classifiers"]
    assert project["urls"] == EXPECTED_URLS
    assert project["optional-dependencies"]["docs"] == ["mkdocs>=1.6,<2"]

    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include benchmarks " in manifest
    assert all(
        f"include docs/{name}" in manifest
        for name in (
            "index.md",
            "tutorial.md",
            "faq.md",
            "api-minimum.md",
            "evaluation-spec.md",
            "glossary.md",
            "ci-regression.md",
            "benchmark.md",
            "release-notes-0.1.0rc1.md",
        )
    )
    assert "recursive-include notebooks *.ipynb" in manifest
    assert "recursive-include tools *.py" in manifest


def test_runtime_version_matches_distribution_metadata() -> None:
    result = EvaluationRunner.quick_evaluate(
        documents=(Document(id="doc-1", text="retrieval quality"),),
        queries=(
            EvaluationQuery(
                id="query-1",
                query="retrieval",
                relevant_document_ids={"doc-1"},
            ),
        ),
        top_k=(1,),
    )
    runtime = result.manifest["runtime"]
    assert isinstance(runtime, dict)
    assert runtime["retrieval_lab_version"] == "0.1.0rc1"


def test_mkdocs_nav_is_user_facing_only() -> None:
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    nav = config["nav"]
    paths = _nav_paths(nav)
    assert paths == {
        "index.md",
        "tutorial.md",
        "faq.md",
        "ci-regression.md",
        "api-minimum.md",
        "evaluation-spec.md",
        "glossary.md",
        "benchmark.md",
        "release-notes-0.1.0rc1.md",
    }
    assert (
        not {
            "issues-v0.1.md",
            "product-plan.md",
            "technical-design.md",
        }
        & paths
    )


def test_mkdocs_build_excludes_protected_documents(tmp_path: Path) -> None:
    site_dir = tmp_path / "site"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--site-dir",
            str(site_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (site_dir / "index.html").is_file()
    assert (site_dir / "benchmark" / "index.html").is_file()
    generated = tuple(
        path.relative_to(site_dir).as_posix() for path in site_dir.rglob("*")
    )
    assert not any(
        any(
            forbidden in relative
            for forbidden in (
                "issues-v0.1",
                "product-plan",
                "technical-design",
                "release-checklist",
            )
        )
        or relative.startswith("adr/")
        for relative in generated
    )


def test_release_candidate_workflow_is_manual_and_read_only() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "release-candidate.yml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(workflow, dict)
    trigger = workflow.get("on", workflow.get(True))
    assert trigger == {"workflow_dispatch": None}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["install"]["strategy"]["matrix"]["variant"] == [
        "core",
        "dense",
        "docs",
        "all",
    ]
    serialized = json.dumps(workflow, ensure_ascii=False).lower()
    for forbidden in ("pypi", "twine", "github release", "pages deploy", "secrets"):
        assert forbidden not in serialized


def test_clean_build_artifacts_have_expected_content_and_metadata(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--wheel",
            "--outdir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel_paths = tuple(output_dir.glob("*.whl"))
    sdist_paths = tuple(output_dir.glob("*.tar.gz"))
    assert len(wheel_paths) == 1
    assert len(sdist_paths) == 1

    with zipfile.ZipFile(wheel_paths[0]) as wheel:
        wheel_names = tuple(wheel.namelist())
        _assert_safe_paths(wheel_names)
        package_names = tuple(name for name in wheel_names if ".dist-info/" not in name)
        assert package_names
        assert all(name.startswith("retrieval_lab/") for name in package_names)
        assert "retrieval_lab/py.typed" in wheel_names
        assert any(name.endswith("/licenses/LICENSE") for name in wheel_names)
        metadata_name = next(name for name in wheel_names if name.endswith("METADATA"))
        metadata = email.message_from_bytes(wheel.read(metadata_name))
        assert metadata["Name"] == "retrieval-lab-sdk"
        assert metadata["Version"] == "0.1.0rc1"
        description = metadata.get_payload(decode=True)
        assert isinstance(description, bytes)
        description_text = description.decode("utf-8")
        assert "pip install retrieval-lab-sdk" in description_text
        assert 'pip install retrieval-lab"' not in description_text
        assert "CLI under development" not in description_text
        assert "Development Status :: 3 - Alpha" in metadata.get_all("Classifier", [])
        project_urls = dict(
            item.split(", ", maxsplit=1) for item in metadata.get_all("Project-URL", [])
        )
        assert project_urls == EXPECTED_URLS
        entry_points_name = next(
            name for name in wheel_names if name.endswith("entry_points.txt")
        )
        entry_points = wheel.read(entry_points_name).decode("utf-8")
        assert "retrieval-lab = retrieval_lab.cli.app:main" in entry_points
        assert not any(
            name.startswith(("docs/", "examples/", "tests/", "tools/", "notebooks/"))
            for name in package_names
        )

    with tarfile.open(sdist_paths[0], mode="r:gz") as sdist:
        sdist_names = tuple(member.name for member in sdist.getmembers())
        _assert_safe_paths(sdist_names)
        prefix = sdist_names[0].split("/", maxsplit=1)[0]
        for relative in (
            "docs/index.md",
            "docs/benchmark.md",
            "docs/release-notes-0.1.0rc1.md",
            "examples/evaluation.jsonl",
            "notebooks/japanese_bm25_tutorial.ipynb",
            "tools/generate_tutorial_notebook.py",
        ):
            assert f"{prefix}/{relative}" in sdist_names
        assert not any(
            name in sdist_names
            for name in (
                f"{prefix}/README.md",
                f"{prefix}/docs/product-plan.md",
                f"{prefix}/docs/technical-design.md",
                f"{prefix}/docs/issues-v0.1.md",
                f"{prefix}/docs/release-checklist.md",
            )
        )
        assert not any(
            PurePosixPath(name).name.lower() in {".env", "id_rsa", "credentials"}
            for name in sdist_names
        )


def _nav_paths(value: object) -> set[str]:
    if isinstance(value, list):
        paths: set[str] = set()
        for item in value:
            paths.update(_nav_paths(item))
        return paths
    if isinstance(value, dict):
        paths: set[str] = set()
        for item in value.values():
            paths.update(_nav_paths(item))
        return paths
    if isinstance(value, str):
        return {value}
    return set()


def _assert_safe_paths(names: tuple[str, ...]) -> None:
    for name in names:
        path = PurePosixPath(name)
        assert not path.is_absolute()
        assert ".." not in path.parts
