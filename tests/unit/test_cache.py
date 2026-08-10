"""Safety and reuse tests for the internal JSON artifact cache."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from retrieval_lab import (
    DenseRetriever,
    Document,
    EmbeddingModelMetadata,
    EvaluationQuery,
    EvaluationRunner,
    HybridRetriever,
    KeywordRetriever,
)
from retrieval_lab.artifacts import cache as cache_module
from retrieval_lab.artifacts.cache import (
    CacheStatus,
    canonical_json_bytes,
    chunk_path,
    dense_index_hash,
    dense_index_path,
    read_chunk_artifact,
    read_dense_index_artifact,
    write_chunk_artifact,
    write_dense_index_artifact,
)
from retrieval_lab.exceptions import EvaluationError
from retrieval_lab.models import Chunk


def _chunk(identifier: str = "chunk") -> Chunk:
    return Chunk(
        id=identifier,
        document_id="document",
        text="chunk text",
        start_offset=0,
        end_offset=10,
        metadata={"source": "fixture"},
    )


class _Backend:
    def __init__(self, *, model_id: str = "fake/model", revision: str = "rev") -> None:
        self.metadata = EmbeddingModelMetadata(
            model_id=model_id,
            requested_revision=revision,
            resolved_revision="a" * 40,
        )
        self.calls: list[tuple[str, ...]] = []

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(texts))
        return [[1.0, 0.0] for _ in texts]


def _experiment(
    cache_dir: Path,
    backend: _Backend,
    *,
    retrievers: Sequence[DenseRetriever | HybridRetriever] | None = None,
    prompt: str = "query: ",
    document_text: str = "chunk text",
) -> object:
    dense = DenseRetriever(
        backend=backend,
        query_prompt=prompt,
    )
    selected = dense if retrievers is None else retrievers
    return EvaluationRunner(
        documents=[Document(id="document", text=document_text)],
        queries=[
            EvaluationQuery(
                id="query",
                query="find",
                relevant_document_ids={"document"},
            )
        ],
        retrievers=[selected] if isinstance(selected, DenseRetriever) else selected,
        top_k=[1],
        cache_dir=cache_dir,
    ).run()


def test_chunk_artifact_round_trip_and_generated_safe_path(tmp_path: Path) -> None:
    digest = "a" * 64
    path = write_chunk_artifact(tmp_path, digest, [_chunk()])

    assert path == chunk_path(tmp_path, digest)
    assert path.name == f"{digest}.json"
    result = read_chunk_artifact(tmp_path, digest)
    assert result.status is CacheStatus.HIT
    assert result.payload == (_chunk(),)
    assert "pickle" not in path.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("value", [float("nan"), object()])
def test_canonical_json_rejects_non_json_values(value: object) -> None:
    with pytest.raises(EvaluationError):
        canonical_json_bytes(cast(object, value))  # type: ignore[arg-type]


def test_cache_json_validation_checks_nested_key_types_before_sorting() -> None:
    with pytest.raises(EvaluationError, match="keys must be strings"):
        dense_index_hash(  # type: ignore[arg-type]
            "a" * 64,
            {"nested": {1: "bad", "good": {"also": "valid"}}},
        )


def test_cache_json_validation_wraps_cycles_and_deep_nesting() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(EvaluationError, match=r"deeply nested|canonical JSON"):
        dense_index_hash("a" * 64, cyclic)  # type: ignore[arg-type]

    deep: object = "leaf"
    for _ in range(2_000):
        deep = {"nested": deep}
    with pytest.raises(EvaluationError, match=r"deeply nested|canonical JSON"):
        dense_index_hash("a" * 64, {"deep": deep})  # type: ignore[arg-type]


def test_cache_reader_rejects_artifacts_over_its_byte_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = "a" * 64
    path = chunk_path(tmp_path, digest)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{}")
    monkeypatch.setattr(cache_module, "_MAX_CACHE_BYTES", 1)

    result = read_chunk_artifact(tmp_path, digest)

    assert result.status is CacheStatus.CORRUPT
    assert result.reason is not None and "max_bytes" in result.reason


def test_cache_writer_and_reader_share_the_same_byte_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roundtrip_digest = "d" * 64
    write_chunk_artifact(
        tmp_path,
        roundtrip_digest,
        [_chunk()],
        max_bytes=1024 * 1024,
    )
    assert (
        read_chunk_artifact(
            tmp_path,
            roundtrip_digest,
            max_bytes=1024 * 1024,
        ).status
        is CacheStatus.HIT
    )

    digest = "c" * 64
    monkeypatch.setattr(cache_module, "_MAX_CACHE_BYTES", 1)

    with pytest.raises(EvaluationError, match="max_bytes"):
        write_chunk_artifact(tmp_path, digest, [_chunk()])
    assert not chunk_path(tmp_path, digest).exists()


def test_cache_reader_handles_short_reads_and_sysmax_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = "e" * 64
    write_chunk_artifact(tmp_path, digest, [_chunk()])
    original_read = cache_module.os.read
    calls = 0

    def short_read(descriptor: int, size: int) -> bytes:
        nonlocal calls
        calls += 1
        return original_read(descriptor, min(size, 1))

    monkeypatch.setattr(cache_module.os, "read", short_read)
    result = read_chunk_artifact(tmp_path, digest, max_bytes=sys.maxsize)

    assert result.status is CacheStatus.HIT
    assert calls > 100


def test_cache_reader_detects_growth_after_fstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = "f" * 64
    write_chunk_artifact(tmp_path, digest, [_chunk()])
    original_fstat = cache_module.os.fstat

    def pretend_empty(descriptor: int) -> os.stat_result:
        result = original_fstat(descriptor)
        values = list(result)
        values[6] = 0
        return os.stat_result(values)

    monkeypatch.setattr(cache_module.os, "fstat", pretend_empty)
    result = read_chunk_artifact(tmp_path, digest, max_bytes=1)

    assert result.status is CacheStatus.CORRUPT
    assert result.reason is not None and "max_bytes" in result.reason


def test_cache_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes are unavailable")
    digest = "b" * 64
    path = chunk_path(tmp_path, digest)
    path.parent.mkdir(parents=True)
    try:
        os.mkfifo(path)
    except OSError:
        pytest.skip("named pipes are unavailable")

    result = read_chunk_artifact(tmp_path, digest)

    assert result.status is CacheStatus.CORRUPT


def test_runner_skips_oversize_cache_and_reuses_sufficient_cache(
    tmp_path: Path,
) -> None:
    documents = [Document(id="document", text="chunk text")]
    queries = [
        EvaluationQuery(
            id="query",
            query="chunk",
            relevant_document_ids={"document"},
        )
    ]
    skipped = EvaluationRunner(
        documents=documents,
        queries=queries,
        strategies=["keyword"],
        top_k=[1],
        cache_dir=tmp_path / "tiny",
        cache_max_bytes=1,
    ).run()
    skipped_events = skipped.manifest["runtime"]["cache_events"]  # type: ignore[index]
    skipped_event = skipped_events[0]  # type: ignore[index]
    assert skipped_event["status"] == "skipped"  # type: ignore[index]

    first = EvaluationRunner(
        documents=documents,
        queries=queries,
        strategies=["keyword"],
        top_k=[1],
        cache_dir=tmp_path / "sufficient",
        cache_max_bytes=1024 * 1024,
    ).run()
    second = EvaluationRunner(
        documents=documents,
        queries=queries,
        strategies=["keyword"],
        top_k=[1],
        cache_dir=tmp_path / "sufficient",
        cache_max_bytes=1024 * 1024,
    ).run()
    assert first.run_id == second.run_id
    warm_event = second.manifest["runtime"]["cache_events"][0]  # type: ignore[index]
    assert warm_event["status"] == "hit"  # type: ignore[index]


def test_cache_path_rejects_bytes_root(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="text path"):
        chunk_path(cast(str, bytes(tmp_path)), "a" * 64)


@pytest.mark.parametrize("bad_hash", ["../escape", "A" * 64, "0" * 63, "0" * 65])
def test_cache_paths_reject_non_lowercase_content_hashes(
    tmp_path: Path, bad_hash: str
) -> None:
    with pytest.raises(EvaluationError):
        chunk_path(tmp_path, bad_hash)
    with pytest.raises(EvaluationError):
        dense_index_path(tmp_path, index_hash=bad_hash)


def test_dense_identity_is_hashed_into_a_cross_platform_safe_path(
    tmp_path: Path,
) -> None:
    path = dense_index_path(
        tmp_path,
        index_hash="0" * 64,
        retriever_identity={"name": "../../CON", "model": "a:b\\c"},
    )

    assert "CON" not in path.parts
    assert ".." not in path.parts[len(tmp_path.parts) :]
    assert path.parent.name.isalnum()


def test_atomic_write_replaces_from_the_same_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path]] = []
    original_replace = cache_module.os.replace

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        calls.append((source_path, destination_path))
        original_replace(source_path, destination_path)

    monkeypatch.setattr(cache_module.os, "replace", recording_replace)
    target = write_chunk_artifact(tmp_path, "9" * 64, [_chunk()])

    assert calls == [(calls[0][0], target)]
    assert calls[0][0].parent == target.parent
    assert not calls[0][0].exists()


def test_chunk_cache_distinguishes_missing_corrupt_and_unknown_schema(
    tmp_path: Path,
) -> None:
    digest = "b" * 64
    assert read_chunk_artifact(tmp_path, digest).status is CacheStatus.MISS
    path = chunk_path(tmp_path, digest)
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")
    assert read_chunk_artifact(tmp_path, digest).status is CacheStatus.CORRUPT
    path.write_text(
        json.dumps(
            {
                "artifact_type": "retrieval_lab.chunk",
                "chunk_hash": digest,
                "chunks": [],
                "count": 0,
                "schema_version": 999,
            }
        ),
        encoding="utf-8",
    )
    assert read_chunk_artifact(tmp_path, digest).status is CacheStatus.VERSION_MISMATCH


def test_dense_artifact_round_trip_validates_shared_ids_and_model_identity(
    tmp_path: Path,
) -> None:
    chunks = [_chunk("a"), _chunk("b")]
    settings = {"name": "dense", "model_id": "fake/model", "revision": "r"}
    chunk_hash = "c" * 64
    index_hash = dense_index_hash(chunk_hash, settings)
    identity = {"name": "dense", "settings": settings}
    write_dense_index_artifact(
        tmp_path,
        index_hash=index_hash,
        chunk_hash=chunk_hash,
        retriever_identity=identity,
        chunks=chunks,
        vectors=((1.0, 0.0), (0.0, 1.0)),
        dimension=2,
        model_id="fake/model",
        requested_revision="r",
        resolved_revision="d" * 40,
    )
    result = read_dense_index_artifact(
        tmp_path,
        index_hash=index_hash,
        chunk_hash=chunk_hash,
        retriever_identity=identity,
        chunks=chunks,
        model_id="fake/model",
        requested_revision="r",
    )
    assert result.status is CacheStatus.HIT
    assert cast(dict[str, object], result.payload)["dimension"] == 2

    wrong = read_dense_index_artifact(
        tmp_path,
        index_hash=index_hash,
        chunk_hash=chunk_hash,
        retriever_identity=identity,
        chunks=[_chunk("a")],
        model_id="fake/model",
        requested_revision="r",
    )
    assert wrong.status is CacheStatus.CORRUPT
    revision_mismatch = read_dense_index_artifact(
        tmp_path,
        index_hash=index_hash,
        chunk_hash=chunk_hash,
        retriever_identity=identity,
        chunks=chunks,
        model_id="fake/model",
        requested_revision="r",
        expected_resolved_revision="e" * 40,
    )
    assert revision_mismatch.status is CacheStatus.CORRUPT


def test_dense_reader_rejects_bad_schema_fields_and_values(tmp_path: Path) -> None:
    chunks = [_chunk("a")]
    settings = {"name": "dense"}
    chunk_hash = "f" * 64
    index_hash = dense_index_hash(chunk_hash, settings)
    identity = {"name": "dense"}
    path = write_dense_index_artifact(
        tmp_path,
        index_hash=index_hash,
        chunk_hash=chunk_hash,
        retriever_identity=identity,
        chunks=chunks,
        vectors=((1.0, 0.0),),
        dimension=2,
        model_id="model",
        requested_revision=None,
        resolved_revision=None,
    )
    original = json.loads(path.read_text(encoding="utf-8"))
    mutations = [
        {"extra": 1},
        {"artifact_type": "other"},
        {"chunk_hash": "0" * 64},
        {"implementation_version": "old"},
        {"dtype": "float32"},
        {"retriever_identity": {"other": True}},
        {"model": []},
        {"model": {"model_id": "model"}},
        {
            "model": {
                "model_id": "other",
                "requested_revision": None,
                "resolved_revision": None,
            }
        },
        {
            "model": {
                "model_id": "model",
                "requested_revision": None,
                "resolved_revision": " ",
            }
        },
        {"shape": [2, 2]},
        {"vectors": [[0.5, 0.5]]},
        {"vectors": [[0.0, 0.0]]},
    ]
    for mutation in mutations:
        payload = dict(original)
        payload.update(mutation)
        path.write_text(json.dumps(payload), encoding="utf-8")
        read = read_dense_index_artifact(
            tmp_path,
            index_hash=index_hash,
            chunk_hash=chunk_hash,
            retriever_identity=identity,
            chunks=chunks,
            model_id="model",
            requested_revision=None,
        )
        assert read.status in {CacheStatus.CORRUPT, CacheStatus.VERSION_MISMATCH}


@pytest.mark.parametrize(
    ("vectors", "dimension"),
    [(((0.0, 0.0),), 2), (((1.0,),), 2), (((float("nan"), 1.0),), 2)],
)
def test_dense_artifact_writer_rejects_unsafe_vectors(
    tmp_path: Path,
    vectors: Sequence[Sequence[float]],
    dimension: int,
) -> None:
    with pytest.raises(EvaluationError):
        write_dense_index_artifact(
            tmp_path,
            index_hash="d" * 64,
            chunk_hash="e" * 64,
            retriever_identity={"name": "dense"},
            chunks=[_chunk()],
            vectors=vectors,
            dimension=dimension,
            model_id="model",
            requested_revision=None,
            resolved_revision=None,
        )


def test_runner_dense_cache_hit_skips_document_encode_and_keeps_run_id(
    tmp_path: Path,
) -> None:
    first_backend = _Backend()
    first = _experiment(tmp_path, first_backend)
    second_backend = _Backend()
    second = _experiment(tmp_path, second_backend)

    assert len(first_backend.calls) == 2
    assert len(second_backend.calls) == 1  # query encoding only
    assert first.run_id == second.run_id  # type: ignore[union-attr]
    assert second.manifest["runtime"]["cache_events"][1]["status"] == "hit"  # type: ignore[index,union-attr]
    assert str(tmp_path) not in second.to_json()  # type: ignore[union-attr]


@pytest.mark.parametrize("change", ["corpus", "prompt", "revision"])
def test_dense_cache_key_changes_when_reproducibility_inputs_change(
    tmp_path: Path,
    change: str,
) -> None:
    _experiment(tmp_path, _Backend())
    backend = _Backend(revision="next" if change == "revision" else "rev")
    _experiment(
        tmp_path,
        backend,
        prompt="question: " if change == "prompt" else "query: ",
        document_text="chunk text changed" if change == "corpus" else "chunk text",
    )

    assert len(backend.calls) == 2


def test_chunk_cache_is_usable_without_dense_retrieval(tmp_path: Path) -> None:
    def run() -> object:
        return EvaluationRunner(
            documents=[Document(id="document", text="chunk text")],
            queries=[
                EvaluationQuery(
                    id="query",
                    query="chunk",
                    relevant_document_ids={"document"},
                )
            ],
            retrievers=[KeywordRetriever()],
            top_k=[1],
            cache_dir=tmp_path,
        ).run()

    first = run()
    second = run()

    assert first.run_id == second.run_id  # type: ignore[union-attr]
    events = second.manifest["runtime"]["cache_events"]  # type: ignore[index,union-attr]
    assert events[0]["artifact"] == "chunks"  # type: ignore[index]
    assert events[0]["status"] == "hit"  # type: ignore[index]
    assert events[0]["duration_ms"] >= 0.0  # type: ignore[index]


def test_runner_corrupt_dense_cache_rebuilds_with_explicit_status(
    tmp_path: Path,
) -> None:
    _experiment(tmp_path, _Backend())
    index_file = next((tmp_path / "indexes").glob("*/*.json"))
    index_file.write_text("{}", encoding="utf-8")

    backend = _Backend()
    result = _experiment(tmp_path, backend)
    assert len(backend.calls) == 2
    event = result.manifest["runtime"]["cache_events"][1]  # type: ignore[index]
    assert event["status"] == "version_mismatch"


def test_shared_dense_instance_inside_hybrid_is_encoded_once(tmp_path: Path) -> None:
    backend = _Backend()
    dense = DenseRetriever(backend=backend)
    hybrid = HybridRetriever([dense, KeywordRetriever()], candidate_k=2)
    # Both top-level and Hybrid use the same first Dense object.
    result = EvaluationRunner(
        documents=[Document(id="document", text="chunk text")],
        queries=[
            EvaluationQuery(
                id="query",
                query="find",
                relevant_document_ids={"document"},
            )
        ],
        retrievers=[dense, hybrid],
        top_k=[1],
        cache_dir=tmp_path,
    ).run()
    assert result.run_id
    assert backend.calls.count(("passage: chunk text",)) == 1


def test_hybrid_build_time_includes_cached_dense_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter(
        [
            0,
            1_000_000,  # chunk cache
            2_000_000,
            7_000_000,  # Dense cache preparation
            8_000_000,
            10_000_000,  # Hybrid index call
            11_000_000,
            12_000_000,  # query search
        ]
    )
    monkeypatch.setattr("retrieval_lab.runner.perf_counter_ns", lambda: next(clock))
    dense = DenseRetriever(backend=_Backend())
    hybrid = HybridRetriever([dense, KeywordRetriever()], candidate_k=2)

    result = EvaluationRunner(
        documents=[Document(id="document", text="chunk text")],
        queries=[
            EvaluationQuery(
                id="query",
                query="find",
                relevant_document_ids={"document"},
            )
        ],
        retrievers=[hybrid],
        top_k=[1],
        cache_dir=tmp_path,
    ).run()

    runtime = result.manifest["runtime"]
    assert runtime["build_ms"] == {"hybrid": 8.0}  # type: ignore[index]


def test_runtime_manifest_never_serializes_cache_path_or_environment_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "absolute-secret-cache-path"
    monkeypatch.setenv("RETRIEVAL_LAB_TEST_SECRET", "do-not-record-this-value")

    result = EvaluationRunner(
        documents=[Document(id="document", text="chunk text")],
        queries=[
            EvaluationQuery(
                id="query",
                query="find",
                relevant_document_ids={"document"},
            )
        ],
        retrievers=[KeywordRetriever()],
        top_k=[1],
        cache_dir=cache_dir,
    ).run()

    serialized = result.to_json()
    assert str(cache_dir) not in serialized
    assert "absolute-secret-cache-path" not in serialized
    assert "RETRIEVAL_LAB_TEST_SECRET" not in serialized
    assert "do-not-record-this-value" not in serialized
