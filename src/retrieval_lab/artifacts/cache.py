"""Safe, content-addressed JSON artifacts used by the evaluation runner.

The cache is deliberately boring: all data is canonical JSON, every path
component is generated from validated hexadecimal hashes, and writes are
atomic.  A cache read is advisory; malformed or incompatible data is reported
to the caller and is never returned as an artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from retrieval_lab.domain import Chunk, JSONValue
from retrieval_lab.exceptions import EvaluationError

CHUNK_ARTIFACT_TYPE = "retrieval_lab.chunk"
DENSE_INDEX_ARTIFACT_TYPE = "retrieval_lab.dense_index"
CHUNK_SCHEMA_VERSION = 1
DENSE_INDEX_SCHEMA_VERSION = 1
DENSE_IMPLEMENTATION_VERSION = "dense-index-v1"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
CACHE_MAX_BYTES = 64 * 1024 * 1024
_MAX_CACHE_BYTES = CACHE_MAX_BYTES
_READ_CHUNK_BYTES = 64 * 1024


class CacheCapacityError(EvaluationError):
    """Raised when a cache artifact exceeds the configured write capacity."""


class CacheStatus(StrEnum):
    """Outcome of reading one cache artifact."""

    HIT = "hit"
    MISS = "miss"
    CORRUPT = "corrupt"
    VERSION_MISMATCH = "version_mismatch"


@dataclass(frozen=True)
class CacheRead:
    """A typed cache read outcome, with no invalid payload on failure."""

    status: CacheStatus
    payload: object | None = None
    reason: str | None = None


def canonical_json_bytes(value: JSONValue) -> bytes:
    """Serialize a JSON value deterministically and reject non-finite floats."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError) as exc:
        raise EvaluationError("cache artifact is not canonical JSON") from exc


def _validate_hash(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise EvaluationError(
            f"{field_name} must be a lowercase 64-character hexadecimal hash"
        )
    return value


def _hash_json(value: JSONValue) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _cache_root(cache_dir: str | os.PathLike[str]) -> Path:
    if isinstance(cache_dir, bytes):
        raise EvaluationError("cache_dir must be a text path")
    if isinstance(cache_dir, str) and not cache_dir.strip():
        raise EvaluationError("cache_dir must not be empty")
    try:
        root = Path(cache_dir)
    except (TypeError, ValueError) as exc:
        raise EvaluationError("cache_dir must be a valid path") from exc
    if not str(root):
        raise EvaluationError("cache_dir must not be empty")
    return root


def chunk_path(cache_dir: str | os.PathLike[str], chunk_hash: str) -> Path:
    """Return the generated path for a chunk artifact."""

    valid_hash = _validate_hash(chunk_hash, field_name="chunk_hash")
    return _cache_root(cache_dir) / "chunks" / f"{valid_hash}.json"


def dense_index_hash(
    chunk_hash: str,
    settings: Mapping[str, JSONValue],
) -> str:
    """Compute a dense index key from chunk identity and normalized settings."""

    valid_chunk_hash = _validate_hash(chunk_hash, field_name="chunk_hash")
    normalized = _normalize_json(settings, location="settings")
    return _hash_json(
        {
            "chunk_hash": valid_chunk_hash,
            "implementation_version": DENSE_IMPLEMENTATION_VERSION,
            "schema_version": DENSE_INDEX_SCHEMA_VERSION,
            "settings": normalized,
        }
    )


def dense_index_path(
    cache_dir: str | os.PathLike[str],
    *,
    index_hash: str,
    retriever_identity: Mapping[str, JSONValue] | None = None,
) -> Path:
    """Return a path whose user-controlled identity is hashed, never raw."""

    valid_index_hash = _validate_hash(index_hash, field_name="index_hash")
    identity = (
        {}
        if retriever_identity is None
        else _normalize_json(retriever_identity, location="retriever_identity")
    )
    identity_hash = _hash_json(identity)
    return (
        _cache_root(cache_dir) / "indexes" / identity_hash / f"{valid_index_hash}.json"
    )


def _cache_max_bytes(value: int | None) -> int:
    limit = _MAX_CACHE_BYTES if value is None else value
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise EvaluationError("cache max_bytes must be a positive integer")
    return limit


def _atomic_write(
    path: Path,
    payload: JSONValue,
    *,
    max_bytes: int | None = None,
) -> None:
    """Write canonical bytes with fsync + same-directory replacement."""

    limit = _cache_max_bytes(max_bytes)
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            encoder = json.JSONEncoder(
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            written = 0
            for fragment in encoder.iterencode(payload):
                data = fragment.encode("utf-8")
                written += len(data)
                if written + 1 > limit:
                    raise CacheCapacityError(
                        f"cache artifact exceeds max_bytes ({limit})"
                    )
                stream.write(data)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError) as exc:
        raise EvaluationError("cache artifact is not canonical JSON") from exc
    except OSError as exc:
        raise EvaluationError(
            f"could not atomically write cache artifact {path}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


class _CacheReadLimitError(ValueError):
    pass


def _read_bytes(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("cache artifact is not a regular file")
        if metadata.st_size > limit:
            raise _CacheReadLimitError(f"cache artifact exceeds max_bytes ({limit})")
        data = bytearray()
        while True:
            request = min(_READ_CHUNK_BYTES, limit + 1 - len(data))
            if request <= 0:
                raise _CacheReadLimitError(
                    f"cache artifact exceeds max_bytes ({limit})"
                )
            block = os.read(descriptor, request)
            if not block:
                return bytes(data)
            data.extend(block)
            if len(data) > limit:
                raise _CacheReadLimitError(
                    f"cache artifact exceeds max_bytes ({limit})"
                )
    finally:
        os.close(descriptor)


def _read_json(path: Path, *, max_bytes: int | None = None) -> CacheRead:
    limit = _cache_max_bytes(max_bytes)
    if not path.exists():
        return CacheRead(CacheStatus.MISS, reason="artifact does not exist")
    try:
        raw = _read_bytes(path, limit)
        value = json.loads(raw.decode("utf-8"))
    except _CacheReadLimitError as exc:
        return CacheRead(CacheStatus.CORRUPT, reason=str(exc))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        return CacheRead(CacheStatus.CORRUPT, reason=f"invalid JSON: {exc}")
    if not isinstance(value, dict):
        return CacheRead(CacheStatus.CORRUPT, reason="artifact root must be an object")
    return CacheRead(CacheStatus.HIT, payload=value)


def _normalize_json(value: object, *, location: str) -> JSONValue:
    try:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise EvaluationError(f"{location} must not contain NaN or infinity")
            return value
        if isinstance(value, Mapping):
            result: dict[str, JSONValue] = {}
            keys = tuple(value)
            if any(not isinstance(key, str) for key in keys):
                raise EvaluationError(f"{location} keys must be strings")
            for key in sorted(cast(tuple[str, ...], keys)):
                result[key] = _normalize_json(value[key], location=f"{location}.{key}")
            return result
        if isinstance(value, (list, tuple)):
            return [
                _normalize_json(item, location=f"{location}[{index}]")
                for index, item in enumerate(value)
            ]
        raise EvaluationError(
            f"{location} contains unsupported JSON value type {type(value).__name__}"
        )
    except RecursionError as exc:
        raise EvaluationError("cache JSON is cyclic or too deeply nested") from exc


def _chunk_record(chunk: Chunk) -> dict[str, JSONValue]:
    return {
        "document_id": chunk.document_id,
        "end_offset": chunk.end_offset,
        "id": chunk.id,
        "metadata": _normalize_json(chunk.metadata, location="chunk.metadata"),
        "start_offset": chunk.start_offset,
        "text": chunk.text,
    }


def _parse_chunk_records(value: object) -> tuple[Chunk, ...]:
    if not isinstance(value, list):
        raise ValueError("chunks must be a list")
    records: list[Chunk] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"chunks[{index}] must be an object")
        required = {
            "document_id",
            "end_offset",
            "id",
            "metadata",
            "start_offset",
            "text",
        }
        if set(raw) != required:
            raise ValueError(f"chunks[{index}] has invalid fields")
        try:
            chunk = Chunk(
                id=raw["id"],
                document_id=raw["document_id"],
                text=raw["text"],
                start_offset=raw["start_offset"],
                end_offset=raw["end_offset"],
                metadata=raw["metadata"],
            )
        except Exception as exc:
            raise ValueError(f"chunks[{index}] is invalid") from exc
        if chunk.id in seen:
            raise ValueError(f"duplicate chunk id {chunk.id!r}")
        seen.add(chunk.id)
        records.append(chunk)
    return tuple(records)


def write_chunk_artifact(
    cache_dir: str | os.PathLike[str],
    chunk_hash: str,
    chunks: Sequence[Chunk],
    *,
    max_bytes: int | None = None,
) -> Path:
    """Persist validated chunks as a schema-versioned JSON artifact."""

    valid_hash = _validate_hash(chunk_hash, field_name="chunk_hash")
    try:
        normalized_chunks = tuple(chunks)
    except TypeError as exc:
        raise EvaluationError("chunks must be a sequence") from exc
    if not all(isinstance(chunk, Chunk) for chunk in normalized_chunks):
        raise EvaluationError("chunks must contain only Chunk records")
    path = chunk_path(cache_dir, valid_hash)
    records: list[JSONValue] = [_chunk_record(chunk) for chunk in normalized_chunks]
    payload: dict[str, JSONValue] = {
        "artifact_type": CHUNK_ARTIFACT_TYPE,
        "chunk_hash": valid_hash,
        "chunks": records,
        "count": len(normalized_chunks),
        "content_hash": _hash_json(records),
        "schema_version": CHUNK_SCHEMA_VERSION,
    }
    _atomic_write(path, payload, max_bytes=max_bytes)
    return path


def read_chunk_artifact(
    cache_dir: str | os.PathLike[str],
    chunk_hash: str,
    *,
    max_bytes: int | None = None,
) -> CacheRead:
    """Read and fully validate a chunk artifact."""

    valid_hash = _validate_hash(chunk_hash, field_name="chunk_hash")
    result = _read_json(chunk_path(cache_dir, valid_hash), max_bytes=max_bytes)
    if result.status is not CacheStatus.HIT:
        return result
    assert isinstance(result.payload, dict)
    if result.payload.get("schema_version") != CHUNK_SCHEMA_VERSION:
        return CacheRead(CacheStatus.VERSION_MISMATCH, reason="unknown chunk schema")
    if result.payload.get("artifact_type") != CHUNK_ARTIFACT_TYPE:
        return CacheRead(CacheStatus.CORRUPT, reason="unexpected chunk artifact type")
    if result.payload.get("chunk_hash") != valid_hash:
        return CacheRead(CacheStatus.CORRUPT, reason="chunk hash mismatch")
    try:
        raw_chunks = result.payload.get("chunks")
        chunks = _parse_chunk_records(raw_chunks)
        if result.payload.get("count") != len(chunks):
            raise ValueError("chunk count mismatch")
        if result.payload.get("content_hash") != _hash_json(
            [_chunk_record(chunk) for chunk in chunks]
        ):
            raise ValueError("chunk content hash mismatch")
    except (TypeError, ValueError) as exc:
        return CacheRead(CacheStatus.CORRUPT, reason=str(exc))
    return CacheRead(CacheStatus.HIT, payload=chunks)


def write_dense_index_artifact(
    cache_dir: str | os.PathLike[str],
    *,
    index_hash: str,
    chunk_hash: str,
    retriever_identity: Mapping[str, JSONValue],
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    dimension: int,
    model_id: str,
    requested_revision: str | None,
    resolved_revision: str | None,
    max_bytes: int | None = None,
) -> Path:
    """Persist a validated dense index without serializing executable data."""

    valid_index_hash = _validate_hash(index_hash, field_name="index_hash")
    valid_chunk_hash = _validate_hash(chunk_hash, field_name="chunk_hash")
    if not isinstance(model_id, str) or not model_id.strip():
        raise EvaluationError("model_id must be a non-empty string")
    if requested_revision is not None and not isinstance(requested_revision, str):
        raise EvaluationError("requested_revision must be a string or None")
    if resolved_revision is not None and not isinstance(resolved_revision, str):
        raise EvaluationError("resolved_revision must be a string or None")
    chunk_ids = _validate_chunk_ids(chunks)
    matrix = _validate_vectors(vectors, count=len(chunk_ids), dimension=dimension)
    normalized_identity = _normalize_json(
        retriever_identity, location="retriever_identity"
    )
    path = dense_index_path(
        cache_dir,
        index_hash=valid_index_hash,
        retriever_identity=cast(Mapping[str, JSONValue], normalized_identity),
    )
    payload: dict[str, JSONValue] = {
        "artifact_type": DENSE_INDEX_ARTIFACT_TYPE,
        "chunk_hash": valid_chunk_hash,
        "chunk_ids": list(chunk_ids),
        "count": len(chunk_ids),
        "dimension": dimension,
        "dtype": "float64",
        "implementation_version": DENSE_IMPLEMENTATION_VERSION,
        "index_hash": valid_index_hash,
        "model": {
            "model_id": model_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
        },
        "retriever_identity": normalized_identity,
        "schema_version": DENSE_INDEX_SCHEMA_VERSION,
        "shape": [len(chunk_ids), dimension],
        "vectors_hash": _hash_json(
            {
                "chunk_ids": list(chunk_ids),
                "dimension": dimension,
                "dtype": "float64",
                "model": {
                    "model_id": model_id,
                    "requested_revision": requested_revision,
                    "resolved_revision": resolved_revision,
                },
                "vectors": [list(row) for row in matrix],
            }
        ),
        "vectors": [list(row) for row in matrix],
    }
    _atomic_write(path, payload, max_bytes=max_bytes)
    return path


def _validate_chunk_ids(chunks: Sequence[Chunk]) -> tuple[str, ...]:
    normalized = tuple(chunks)
    if not all(isinstance(chunk, Chunk) for chunk in normalized):
        raise EvaluationError("chunks must contain only Chunk records")
    ids = tuple(chunk.id for chunk in normalized)
    if len(set(ids)) != len(ids):
        raise EvaluationError("chunks must have unique IDs")
    return ids


def _validate_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    count: int,
    dimension: int,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise EvaluationError("dimension must be a positive integer")
    matrix = tuple(tuple(row) for row in vectors)
    if len(matrix) != count:
        raise EvaluationError("vector count does not match chunk count")
    for row in matrix:
        if len(row) != dimension:
            raise EvaluationError("vector dimension does not match shape")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in row
        ):
            raise EvaluationError("vectors must contain numeric values")
        values = tuple(float(value) for value in row)
        if any(not math.isfinite(value) for value in values):
            raise EvaluationError("vectors must contain finite values")
        if math.hypot(*values) == 0.0:
            raise EvaluationError("vectors must not contain zero vectors")
    return tuple(tuple(float(value) for value in row) for row in matrix)


def read_dense_index_artifact(
    cache_dir: str | os.PathLike[str],
    *,
    index_hash: str,
    chunk_hash: str,
    retriever_identity: Mapping[str, JSONValue],
    chunks: Sequence[Chunk],
    model_id: str,
    requested_revision: str | None,
    expected_resolved_revision: str | None = None,
    max_bytes: int | None = None,
) -> CacheRead:
    """Read a dense index and validate identity, shape, IDs and all values."""

    valid_index_hash = _validate_hash(index_hash, field_name="index_hash")
    valid_chunk_hash = _validate_hash(chunk_hash, field_name="chunk_hash")
    normalized_identity = _normalize_json(
        retriever_identity, location="retriever_identity"
    )
    path = dense_index_path(
        cache_dir,
        index_hash=valid_index_hash,
        retriever_identity=cast(Mapping[str, JSONValue], normalized_identity),
    )
    result = _read_json(path, max_bytes=max_bytes)
    if result.status is not CacheStatus.HIT:
        return result
    assert isinstance(result.payload, dict)
    payload = result.payload
    if payload.get("schema_version") != DENSE_INDEX_SCHEMA_VERSION:
        return CacheRead(CacheStatus.VERSION_MISMATCH, reason="unknown dense schema")
    required = {
        "artifact_type",
        "chunk_hash",
        "chunk_ids",
        "count",
        "dimension",
        "dtype",
        "implementation_version",
        "index_hash",
        "model",
        "retriever_identity",
        "schema_version",
        "shape",
        "vectors",
        "vectors_hash",
    }
    if set(payload) != required:
        return CacheRead(
            CacheStatus.CORRUPT, reason="dense artifact fields are invalid"
        )
    if payload["artifact_type"] != DENSE_INDEX_ARTIFACT_TYPE:
        return CacheRead(CacheStatus.CORRUPT, reason="unexpected dense artifact type")
    if (
        payload["chunk_hash"] != valid_chunk_hash
        or payload["index_hash"] != valid_index_hash
    ):
        return CacheRead(CacheStatus.CORRUPT, reason="dense artifact hash mismatch")
    if payload["implementation_version"] != DENSE_IMPLEMENTATION_VERSION:
        return CacheRead(
            CacheStatus.VERSION_MISMATCH, reason="unknown dense implementation"
        )
    if payload["dtype"] != "float64":
        return CacheRead(CacheStatus.CORRUPT, reason="unsupported vector dtype")
    if payload["retriever_identity"] != normalized_identity:
        return CacheRead(CacheStatus.CORRUPT, reason="retriever identity mismatch")
    model = payload["model"]
    if not isinstance(model, dict):
        return CacheRead(CacheStatus.CORRUPT, reason="model identity is invalid")
    if set(model) != {"model_id", "requested_revision", "resolved_revision"}:
        return CacheRead(
            CacheStatus.CORRUPT, reason="model identity fields are invalid"
        )
    if (
        model["model_id"] != model_id
        or model["requested_revision"] != requested_revision
    ):
        return CacheRead(CacheStatus.CORRUPT, reason="model identity mismatch")
    resolved_revision = model["resolved_revision"]
    if resolved_revision is not None and (
        not isinstance(resolved_revision, str) or not resolved_revision.strip()
    ):
        return CacheRead(CacheStatus.CORRUPT, reason="resolved revision is invalid")
    if (
        expected_resolved_revision is not None
        and expected_resolved_revision != model["resolved_revision"]
    ):
        return CacheRead(CacheStatus.CORRUPT, reason="resolved revision mismatch")
    try:
        count = payload["count"]
        dimension = payload["dimension"]
        shape = payload["shape"]
        chunk_ids = payload["chunk_ids"]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or count < 0
            or dimension <= 0
            or shape != [count, dimension]
            or not isinstance(chunk_ids, list)
            or tuple(chunk_ids) != _validate_chunk_ids(chunks)
        ):
            raise ValueError("dense shape or chunk IDs are invalid")
        matrix = _validate_vectors(payload["vectors"], count=count, dimension=dimension)
        expected_vectors_hash = _hash_json(
            {
                "chunk_ids": list(chunk_ids),
                "dimension": dimension,
                "dtype": payload["dtype"],
                "model": model,
                "vectors": [list(row) for row in matrix],
            }
        )
        if payload["vectors_hash"] != expected_vectors_hash:
            raise ValueError("dense vector content hash mismatch")
    except (TypeError, ValueError, EvaluationError) as exc:
        return CacheRead(CacheStatus.CORRUPT, reason=str(exc))
    return CacheRead(
        CacheStatus.HIT,
        payload={
            "vectors": matrix,
            "dimension": dimension,
            "resolved_revision": resolved_revision,
        },
    )


__all__ = [
    "CACHE_MAX_BYTES",
    "CHUNK_ARTIFACT_TYPE",
    "CHUNK_SCHEMA_VERSION",
    "DENSE_INDEX_ARTIFACT_TYPE",
    "DENSE_INDEX_SCHEMA_VERSION",
    "CacheCapacityError",
    "CacheRead",
    "CacheStatus",
    "canonical_json_bytes",
    "chunk_path",
    "dense_index_hash",
    "dense_index_path",
    "read_chunk_artifact",
    "read_dense_index_artifact",
    "write_chunk_artifact",
    "write_dense_index_artifact",
]
