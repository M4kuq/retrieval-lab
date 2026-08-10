"""Internal, safe artifact persistence helpers.

This package is intentionally not re-exported from :mod:`retrieval_lab`.
Artifacts contain only canonical JSON and are never deserialized as Python
objects.
"""

from retrieval_lab.artifacts.cache import (
    CacheRead,
    CacheStatus,
    dense_index_hash,
    dense_index_path,
    read_chunk_artifact,
    read_dense_index_artifact,
    write_chunk_artifact,
    write_dense_index_artifact,
)

__all__ = [
    "CacheRead",
    "CacheStatus",
    "dense_index_hash",
    "dense_index_path",
    "read_chunk_artifact",
    "read_dense_index_artifact",
    "write_chunk_artifact",
    "write_dense_index_artifact",
]
