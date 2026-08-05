"""Stable identifier generation for chunks."""

import hashlib
import json


def stable_chunk_id(
    document_id: str,
    start_offset: int,
    end_offset: int,
    text: str,
) -> str:
    """Return a deterministic content-derived identifier for a chunk."""
    payload = json.dumps(
        [document_id, start_offset, end_offset, text],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]
