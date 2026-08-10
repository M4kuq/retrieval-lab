"""Shared redaction for report-visible identifiers."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath, PureWindowsPath


def safe_identifier(value: str) -> str:
    """Hide absolute paths with deterministic, non-reversible identifiers.

    The digest keeps distinct paths distinct without putting any part of the
    original path in a report.  Keeping the historical ``[redacted path]``
    prefix also makes the redaction obvious to humans while the digest avoids
    collisions between unrelated paths.
    """

    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return f"[redacted path]#{digest}"
    return value


__all__ = ["safe_identifier"]
