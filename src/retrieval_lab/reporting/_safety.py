"""Shared redaction for report-visible identifiers."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath


def safe_identifier(value: str) -> str:
    """Hide identifiers that are absolute POSIX or Windows paths."""

    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return "[redacted path]"
    return value


__all__ = ["safe_identifier"]
