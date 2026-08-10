"""Shared validation helpers for domain records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import NoReturn, TypeVar

from retrieval_lab.exceptions import RetrievalLabError

from .json_types import JSONValue

_ErrorT = TypeVar("_ErrorT", bound=RetrievalLabError)


def fail(error_type: type[_ErrorT], message: str) -> NoReturn:
    """Raise a package exception without exposing an implementation exception."""

    raise error_type(message)


def require_non_empty_string(
    value: object,
    *,
    field_name: str,
    error_type: type[_ErrorT],
) -> str:
    """Return a string after validating that it contains visible content."""

    if not isinstance(value, str) or not value.strip():
        fail(error_type, f"{field_name} must be a non-empty string")
    return value


def require_positive_int(
    value: object,
    *,
    field_name: str,
    error_type: type[_ErrorT],
) -> int:
    """Return a positive integer, rejecting booleans explicitly."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(error_type, f"{field_name} must be a positive integer")
    return value


def require_non_negative_int(
    value: object,
    *,
    field_name: str,
    error_type: type[_ErrorT],
) -> int:
    """Return a non-negative integer, rejecting booleans explicitly."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(error_type, f"{field_name} must be a non-negative integer")
    return value


def require_finite_float(
    value: object,
    *,
    field_name: str,
    error_type: type[_ErrorT],
) -> float:
    """Return a finite numeric value as ``float``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(error_type, f"{field_name} must be a finite number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise error_type(f"{field_name} must be a finite number") from exc
    if not math.isfinite(normalized):
        fail(error_type, f"{field_name} must be a finite number")
    return normalized


def normalize_json_mapping(
    value: object,
    *,
    field_name: str,
    error_type: type[_ErrorT],
) -> Mapping[str, JSONValue]:
    """Validate and defensively copy a mapping containing JSON values."""

    if not isinstance(value, Mapping):
        fail(error_type, f"{field_name} must be a mapping with string keys")

    normalized: dict[str, JSONValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            fail(error_type, f"{field_name} keys must be strings")
        normalized[key] = _normalize_json_value(
            item,
            location=f"{field_name}.{key}",
            error_type=error_type,
        )
    return MappingProxyType(normalized)


def _normalize_json_value(
    value: object,
    *,
    location: str,
    error_type: type[_ErrorT],
) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            fail(error_type, f"{location} must not contain NaN or infinity")
        return value
    if isinstance(value, list):
        return [
            _normalize_json_value(
                item,
                location=f"{location}[{index}]",
                error_type=error_type,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                fail(error_type, f"{location} keys must be strings")
            result[key] = _normalize_json_value(
                item,
                location=f"{location}.{key}",
                error_type=error_type,
            )
        return result
    fail(
        error_type,
        f"{location} contains unsupported JSON value type {type(value).__name__}",
    )
