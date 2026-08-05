"""JSON-compatible types used by public Retrieval Lab records."""

from typing import TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

__all__ = ["JSONScalar", "JSONValue"]
