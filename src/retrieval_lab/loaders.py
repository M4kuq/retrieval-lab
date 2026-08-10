"""Local corpus loaders for UTF-8 text, Markdown, and native JSONL."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from pathlib import Path
from typing import cast

from retrieval_lab.domain import Document
from retrieval_lab.domain._validation import normalize_json_mapping
from retrieval_lab.exceptions import CorpusValidationError

_SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".jsonl"})


def load_documents(
    path: str | Path,
    *,
    include: Sequence[str] = (),
) -> tuple[Document, ...]:
    """Load supported local documents in deterministic source-path order."""

    source = _coerce_path(path)
    files, root = _discover_files(source, include=include)
    documents: list[Document] = []
    identifier_locations: dict[str, str] = {}

    for file_path in files:
        relative_source = _relative_source(file_path, root)
        if file_path.suffix.lower() == ".jsonl":
            loaded = _load_jsonl(file_path, source=relative_source)
        else:
            loaded = (_load_text(file_path, source=relative_source),)
        for document in loaded:
            location = (
                document.source if document.source is not None else relative_source
            )
            if document.id in identifier_locations:
                raise CorpusValidationError(
                    f"duplicate document ID {document.id!r} at {location}; first "
                    f"seen at {identifier_locations[document.id]}"
                )
            identifier_locations[document.id] = location
            documents.append(document)

    if not documents:
        raise CorpusValidationError(f"corpus contains no documents: {source}")
    return tuple(documents)


def _coerce_path(path: str | Path) -> Path:
    try:
        source = Path(path)
    except TypeError as exc:
        raise CorpusValidationError("corpus path must be a string or Path") from exc
    try:
        exists = source.exists()
    except OSError as exc:
        raise CorpusValidationError(
            f"cannot inspect corpus path {source}: {exc}"
        ) from exc
    if not exists:
        raise CorpusValidationError(f"corpus path does not exist: {source}")
    return source


def _discover_files(
    source: Path,
    *,
    include: Sequence[str] = (),
) -> tuple[tuple[Path, ...], Path]:
    if isinstance(include, (str, bytes)) or not isinstance(include, Sequence):
        raise CorpusValidationError("corpus include must be a sequence of glob strings")
    patterns = tuple(include)
    if any(not isinstance(pattern, str) or not pattern.strip() for pattern in patterns):
        raise CorpusValidationError(
            "corpus include must contain non-empty glob strings"
        )
    try:
        if source.is_file():
            if source.suffix.lower() not in _SUPPORTED_SUFFIXES:
                raise CorpusValidationError(
                    f"unsupported corpus file type {source.suffix or '<none>'!r}: "
                    f"{source}; expected one of {sorted(_SUPPORTED_SUFFIXES)}"
                )
            relative = unicodedata.normalize("NFC", source.name)
            selected = (
                (source,)
                if not patterns or _matches_include(relative, patterns)
                else ()
            )
            if patterns and not selected:
                raise CorpusValidationError(
                    "corpus include matched no documents; adjust the configured globs"
                )
            return selected, source.parent
        if not source.is_dir():
            raise CorpusValidationError(
                f"corpus path must be a regular file or directory: {source}"
            )
        candidates = tuple(
            file_path
            for file_path in source.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in _SUPPORTED_SUFFIXES
        )
    except CorpusValidationError:
        raise
    except OSError as exc:
        raise CorpusValidationError(f"cannot inspect corpus {source}: {exc}") from exc

    files = tuple(
        sorted(
            (
                item
                for item in candidates
                if not patterns
                or _matches_include(
                    unicodedata.normalize("NFC", item.relative_to(source).as_posix()),
                    patterns,
                )
            ),
            key=lambda item: unicodedata.normalize(
                "NFC", item.relative_to(source).as_posix()
            ),
        )
    )
    if not files:
        if patterns and candidates:
            raise CorpusValidationError(
                "corpus include matched no documents; adjust the configured globs"
            )
        raise CorpusValidationError(
            f"corpus directory has no supported files: {source}; expected "
            f"{sorted(_SUPPORTED_SUFFIXES)}"
        )
    return files, source


def _matches_include(source: str, patterns: Sequence[str]) -> bool:
    normalized_source = source.replace("\\", "/")
    for raw_pattern in patterns:
        pattern = raw_pattern.replace("\\", "/")
        if fnmatchcase(normalized_source, pattern):
            return True
        if pattern.startswith("**/") and fnmatchcase(
            normalized_source, pattern.removeprefix("**/")
        ):
            return True
    return False


def _relative_source(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise CorpusValidationError(
            f"document path {path} is outside corpus root {root}"
        ) from exc
    return unicodedata.normalize("NFC", relative)


def _load_text(path: Path, *, source: str) -> Document:
    text = _normalize_text(_read_utf8(path))
    if not text.strip():
        raise CorpusValidationError(f"{path}:1: document text must not be empty")
    try:
        return Document(id=source, text=text, source=source)
    except CorpusValidationError as exc:
        raise CorpusValidationError(f"{path}:1: {exc}") from exc


def _load_jsonl(path: Path, *, source: str) -> tuple[Document, ...]:
    text = _read_utf8(path)
    if not text:
        raise CorpusValidationError(f"{path}:1: corpus JSONL must not be empty")
    documents: list[Document] = []
    identifiers: dict[str, int] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise CorpusValidationError(
                f"{path}:{line_number}: blank lines are not valid JSONL records"
            )
        record = _parse_json_object(line, path=path, line_number=line_number)
        document = _parse_corpus_record(
            record,
            path=path,
            line_number=line_number,
            source=source,
        )
        if document.id in identifiers:
            raise CorpusValidationError(
                f"{path}:{line_number}: duplicate document ID {document.id!r}; "
                f"first seen on line {identifiers[document.id]}"
            )
        identifiers[document.id] = line_number
        documents.append(document)
    if not documents:
        raise CorpusValidationError(f"{path}:1: corpus JSONL must not be empty")
    return tuple(documents)


def _read_utf8(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CorpusValidationError(f"cannot read corpus file {path}: {exc}") from exc
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        line_number = data[: exc.start].count(b"\n") + 1
        raise CorpusValidationError(
            f"{path}:{line_number}: corpus files must be valid UTF-8"
        ) from exc


def _normalize_text(value: str) -> str:
    normalized_newlines = value.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", normalized_newlines)


def _parse_json_object(
    line: str,
    *,
    path: Path,
    line_number: int,
) -> dict[str, object]:
    try:
        value = cast(
            object,
            json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_non_json_constant,
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise CorpusValidationError(
            f"{path}:{line_number}: invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CorpusValidationError(
            f"{path}:{line_number}: each record must be a JSON object"
        )
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _parse_corpus_record(
    record: Mapping[str, object],
    *,
    path: Path,
    line_number: int,
    source: str,
) -> Document:
    required = {"id", "text"}
    allowed = required | {"metadata"}
    missing = sorted(required - set(record))
    unknown = sorted(set(record) - allowed)
    if missing or unknown:
        raise CorpusValidationError(
            f"{path}:{line_number}: record fields are invalid; "
            f"missing={missing}, unknown={unknown}"
        )
    identifier = _record_string(record["id"], "id", path, line_number)
    text = _normalize_text(_record_string(record["text"], "text", path, line_number))
    metadata_value = record.get("metadata", {})
    try:
        metadata = normalize_json_mapping(
            metadata_value,
            field_name=f"Document[{identifier!r}].metadata",
            error_type=CorpusValidationError,
        )
        return Document(
            id=identifier,
            text=text,
            metadata=metadata,
            source=source,
        )
    except CorpusValidationError as exc:
        raise CorpusValidationError(f"{path}:{line_number}: {exc}") from exc


def _record_string(
    value: object,
    field_name: str,
    path: Path,
    line_number: int,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusValidationError(
            f"{path}:{line_number}: {field_name} must be a non-empty string"
        )
    return unicodedata.normalize("NFC", value)


__all__ = ["load_documents"]
