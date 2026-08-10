"""Deterministic, dependency-free BM25 retrieval."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from inspect import getclosurevars
from itertools import pairwise
from json import dumps
from types import CodeType, FunctionType

from retrieval_lab.exceptions import RetrieverContractError
from retrieval_lab.models import Chunk, JSONValue, SearchResult
from retrieval_lab.retrievers.base import (
    BaseRetriever,
    _chunk_payload,
    _serialized_index_size_bytes,
)

Tokenizer = Callable[[str], Sequence[str]]


def _is_cjk_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
        or 0x3040 <= codepoint <= 0x30FF
        or 0x31F0 <= codepoint <= 0x31FF
        or 0xFF66 <= codepoint <= 0xFF9F
        or 0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def _append_cjk_tokens(characters: list[str], tokens: list[str]) -> None:
    if not characters:
        return
    tokens.extend(characters)
    tokens.extend(left + right for left, right in pairwise(characters))
    characters.clear()


def _append_word_token(characters: list[str], tokens: list[str]) -> None:
    if characters:
        tokens.append("".join(characters))
        characters.clear()


def _default_tokenizer(value: str) -> tuple[str, ...]:
    """Tokenize normalized Unicode words and CJK character n-grams."""
    normalized = unicodedata.normalize("NFC", value).casefold()
    tokens: list[str] = []
    word: list[str] = []
    cjk: list[str] = []

    for character in normalized:
        if _is_cjk_character(character):
            _append_word_token(word, tokens)
            cjk.append(character)
            continue

        category = unicodedata.category(character)
        if category[0] in {"L", "N"} or (category[0] == "M" and word):
            _append_cjk_tokens(cjk, tokens)
            word.append(character)
            continue

        _append_word_token(word, tokens)
        _append_cjk_tokens(cjk, tokens)

    _append_word_token(word, tokens)
    _append_cjk_tokens(cjk, tokens)
    return tuple(tokens)


def _validate_k1(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrieverContractError("k1 must be a finite number greater than zero")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise RetrieverContractError(
            "k1 must be a finite number greater than zero"
        ) from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise RetrieverContractError("k1 must be a finite number greater than zero")
    return normalized


def _validate_b(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrieverContractError("b must be a finite number between zero and one")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise RetrieverContractError(
            "b must be a finite number between zero and one"
        ) from exc
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise RetrieverContractError("b must be a finite number between zero and one")
    return normalized


class BM25Retriever(BaseRetriever):
    """Rank chunks with deterministic Okapi BM25 scores.

    The default tokenizer applies Unicode NFC normalization and ``casefold()``.
    Letters and numbers outside CJK ranges form word tokens. Contiguous CJK text
    emits character unigrams and overlapping bigrams so unsegmented Japanese text
    does not collapse into one sentence-sized token. A custom tokenizer can replace
    this behavior completely.
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Tokenizer | None = None,
        tokenizer_identity: str | None = None,
    ) -> None:
        """Create an unindexed BM25 retriever with validated parameters."""
        self._k1 = _validate_k1(k1)
        self._b = _validate_b(b)
        if tokenizer is not None and not callable(tokenizer):
            raise RetrieverContractError("tokenizer must be callable or None")
        if tokenizer_identity is not None and (
            not isinstance(tokenizer_identity, str) or not tokenizer_identity.strip()
        ):
            raise RetrieverContractError(
                "tokenizer_identity must be a non-empty string or None"
            )
        if tokenizer is None and tokenizer_identity is not None:
            raise RetrieverContractError(
                "tokenizer_identity is only valid with a custom tokenizer"
            )
        self._tokenizer = _default_tokenizer if tokenizer is None else tokenizer
        self._tokenizer_identity = (
            "default"
            if tokenizer is None
            else _callable_identity(tokenizer, explicit=tokenizer_identity)
        )
        self._chunks: tuple[Chunk, ...] | None = None
        self._term_frequencies: tuple[Counter[str], ...] = ()
        self._document_lengths: tuple[int, ...] = ()
        self._document_frequencies: Counter[str] = Counter()
        self._average_document_length = 0.0

    @property
    def name(self) -> str:
        """Return the stable BM25 strategy name."""
        return "bm25"

    @property
    def settings(self) -> Mapping[str, JSONValue]:
        """Return every configured value that affects BM25 rankings."""

        return {
            "b": self._b,
            "k1": self._k1,
            "name": self.name,
            "tokenizer": self._tokenizer_identity,
            "type": "bm25",
        }

    @property
    def index_size_bytes(self) -> int | None:
        """Return the deterministic logical size of the BM25 index.

        The payload includes both shared chunks and the token frequency tables
        used for scoring.  It intentionally avoids ``sys.getsizeof`` because
        Python allocator details are not stable across environments.
        """

        if self._chunks is None:
            return None
        if not self._chunks:
            return 0
        payload = {
            "chunks": [_chunk_payload(chunk) for chunk in self._chunks],
            "document_frequencies": dict(sorted(self._document_frequencies.items())),
            "document_lengths": self._document_lengths,
            "term_frequencies": [
                dict(sorted(frequencies.items()))
                for frequencies in self._term_frequencies
            ],
        }
        return _serialized_index_size_bytes(payload)

    def _tokenize(self, value: str, *, context: str) -> tuple[str, ...]:
        try:
            raw_tokens = self._tokenizer(value)
            if isinstance(raw_tokens, (str, bytes)) or not isinstance(
                raw_tokens, Sequence
            ):
                raise TypeError("tokenizer must return a sequence of strings")

            tokens: list[str] = []
            for position, token in enumerate(raw_tokens):
                if not isinstance(token, str):
                    raise TypeError(
                        f"tokenizer result[{position}] must be a string, got "
                        f"{type(token).__name__}"
                    )
                # Empty and whitespace-only tokens carry no lexical information.
                if token.strip():
                    tokens.append(token)
            return tuple(tokens)
        except Exception as error:
            raise RetrieverContractError(
                f"bm25 tokenizer failed while processing {context}"
            ) from error

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Atomically replace the index with validated, tokenized chunks."""
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise RetrieverContractError("chunks must be a sequence of Chunk records")

        indexed = tuple(chunks)
        seen: set[str] = set()
        duplicates: set[str] = set()
        frequencies: list[Counter[str]] = []
        lengths: list[int] = []
        document_frequencies: Counter[str] = Counter()

        for position, chunk in enumerate(indexed):
            if not isinstance(chunk, Chunk):
                raise RetrieverContractError(
                    f"chunks[{position}] must be a Chunk record"
                )
            if chunk.id in seen:
                duplicates.add(chunk.id)
            seen.add(chunk.id)

        if duplicates:
            duplicate_list = ", ".join(sorted(duplicates))
            raise RetrieverContractError(
                "bm25 index received duplicate chunk identifiers: "
                f"{duplicate_list}; provide one chunk per identifier"
            )

        for chunk in indexed:
            tokens = self._tokenize(chunk.text, context=f"chunk {chunk.id!r}")
            term_frequency = Counter(tokens)
            frequencies.append(term_frequency)
            lengths.append(len(tokens))
            document_frequencies.update(term_frequency.keys())

        average_document_length = sum(lengths) / len(lengths) if lengths else 0.0

        self._chunks = indexed
        self._term_frequencies = tuple(frequencies)
        self._document_lengths = tuple(lengths)
        self._document_frequencies = document_frequencies
        self._average_document_length = average_document_length

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return at most ``top_k`` positive-score chunks in stable rank order."""
        if self._chunks is None:
            raise RetrieverContractError(
                "bm25 retriever is not indexed; call index(chunks) before search"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RetrieverContractError(
                "top_k must be a positive integer; for example, top_k=5"
            )
        if not isinstance(query, str):
            raise RetrieverContractError("query must be a string")

        query_frequencies = Counter(self._tokenize(query, context="query"))
        if (
            not query_frequencies
            or not self._chunks
            or self._average_document_length == 0.0
        ):
            return []

        document_count = len(self._chunks)
        scored: list[tuple[float, Chunk]] = []
        for chunk, term_frequencies, document_length in zip(
            self._chunks,
            self._term_frequencies,
            self._document_lengths,
            strict=True,
        ):
            length_ratio = document_length / self._average_document_length
            length_normalization = 1.0 - self._b + self._b * length_ratio
            score = 0.0
            for term, query_frequency in query_frequencies.items():
                term_frequency = term_frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                document_frequency = self._document_frequencies[term]
                inverse_document_frequency = math.log1p(
                    (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                saturated_frequency = (term_frequency * (self._k1 + 1.0)) / (
                    term_frequency + self._k1 * length_normalization
                )
                score += (
                    float(query_frequency)
                    * inverse_document_frequency
                    * saturated_frequency
                )
            if score > 0.0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [
            SearchResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=score,
                rank=rank,
                metadata=dict(chunk.metadata),
            )
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]


__all__ = ["BM25Retriever", "Tokenizer"]


def _callable_identity(value: Tokenizer, *, explicit: str | None) -> str:
    """Return a privacy-preserving deterministic identity for a tokenizer."""

    if explicit is not None:
        payload: object = {"explicit": explicit}
    elif isinstance(value, FunctionType):
        try:
            closure_variables = getclosurevars(value)
            closure = tuple(
                _identity_value(cell.cell_contents)
                for cell in (value.__closure__ or ())
            )
            payload = {
                "closure": closure,
                "code": _code_identity(value.__code__),
                "defaults": _identity_value(value.__defaults__),
                "globals": {
                    name: _identity_value(global_value)
                    for name, global_value in sorted(closure_variables.globals.items())
                },
                "kwdefaults": _identity_value(value.__kwdefaults__),
                "module": value.__module__,
                "qualname": value.__qualname__,
            }
        except (TypeError, ValueError) as exc:
            raise RetrieverContractError(
                "custom tokenizer identity cannot be derived safely; pass a stable "
                "tokenizer_identity"
            ) from exc
    else:
        raise RetrieverContractError(
            "custom callable tokenizer must provide a stable tokenizer_identity"
        )

    canonical = dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"custom:sha256:{sha256(canonical).hexdigest()}"


def _code_identity(code: CodeType) -> dict[str, object]:
    """Describe executable function code without paths or source locations."""

    return {
        "argcount": code.co_argcount,
        "bytecode": code.co_code.hex(),
        "cellvars": code.co_cellvars,
        "consts": tuple(_identity_value(item) for item in code.co_consts),
        "flags": code.co_flags,
        "freevars": code.co_freevars,
        "kwonlyargcount": code.co_kwonlyargcount,
        "names": code.co_names,
        "posonlyargcount": code.co_posonlyargcount,
        "varnames": code.co_varnames,
    }


def _identity_value(value: object) -> object:
    """Normalize immutable function state used by the tokenizer fingerprint."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identity numbers must be finite")
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, tuple):
        return tuple(_identity_value(item) for item in value)
    if isinstance(value, frozenset):
        normalized = tuple(_identity_value(item) for item in value)
        return {"frozenset": sorted(normalized, key=repr)}
    if isinstance(value, CodeType):
        return {"code": _code_identity(value)}
    raise TypeError(f"unsupported tokenizer identity state: {type(value).__name__}")
