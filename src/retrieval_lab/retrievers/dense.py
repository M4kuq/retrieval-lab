"""Exact dense retrieval with a lazy sentence-transformers adapter."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib import import_module
from json import dumps
from numbers import Real
from typing import Protocol, cast

from retrieval_lab.exceptions import (
    OptionalDependencyError,
    RetrievalLabError,
    RetrieverContractError,
)
from retrieval_lab.models import Chunk, JSONValue, SearchResult
from retrieval_lab.retrievers.base import BaseRetriever

DEFAULT_MODEL_ID = "intfloat/multilingual-e5-small"
DEFAULT_QUERY_PROMPT = "query: "
DEFAULT_DOCUMENT_PROMPT = "passage: "
_SENTENCE_TRANSFORMER_CLASS = "SentenceTransformer"


@dataclass(frozen=True)
class EmbeddingModelMetadata:
    """Identity and reproducibility metadata for an embedding backend."""

    model_id: str
    requested_revision: str | None = None
    resolved_revision: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.model_id, field_name="model_id")
        _validate_optional_revision(
            self.requested_revision,
            field_name="requested_revision",
        )
        _validate_optional_revision(
            self.resolved_revision,
            field_name="resolved_revision",
        )


class EmbeddingBackend(Protocol):
    """Typed embedding service used by :class:`DenseRetriever`.

    ``texts`` are already prefixed with the retriever's query or document prompt.
    Implementations return one finite numeric vector per input string; the
    retriever validates the concrete values before storing or scoring them.
    """

    @property
    def metadata(self) -> EmbeddingModelMetadata:
        """Return the model identity used to produce embeddings."""

    @property
    def cache_identity(self) -> JSONValue:
        """Return stable JSON configuration identity for this backend."""

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> Sequence[Sequence[float]]:
        """Encode the already-prefixed input strings."""


class _SentenceTransformerModel(Protocol):
    """Minimal typed surface used from sentence-transformers at runtime."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
        prompt: str,
    ) -> object:
        """Return the embeddings for ``sentences``."""


class _SentenceTransformerFactory(Protocol):
    """Runtime factory shape for the optional SentenceTransformer class."""

    def __call__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
    ) -> _SentenceTransformerModel:
        """Load one sentence-transformers model."""


class _SentenceTransformersEmbeddingBackend:
    """Lazy optional adapter around ``sentence_transformers.SentenceTransformer``."""

    def __init__(self, *, model_id: str, revision: str | None) -> None:
        self._metadata = EmbeddingModelMetadata(
            model_id=model_id,
            requested_revision=revision,
        )
        self._model: _SentenceTransformerModel | None = None

    @property
    def metadata(self) -> EmbeddingModelMetadata:
        """Return the requested and, if known, resolved model revision."""

        return self._metadata

    @property
    def cache_identity(self) -> JSONValue:
        """Return the built-in provider identity required by the protocol."""

        return "sentence-transformers"

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> Sequence[Sequence[float]]:
        """Load on first use and return unnormalized NumPy-backed embeddings."""

        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
            prompt="",
        )
        return cast(Sequence[Sequence[float]], vectors)

    def _get_model(self) -> _SentenceTransformerModel:
        if self._model is not None:
            return self._model

        try:
            module = import_module("sentence_transformers")
        except ImportError as exc:
            # A missing top-level optional package is actionable for callers.
            # Import failures from one of its transitive dependencies are
            # provider failures and must retain their original cause.
            if (
                isinstance(exc, ModuleNotFoundError)
                and exc.name == "sentence_transformers"
            ):
                raise OptionalDependencyError(
                    "Dense retrieval requires the optional dependency; install it with "
                    "`pip install retrieval-lab[dense]`."
                ) from None
            raise RetrieverContractError(
                "dense embedding backend failed while importing sentence-transformers"
            ) from exc

        factory = cast(
            _SentenceTransformerFactory,
            getattr(module, _SENTENCE_TRANSFORMER_CLASS),
        )
        model = factory(
            self._metadata.model_id,
            revision=self._metadata.requested_revision,
        )
        resolved_revision = _read_resolved_revision(
            model,
            requested_revision=self._metadata.requested_revision,
        )
        if resolved_revision is not None:
            self._metadata = replace(
                self._metadata,
                resolved_revision=resolved_revision,
            )
        self._model = model
        return model


class DenseRetriever(BaseRetriever):
    """Rank chunks by exact inner product over validated embedding vectors."""

    def __init__(
        self,
        *,
        backend: EmbeddingBackend | None = None,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        query_prompt: str = DEFAULT_QUERY_PROMPT,
        document_prompt: str = DEFAULT_DOCUMENT_PROMPT,
    ) -> None:
        """Create an unindexed dense retriever with deterministic settings.

        Supplying ``backend`` is useful for custom embedding services and tests.
        The default adapter does not import, load, or download sentence-transformers
        until the first indexing or search encode call.
        """

        _validate_non_empty_string(model_id, field_name="model_id")
        _validate_optional_revision(revision, field_name="revision")
        if not isinstance(normalize_embeddings, bool):
            raise RetrieverContractError("normalize_embeddings must be a boolean")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise RetrieverContractError("batch_size must be a positive integer")
        if not isinstance(query_prompt, str):
            raise RetrieverContractError("query_prompt must be a string")
        if not isinstance(document_prompt, str):
            raise RetrieverContractError("document_prompt must be a string")

        resolved_backend: EmbeddingBackend
        if backend is None:
            resolved_backend = _SentenceTransformersEmbeddingBackend(
                model_id=model_id,
                revision=revision,
            )
        else:
            resolved_backend = backend
        self._metadata_for(resolved_backend)

        self._backend = resolved_backend
        self._normalize_embeddings = normalize_embeddings
        self._batch_size = batch_size
        self._query_prompt = query_prompt
        self._document_prompt = document_prompt
        self._chunks: tuple[Chunk, ...] | None = None
        self._vectors: tuple[tuple[float, ...], ...] = ()
        self._dimension: int | None = None
        self._indexed_identity: Mapping[str, JSONValue] | None = None
        # Cache restoration may know a resolved revision that the backend does
        # not expose until it is loaded.  Keep that metadata on the retriever,
        # never by mutating an arbitrary user-supplied backend object.
        self._metadata_override: EmbeddingModelMetadata | None = None

    @property
    def name(self) -> str:
        """Return the stable dense retriever name."""

        return "dense"

    @property
    def uses_default_backend(self) -> bool:
        """Return whether this retriever uses the built-in ST adapter."""

        return isinstance(self._backend, _SentenceTransformersEmbeddingBackend)

    def _cache_identity(self, *, require_custom: bool = True) -> dict[str, JSONValue]:
        """Return the backend identity used to isolate custom index artifacts.

        Custom providers must expose a stable JSON-compatible ``cache_identity``.
        Only its digest is serialized so provider configuration and credentials
        cannot leak through result manifests.
        """

        implementation = type(self._backend)
        identity: dict[str, JSONValue] = {
            "backend_type": f"{implementation.__module__}.{implementation.__qualname__}"
        }
        if self.uses_default_backend:
            return identity
        missing = object()
        try:
            custom_identity = getattr(self._backend, "cache_identity", missing)
        except Exception as exc:
            raise RetrieverContractError(
                "embedding backend cache identity could not be read"
            ) from exc
        if custom_identity is missing or custom_identity is None:
            if require_custom:
                raise RetrieverContractError(
                    "custom embedding backend must provide a stable non-null "
                    "cache_identity"
                )
            return identity
        try:
            normalized = _normalize_cache_identity(custom_identity)
            canonical = dumps(
                normalized,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            identity["cache_identity_sha256"] = sha256(canonical).hexdigest()
        except RecursionError as exc:
            raise RetrieverContractError(
                "embedding backend cache identity is cyclic or too deeply nested"
            ) from exc
        return identity

    @property
    def settings(self) -> Mapping[str, JSONValue]:
        """Return the deterministic dense configuration for a run manifest."""

        metadata = self._effective_metadata()
        return {
            "backend": self._cache_identity(),
            "batch_size": self._batch_size,
            "document_prompt": self._document_prompt,
            "model_id": metadata.model_id,
            "name": self.name,
            "normalize_embeddings": self._normalize_embeddings,
            "query_prompt": self._query_prompt,
            "requested_revision": metadata.requested_revision,
            "resolved_revision": metadata.resolved_revision,
            "similarity": "inner_product",
            "type": "dense",
        }

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Atomically replace the index after validating every document vector."""

        indexed = _validate_chunks(chunks)
        # HybridRetriever can share one DenseRetriever instance with a
        # top-level strategy.  Re-indexing the exact same chunks is a no-op,
        # which also prevents a second backend encode call.
        current_identity = self._index_identity()
        if self._chunks == indexed and self._indexed_identity == current_identity:
            return
        previous_override = self._metadata_override
        self._metadata_override = None
        document_texts = tuple(self._document_prompt + chunk.text for chunk in indexed)
        try:
            if document_texts:
                vectors = self._encode(document_texts, context="documents")
            else:
                vectors = ()
            # A lazy backend can update its resolved model revision during
            # encoding. Validate it before publishing the replacement index.
            self._metadata_for(self._backend)
            replacement_identity = self._index_identity()
        except Exception:
            self._metadata_override = previous_override
            raise
        dimension = len(vectors[0]) if vectors else None
        self._chunks = indexed
        self._vectors = vectors
        self._dimension = dimension
        self._indexed_identity = replacement_identity

    def _index_identity(self) -> Mapping[str, JSONValue]:
        """Return settings that determine the stored document vectors."""

        metadata = self._effective_metadata()
        return {
            "backend": self._cache_identity(require_custom=False),
            "document_prompt": self._document_prompt,
            "model_id": metadata.model_id,
            "normalize_embeddings": self._normalize_embeddings,
            "requested_revision": metadata.requested_revision,
            "resolved_revision": metadata.resolved_revision,
        }

    def _effective_metadata(self) -> EmbeddingModelMetadata:
        """Return backend metadata plus any safe cache-restored revision."""

        metadata = self._metadata_for(self._backend)
        override = self._metadata_override
        if override is not None and (
            override.model_id == metadata.model_id
            and override.requested_revision == metadata.requested_revision
            and metadata.resolved_revision is None
        ):
            return override
        return metadata

    def _cache_export(
        self,
    ) -> tuple[
        tuple[Chunk, ...],
        tuple[tuple[float, ...], ...],
        int,
        EmbeddingModelMetadata,
    ]:
        """Return validated index data for the internal JSON cache adapter."""

        if self._chunks is None or self._dimension is None:
            raise RetrieverContractError("dense retriever has no cacheable index")
        return self._chunks, self._vectors, self._dimension, self._effective_metadata()

    def _cache_restore(
        self,
        chunks: Sequence[Chunk],
        vectors: object,
        *,
        resolved_revision: str | None,
    ) -> None:
        """Restore a validated finite matrix without invoking the backend."""

        indexed = _validate_chunks(chunks)
        if not indexed:
            raise RetrieverContractError("dense cache cannot restore an empty index")
        validated = _validate_embedding_matrix(
            vectors,
            expected_count=len(indexed),
            context="cached documents",
        )
        dimension = len(validated[0])
        metadata = self._metadata_for(self._backend)
        if (
            metadata.resolved_revision is not None
            and metadata.resolved_revision != resolved_revision
        ):
            raise RetrieverContractError("dense cache resolved revision is stale")
        self._metadata_override = EmbeddingModelMetadata(
            model_id=metadata.model_id,
            requested_revision=metadata.requested_revision,
            resolved_revision=resolved_revision,
        )
        self._chunks = indexed
        self._vectors = validated
        self._dimension = dimension
        self._indexed_identity = self._index_identity()

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Encode one query and return the top exact inner-product results."""

        if self._chunks is None:
            raise RetrieverContractError(
                "dense retriever is not indexed; call index(chunks) before search"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RetrieverContractError(
                "top_k must be a positive integer; for example, top_k=5"
            )
        if not isinstance(query, str):
            raise RetrieverContractError("query must be a string")
        if not self._chunks:
            return []

        query_vector = self._encode(
            (self._query_prompt + query,),
            context="query",
        )[0]
        if (
            self._indexed_identity is not None
            and self._index_identity() != self._indexed_identity
        ):
            raise RetrieverContractError(
                "dense backend identity or revision changed after indexing"
            )
        if self._dimension is None:
            return []
        if len(query_vector) != self._dimension:
            raise RetrieverContractError(
                "dense query embedding dimension does not match the indexed "
                f"document dimension ({len(query_vector)} != {self._dimension})"
            )

        scored: list[tuple[float, Chunk]] = []
        for chunk, document_vector in zip(self._chunks, self._vectors, strict=True):
            score = sum(
                document_value * query_value
                for document_value, query_value in zip(
                    document_vector,
                    query_vector,
                    strict=True,
                )
            )
            if not math.isfinite(score):
                raise RetrieverContractError(
                    "dense inner-product score must be finite; embedding values are "
                    "too large to score safely"
                )
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

    def _encode(
        self,
        texts: Sequence[str],
        *,
        context: str,
    ) -> tuple[tuple[float, ...], ...]:
        try:
            raw_vectors = self._backend.encode(texts, batch_size=self._batch_size)
        except RetrievalLabError:
            raise
        except Exception as exc:
            raise RetrieverContractError(
                f"dense embedding backend failed while encoding {context}"
            ) from exc

        vectors = _validate_embedding_matrix(
            raw_vectors,
            expected_count=len(texts),
            context=context,
        )
        if self._normalize_embeddings:
            return tuple(
                _normalize_vector(vector, context=context) for vector in vectors
            )
        return vectors

    @staticmethod
    def _metadata_for(backend: EmbeddingBackend) -> EmbeddingModelMetadata:
        try:
            metadata = backend.metadata
        except RetrievalLabError:
            raise
        except Exception as exc:
            raise RetrieverContractError(
                "embedding backend must expose EmbeddingModelMetadata through "
                "its metadata property"
            ) from exc
        if not isinstance(metadata, EmbeddingModelMetadata):
            raise RetrieverContractError(
                "embedding backend metadata must be an EmbeddingModelMetadata value"
            )
        return metadata


def _validate_non_empty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RetrieverContractError(f"{field_name} must be a non-empty string")


def _validate_optional_revision(value: object, *, field_name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise RetrieverContractError(f"{field_name} must be a non-empty string or None")


def _validate_chunks(chunks: Sequence[Chunk]) -> tuple[Chunk, ...]:
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise RetrieverContractError("chunks must be a sequence of Chunk records")
    indexed = tuple(chunks)
    seen: set[str] = set()
    duplicates: set[str] = set()
    for position, chunk in enumerate(indexed):
        if not isinstance(chunk, Chunk):
            raise RetrieverContractError(f"chunks[{position}] must be a Chunk record")
        if chunk.id in seen:
            duplicates.add(chunk.id)
        seen.add(chunk.id)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise RetrieverContractError(
            "dense index received duplicate chunk identifiers: "
            f"{duplicate_list}; provide one chunk per identifier"
        )
    return indexed


def _validate_embedding_matrix(
    raw_vectors: object,
    *,
    expected_count: int,
    context: str,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(raw_vectors, (str, bytes, Mapping)):
        raise RetrieverContractError(
            f"dense embedding backend returned an invalid {context} matrix"
        )
    try:
        rows = tuple(iter(cast(Iterable[object], raw_vectors)))
    except TypeError as exc:
        raise RetrieverContractError(
            f"dense embedding backend returned an invalid {context} matrix"
        ) from exc
    if len(rows) != expected_count:
        raise RetrieverContractError(
            f"dense embedding backend returned {len(rows)} {context} vectors; "
            f"expected {expected_count}"
        )

    vectors: list[tuple[float, ...]] = []
    dimension: int | None = None
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes, Mapping)):
            raise RetrieverContractError(
                f"dense {context} embedding row {row_index} must be a numeric vector"
            )
        try:
            values = tuple(iter(cast(Iterable[object], row)))
        except TypeError as exc:
            raise RetrieverContractError(
                f"dense {context} embedding row {row_index} must be a numeric vector"
            ) from exc
        if dimension is None:
            dimension = len(values)
            if dimension == 0:
                raise RetrieverContractError(
                    f"dense {context} embedding row {row_index} must not be empty"
                )
        elif len(values) != dimension:
            raise RetrieverContractError(
                f"dense {context} embedding vectors must have consistent dimensions"
            )

        vector: list[float] = []
        for value_index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise RetrieverContractError(
                    f"dense {context} embedding row {row_index} value {value_index} "
                    "must be a finite real number"
                )
            try:
                normalized = float(value)
            except OverflowError as exc:
                raise RetrieverContractError(
                    f"dense {context} embedding row {row_index} value {value_index} "
                    "must be a finite real number"
                ) from exc
            if not math.isfinite(normalized):
                raise RetrieverContractError(
                    f"dense {context} embedding row {row_index} value {value_index} "
                    "must be a finite real number"
                )
            vector.append(normalized)
        if math.hypot(*vector) == 0.0:
            raise RetrieverContractError(
                f"dense {context} embedding vectors must not be zero vectors"
            )
        vectors.append(tuple(vector))
    return tuple(vectors)


def _normalize_vector(vector: tuple[float, ...], *, context: str) -> tuple[float, ...]:
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0.0:
        raise RetrieverContractError(
            f"dense {context} embedding vectors must have a finite, non-zero L2 norm"
        )
    return tuple(value / norm for value in vector)


def _read_resolved_revision(
    model: object,
    *,
    requested_revision: str | None,
) -> str | None:
    """Read a full resolved commit hash from safe model metadata surfaces."""

    direct = _read_commit_candidate(
        model,
        ("resolved_revision", "_commit_hash", "commit_hash", "revision"),
    )
    if direct is not None:
        return direct

    config = _safe_getattr(model, "config")
    from_config = _read_commit_candidate(
        config,
        ("_commit_hash", "commit_hash", "revision"),
    )
    if from_config is not None:
        return from_config

    tokenizer = _safe_getattr(model, "tokenizer")
    init_kwargs = _safe_getattr(tokenizer, "init_kwargs")
    from_tokenizer = _read_commit_candidate(
        init_kwargs,
        ("_commit_hash", "commit_hash", "revision"),
    )
    if from_tokenizer is not None:
        return from_tokenizer

    if _is_commit_hash(requested_revision):
        return requested_revision
    return None


def _safe_getattr(value: object, attribute: str) -> object | None:
    """Return an attribute for best-effort optional metadata introspection."""

    try:
        return getattr(value, attribute, None)
    except Exception:
        return None


def _read_commit_candidate(
    source: object | None,
    names: Sequence[str],
) -> str | None:
    """Return the first full hexadecimal commit from mapping keys or attributes."""

    for name in names:
        value: object | None
        if isinstance(source, Mapping):
            try:
                value = source.get(name)
            except Exception:
                continue
        else:
            value = _safe_getattr(source, name)
        if _is_commit_hash(value):
            return cast(str, value)
    return None


def _is_commit_hash(value: object) -> bool:
    """Return whether ``value`` is a full Git SHA-1 or SHA-256 identifier."""

    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _normalize_cache_identity(value: object) -> JSONValue:
    """Validate a provider-supplied cache identity without invoking JSON sorting."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RetrieverContractError(
                "embedding backend cache identity must be finite"
            )
        return value
    if isinstance(value, Mapping):
        keys = tuple(value)
        if not all(isinstance(key, str) for key in keys):
            raise RetrieverContractError(
                "embedding backend cache identity keys must be strings"
            )
        return {
            key: _normalize_cache_identity(value[key])
            for key in sorted(cast(tuple[str, ...], keys))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_identity(item) for item in value]
    raise RetrieverContractError(
        "embedding backend cache identity must contain JSON values"
    )


__all__ = [
    "DEFAULT_DOCUMENT_PROMPT",
    "DEFAULT_MODEL_ID",
    "DEFAULT_QUERY_PROMPT",
    "DenseRetriever",
    "EmbeddingBackend",
    "EmbeddingModelMetadata",
]
