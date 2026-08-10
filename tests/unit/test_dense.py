"""Unit tests for exact dense retrieval without downloading a real model."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import pytest

from retrieval_lab import (
    DenseRetriever,
    EmbeddingModelMetadata,
    OptionalDependencyError,
)
from retrieval_lab.exceptions import RetrieverContractError
from retrieval_lab.models import Chunk
from retrieval_lab.retrievers import dense


def _chunk(chunk_id: str, text: str, *, document_id: str | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id or f"document-{chunk_id}",
        text=text,
        start_offset=0,
        end_offset=len(text),
        metadata={"chunk": chunk_id},
    )


class FakeEmbeddingBackend:
    """Map complete encode requests to fixed vectors and keep call evidence."""

    def __init__(
        self,
        responses: dict[tuple[str, ...], object],
        *,
        model_id: str = "fake/embedding-model",
        requested_revision: str | None = "requested-revision",
        resolved_revision: str | None = "resolved-revision",
    ) -> None:
        self.responses = responses
        self.cache_identity = {
            "model_id": model_id,
            "requested_revision": requested_revision,
        }
        self.metadata = EmbeddingModelMetadata(
            model_id=model_id,
            requested_revision=requested_revision,
            resolved_revision=resolved_revision,
        )
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> Sequence[Sequence[float]]:
        requested = tuple(texts)
        self.calls.append((requested, batch_size))
        response = self.responses[requested]
        return cast(Sequence[Sequence[float]], response)


def test_exact_inner_product_ranking_prefixes_and_full_result_fields() -> None:
    backend = FakeEmbeddingBackend(
        {
            ("passage: first", "passage: second"): [[3.0, 4.0], [1.0, 0.0]],
            ("query: find",): [[0.6, 0.8]],
        }
    )
    retriever = DenseRetriever(
        backend=backend, normalize_embeddings=False, batch_size=7
    )
    retriever.index([_chunk("a", "first"), _chunk("b", "second")])

    results = retriever.search("find", top_k=2)

    assert backend.calls == [
        (("passage: first", "passage: second"), 7),
        (("query: find",), 7),
    ]
    assert [(item.chunk_id, item.score, item.rank) for item in results] == [
        ("a", pytest.approx(5.0), 1),
        ("b", pytest.approx(0.6), 2),
    ]
    assert [item.document_id for item in results] == ["document-a", "document-b"]
    assert [item.text for item in results] == ["first", "second"]
    assert [item.metadata for item in results] == [
        {"chunk": "a"},
        {"chunk": "b"},
    ]


def test_normalization_changes_scoring_and_can_be_disabled() -> None:
    responses = {
        ("passage: first", "passage: second"): [[2.0, 0.0], [1.0, 0.0]],
        ("query: find",): [[3.0, 0.0]],
    }
    normalized = DenseRetriever(backend=FakeEmbeddingBackend(responses))
    raw = DenseRetriever(
        backend=FakeEmbeddingBackend(responses),
        normalize_embeddings=False,
    )
    chunks = [_chunk("z", "first"), _chunk("a", "second")]
    normalized.index(chunks)
    raw.index(chunks)

    normalized_results = normalized.search("find", top_k=2)
    raw_results = raw.search("find", top_k=2)

    assert [(item.chunk_id, item.score) for item in normalized_results] == [
        ("a", pytest.approx(1.0)),
        ("z", pytest.approx(1.0)),
    ]
    assert [(item.chunk_id, item.score) for item in raw_results] == [
        ("z", pytest.approx(6.0)),
        ("a", pytest.approx(3.0)),
    ]


def test_zero_and_negative_scores_are_ranked_and_returned() -> None:
    backend = FakeEmbeddingBackend(
        {
            ("passage: negative-one", "passage: negative-two", "passage: zero"): [
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
            ],
            ("query: find",): [[-1.0, 0.0]],
        }
    )
    retriever = DenseRetriever(backend=backend, normalize_embeddings=False)
    retriever.index(
        [
            _chunk("b", "negative-one"),
            _chunk("c", "negative-two"),
            _chunk("a", "zero"),
        ]
    )

    results = retriever.search("find", top_k=3)

    assert [(item.chunk_id, item.score, item.rank) for item in results] == [
        ("a", 0.0, 1),
        ("b", -1.0, 2),
        ("c", -2.0, 3),
    ]


def test_search_respects_top_k_and_tie_breaks_by_chunk_id() -> None:
    backend = FakeEmbeddingBackend(
        {
            ("passage: first", "passage: second", "passage: third"): [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            ("query: find",): [[1.0, 0.0]],
        }
    )
    retriever = DenseRetriever(backend=backend, normalize_embeddings=False)
    retriever.index([_chunk("z", "first"), _chunk("a", "second"), _chunk("b", "third")])

    results = retriever.search("find", top_k=2)

    assert [(item.chunk_id, item.rank) for item in results] == [("a", 1), ("z", 2)]


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("model_id", "", "model_id"),
        ("revision", "", "revision"),
        ("normalize_embeddings", 1, "normalize_embeddings"),
        ("batch_size", 0, "batch_size"),
        ("batch_size", True, "batch_size"),
        ("query_prompt", cast(str, object()), "query_prompt"),
        ("document_prompt", cast(str, object()), "document_prompt"),
    ],
)
def test_constructor_rejects_invalid_arguments(
    argument: str,
    value: object,
    message: str,
) -> None:
    kwargs: dict[str, object] = {argument: value}

    with pytest.raises(RetrieverContractError, match=message):
        DenseRetriever(**kwargs)  # type: ignore[arg-type]


def test_constructor_rejects_backend_without_typed_metadata() -> None:
    class MissingMetadataBackend:
        def encode(
            self,
            texts: Sequence[str],
            *,
            batch_size: int,
        ) -> Sequence[Sequence[float]]:
            return [[1.0] for _ in texts]

    with pytest.raises(RetrieverContractError, match="metadata"):
        DenseRetriever(backend=cast("object", MissingMetadataBackend()))


def test_search_requires_index_and_valid_arguments() -> None:
    backend = FakeEmbeddingBackend({})
    retriever = DenseRetriever(backend=backend)

    with pytest.raises(RetrieverContractError, match="not indexed"):
        retriever.search("find", top_k=1)

    retriever.index([])
    for invalid_top_k in (0, -1, True):
        with pytest.raises(RetrieverContractError, match="top_k"):
            retriever.search("find", top_k=invalid_top_k)
    with pytest.raises(RetrieverContractError, match="query"):
        retriever.search(cast(str, object()), top_k=1)
    assert retriever.search("find", top_k=1) == []
    assert backend.calls == []


def test_index_rejects_invalid_chunks_and_duplicate_ids_without_encoding() -> None:
    backend = FakeEmbeddingBackend({})
    retriever = DenseRetriever(backend=backend)

    with pytest.raises(RetrieverContractError, match="sequence"):
        retriever.index(cast(Sequence[Chunk], "not chunks"))
    with pytest.raises(RetrieverContractError, match=r"chunks\[0\]"):
        retriever.index(cast(Sequence[Chunk], [object()]))
    with pytest.raises(RetrieverContractError, match="duplicate"):
        retriever.index([_chunk("same", "first"), _chunk("same", "second")])
    assert backend.calls == []


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[1.0]], "expected 2"),
        ([[1.0], [1.0, 2.0]], "consistent dimensions"),
        ([[0.0, 0.0], [1.0, 0.0]], "zero vectors"),
        ([[float("nan"), 1.0], [1.0, 0.0]], "finite real"),
        ([[float("inf"), 1.0], [1.0, 0.0]], "finite real"),
        ([[True, 1.0], [1.0, 0.0]], "finite real"),
        ([["1.0", 1.0], [1.0, 0.0]], "finite real"),
        ([[], [1.0, 0.0]], "must not be empty"),
    ],
)
def test_index_rejects_invalid_embedding_matrices_atomically(
    matrix: object,
    message: str,
) -> None:
    backend = FakeEmbeddingBackend(
        {
            ("passage: original",): [[1.0, 0.0]],
            ("query: old",): [[1.0, 0.0]],
            ("passage: first", "passage: second"): matrix,
        }
    )
    retriever = DenseRetriever(backend=backend, normalize_embeddings=False)
    retriever.index([_chunk("original", "original")])

    with pytest.raises(RetrieverContractError, match=message):
        retriever.index([_chunk("first", "first"), _chunk("second", "second")])

    assert [item.chunk_id for item in retriever.search("old", top_k=1)] == ["original"]


def test_query_vector_contract_and_dimension_errors_are_rejected() -> None:
    backend = FakeEmbeddingBackend(
        {
            ("passage: document",): [[1.0, 0.0]],
            ("query: wrong-count",): [],
            ("query: wrong-dimension",): [[1.0, 0.0, 0.0]],
            ("query: zero",): [[0.0, 0.0]],
        }
    )
    retriever = DenseRetriever(backend=backend)
    retriever.index([_chunk("document", "document")])

    with pytest.raises(RetrieverContractError, match="expected 1"):
        retriever.search("wrong-count", top_k=1)
    with pytest.raises(RetrieverContractError, match="dimension"):
        retriever.search("wrong-dimension", top_k=1)
    with pytest.raises(RetrieverContractError, match="zero vectors"):
        retriever.search("zero", top_k=1)


def test_backend_failures_are_wrapped_but_library_errors_propagate() -> None:
    class FailingBackend:
        metadata = EmbeddingModelMetadata(model_id="failing")

        def encode(
            self,
            texts: Sequence[str],
            *,
            batch_size: int,
        ) -> Sequence[Sequence[float]]:
            raise RuntimeError("service unavailable")

    retriever = DenseRetriever(backend=FailingBackend())
    with pytest.raises(RetrieverContractError, match="backend failed") as raised:
        retriever.index([_chunk("document", "document")])
    assert isinstance(raised.value.__cause__, RuntimeError)

    class DependencyFailureBackend:
        metadata = EmbeddingModelMetadata(model_id="missing")

        def encode(
            self,
            texts: Sequence[str],
            *,
            batch_size: int,
        ) -> Sequence[Sequence[float]]:
            raise OptionalDependencyError("install the extra")

    with pytest.raises(OptionalDependencyError, match="install the extra"):
        DenseRetriever(backend=DependencyFailureBackend()).index(
            [_chunk("document", "document")]
        )


def test_default_adapter_is_lazy_and_missing_extra_has_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(module_name: str) -> object:
        assert module_name == "sentence_transformers"
        raise ModuleNotFoundError(
            "not installed",
            name="sentence_transformers",
        )

    monkeypatch.setattr(dense, "import_module", missing_dependency)
    retriever = DenseRetriever()

    with pytest.raises(
        OptionalDependencyError, match=r"pip install retrieval-lab\[dense\]"
    ):
        retriever.index([_chunk("document", "document")])


def test_transitive_default_adapter_import_failure_keeps_typed_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def transitive_failure(module_name: str) -> object:
        assert module_name == "sentence_transformers"
        raise ImportError("missing torch", name="torch")

    monkeypatch.setattr(dense, "import_module", transitive_failure)
    retriever = DenseRetriever()

    with pytest.raises(
        RetrieverContractError, match="importing sentence-transformers"
    ) as raised:
        retriever.index([_chunk("document", "document")])

    assert isinstance(raised.value.__cause__, ImportError)


def test_plain_default_adapter_import_failure_keeps_typed_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = ImportError("binary extension is incompatible")

    def import_failure(module_name: str) -> object:
        assert module_name == "sentence_transformers"
        raise cause

    monkeypatch.setattr(dense, "import_module", import_failure)

    with pytest.raises(RetrieverContractError) as raised:
        DenseRetriever().index([_chunk("document", "document")])

    assert raised.value.__cause__ is cause


def test_cached_revision_is_revalidated_after_query_initialization() -> None:
    class MutableRevisionBackend:
        def __init__(self) -> None:
            self.metadata = EmbeddingModelMetadata(model_id="mutable")

        def encode(
            self,
            texts: Sequence[str],
            *,
            batch_size: int,
        ) -> Sequence[Sequence[float]]:
            self.metadata = EmbeddingModelMetadata(
                model_id="mutable",
                resolved_revision="new-revision",
            )
            return [[1.0, 0.0] for _ in texts]

    retriever = DenseRetriever(backend=MutableRevisionBackend())
    retriever._cache_restore(
        [_chunk("document", "document")],
        [[1.0, 0.0]],
        resolved_revision="old-revision",
    )

    with pytest.raises(RetrieverContractError, match="revision changed"):
        retriever.search("query", 1)


def test_huge_integer_embedding_is_a_typed_contract_error() -> None:
    class HugeBackend:
        metadata = EmbeddingModelMetadata(model_id="huge")

        def encode(
            self,
            texts: Sequence[str],
            *,
            batch_size: int,
        ) -> Sequence[Sequence[float]]:
            return [[10**10000, 1.0] for _ in texts]

    with pytest.raises(RetrieverContractError, match="finite real number"):
        DenseRetriever(backend=HugeBackend()).index([_chunk("document", "document")])


def test_default_adapter_passes_sentence_transformers_options_and_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeSentenceTransformer:
        resolved_revision = "d" * 40

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
            calls["encode"] = {
                "batch_size": batch_size,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
                "prompt": prompt,
                "sentences": tuple(sentences),
                "show_progress_bar": show_progress_bar,
            }
            return [[1.0, 0.0] for _ in sentences]

    model = FakeSentenceTransformer()

    def factory(
        model_id: str, *, revision: str | None = None
    ) -> FakeSentenceTransformer:
        calls["factory"] = {"model_id": model_id, "revision": revision}
        return model

    monkeypatch.setattr(
        dense,
        "import_module",
        lambda module_name: SimpleNamespace(SentenceTransformer=factory),
    )
    retriever = DenseRetriever(model_id="model/id", revision="requested", batch_size=5)
    assert "factory" not in calls

    retriever.index([_chunk("document", "document")])

    assert calls["factory"] == {"model_id": "model/id", "revision": "requested"}
    assert calls["encode"] == {
        "batch_size": 5,
        "convert_to_numpy": True,
        "normalize_embeddings": False,
        "prompt": "",
        "sentences": ("passage: document",),
        "show_progress_bar": False,
    }
    assert retriever.settings["resolved_revision"] == "d" * 40


def test_resolved_revision_reads_nested_commit_metadata() -> None:
    commit = "a" * 40
    tokenizer_commit = "b" * 64
    model = SimpleNamespace(
        revision="main",
        config={"revision": "release", "_commit_hash": commit},
        tokenizer=SimpleNamespace(init_kwargs={"_commit_hash": tokenizer_commit}),
    )

    assert dense._read_resolved_revision(model, requested_revision=None) == commit

    attr_model = SimpleNamespace(config=SimpleNamespace(commit_hash=tokenizer_commit))
    assert (
        dense._read_resolved_revision(attr_model, requested_revision=None)
        == tokenizer_commit
    )


def test_resolved_revision_ignores_invalid_values_and_uses_requested_commit() -> None:
    requested_commit = "c" * 40
    model = SimpleNamespace(
        resolved_revision="main",
        config=SimpleNamespace(_commit_hash=123, revision="tag"),
        tokenizer=SimpleNamespace(
            init_kwargs={"_commit_hash": object(), "revision": "v1"}
        ),
    )

    assert (
        dense._read_resolved_revision(model, requested_revision=requested_commit)
        == requested_commit
    )
    assert dense._read_resolved_revision(model, requested_revision="main") is None


def test_settings_have_all_dense_reproducibility_values() -> None:
    backend = FakeEmbeddingBackend({}, requested_revision=None, resolved_revision=None)
    retriever = DenseRetriever(
        backend=backend,
        normalize_embeddings=False,
        batch_size=9,
        query_prompt="question: ",
        document_prompt="document: ",
    )

    assert retriever.settings == {
        "backend": {
            "backend_type": (
                f"{FakeEmbeddingBackend.__module__}.{FakeEmbeddingBackend.__qualname__}"
            ),
            "cache_identity_sha256": retriever.settings["backend"][
                "cache_identity_sha256"
            ],
        },
        "batch_size": 9,
        "document_prompt": "document: ",
        "model_id": "fake/embedding-model",
        "name": "dense",
        "normalize_embeddings": False,
        "query_prompt": "question: ",
        "requested_revision": None,
        "resolved_revision": None,
        "similarity": "inner_product",
        "type": "dense",
    }


def test_custom_backend_settings_require_stable_cache_identity() -> None:
    class BackendWithoutIdentity:
        metadata = EmbeddingModelMetadata(model_id="custom")

        def encode(
            self,
            texts: Sequence[str],
            *,
            batch_size: int,
        ) -> Sequence[Sequence[float]]:
            return [[1.0] for _ in texts]

    retriever = DenseRetriever(backend=BackendWithoutIdentity())

    with pytest.raises(RetrieverContractError, match="cache_identity"):
        _ = retriever.settings


def test_custom_backend_identity_is_hashed_and_changes_settings() -> None:
    first_backend = FakeEmbeddingBackend({})
    second_backend = FakeEmbeddingBackend({})
    first_backend.cache_identity = {"configuration": "first", "token": "secret"}
    second_backend.cache_identity = {"configuration": "second", "token": "secret"}

    first_settings = DenseRetriever(backend=first_backend).settings
    second_settings = DenseRetriever(backend=second_backend).settings

    assert first_settings != second_settings
    serialized = repr(first_settings)
    assert "configuration" not in serialized
    assert "secret" not in serialized
