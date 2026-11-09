from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import pytest

from retrieval_lab import BM25Retriever, Chunk, HybridRetriever, SearchResult
from retrieval_lab.exceptions import RetrieverContractError
from retrieval_lab.retrievers import BaseRetriever


def _chunk(identifier: str, text: str = "text") -> Chunk:
    return Chunk(
        id=identifier,
        document_id=f"doc-{identifier}",
        text=text,
        start_offset=0,
        end_offset=len(text),
    )


def _result(identifier: str, rank: int, *, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk_id=identifier,
        document_id=f"doc-{identifier}",
        text=f"text-{identifier}",
        score=score,
        rank=rank,
    )


class _FakeRetriever(BaseRetriever):
    def __init__(
        self,
        strategy_name: str,
        results: Sequence[SearchResult] = (),
        *,
        settings: Mapping[str, object] | None = None,
    ) -> None:
        self._name = strategy_name
        self._results = tuple(results)
        self._settings = {} if settings is None else dict(settings)
        self.indexed_chunks: Sequence[Chunk] | None = None
        self.search_calls: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def settings(self) -> Mapping[str, object]:
        return self._settings

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.indexed_chunks = chunks

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.search_calls.append((query, top_k))
        return list(self._results[:top_k])


def test_rrf_golden_scores_and_stable_tie_break() -> None:
    first = _FakeRetriever(
        "first",
        [_result("a", 1, score=100.0), _result("b", 2), _result("c", 3)],
    )
    second = _FakeRetriever(
        "second",
        [_result("b", 1, score=-100.0), _result("c", 2), _result("d", 3)],
    )
    retriever = HybridRetriever([first, second], rrf_k=1, candidate_k=4)
    retriever.index([_chunk("a"), _chunk("b")])

    results = retriever.search("query", top_k=4)

    assert [result.chunk_id for result in results] == ["b", "c", "a", "d"]
    assert [result.rank for result in results] == [1, 2, 3, 4]
    assert results[0].score == pytest.approx(1 / 3 + 1 / 2)
    assert results[1].score == pytest.approx(1 / 4 + 1 / 3)
    assert results[2].score == pytest.approx(1 / 2)
    assert results[3].score == pytest.approx(1 / 4)
    assert first.search_calls == [("query", 4)]
    assert second.search_calls == [("query", 4)]


def test_source_order_does_not_change_fused_results_or_settings() -> None:
    first = _FakeRetriever("first", [_result("same", 1)], settings={"z": 2, "a": 1})
    second = _FakeRetriever(
        "second",
        [_result("same", 1)],
        settings={"a": 1, "z": 2},
    )
    left = HybridRetriever([first, second], candidate_k=2)
    right = HybridRetriever([second, first], candidate_k=2)
    chunks = [_chunk("same")]
    left.index(chunks)
    right.index(chunks)

    assert left.search("query", top_k=1) == right.search("query", top_k=1)
    assert left.settings == right.settings
    assert left.settings == {
        "candidate_k": 2,
        "name": "hybrid",
        "rrf_k": 60,
        "sources": [
            {"name": "first", "settings": {"a": 1, "z": 2}},
            {"name": "second", "settings": {"a": 1, "z": 2}},
        ],
        "type": "hybrid",
    }


def test_candidates_present_in_only_one_source_are_retained() -> None:
    first = _FakeRetriever("first", [_result("only-first", 1)])
    second = _FakeRetriever("second", [_result("only-second", 1)])
    retriever = HybridRetriever([first, second], candidate_k=2)
    retriever.index([_chunk("only-first"), _chunk("only-second")])

    assert [item.chunk_id for item in retriever.search("query", top_k=2)] == [
        "only-first",
        "only-second",
    ]


def test_empty_sources_produce_empty_results() -> None:
    retriever = HybridRetriever(
        [_FakeRetriever("first"), _FakeRetriever("second")], candidate_k=2
    )
    retriever.index([])
    assert retriever.search("query", top_k=2) == []


def test_index_rejects_invalid_shared_chunks_before_sources_are_called() -> None:
    first = _FakeRetriever("first")
    second = _FakeRetriever("second")
    retriever = HybridRetriever([first, second])

    with pytest.raises(RetrieverContractError, match="Chunk"):
        retriever.index([cast(Chunk, object())])

    assert first.indexed_chunks is None
    assert second.indexed_chunks is None


@pytest.mark.parametrize(
    "sources, kwargs, match",
    [
        ([_FakeRetriever("only")], {}, "at least two"),
        ([_FakeRetriever("same"), _FakeRetriever("same")], {}, "unique"),
        (
            [cast(BaseRetriever, object()), _FakeRetriever("second")],
            {},
            "BaseRetriever",
        ),
        ([_FakeRetriever(" "), _FakeRetriever("second")], {}, "non-empty"),
        ([_FakeRetriever("first"), _FakeRetriever("second")], {"rrf_k": 0}, "rrf_k"),
        (
            [_FakeRetriever("first"), _FakeRetriever("second")],
            {"candidate_k": True},
            "candidate_k",
        ),
    ],
)
def test_constructor_rejects_invalid_arguments(
    sources: list[BaseRetriever], kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(RetrieverContractError, match=match):
        HybridRetriever(sources, **kwargs)  # type: ignore[arg-type]


def test_constructor_rejects_non_sequence_and_nested_hybrid() -> None:
    with pytest.raises(RetrieverContractError, match="sequence"):
        HybridRetriever(cast(Sequence[BaseRetriever], "bad"))

    with pytest.raises(RetrieverContractError, match="nested hybrid"):
        HybridRetriever(
            [
                HybridRetriever([_FakeRetriever("a"), _FakeRetriever("b")]),
                _FakeRetriever("c"),
            ]
        )


def test_search_rejects_cutoff_and_invalid_source_rankings() -> None:
    first = _FakeRetriever("first", [_result("a", 1)])
    second = _FakeRetriever("second", [_result("b", 1)])
    retriever = HybridRetriever([first, second], candidate_k=1)
    retriever.index([_chunk("a"), _chunk("b")])
    with pytest.raises(RetrieverContractError, match="candidate_k"):
        retriever.search("query", top_k=2)

    duplicate = _FakeRetriever("duplicate", [_result("a", 1), _result("a", 2)])
    invalid = HybridRetriever([duplicate, _FakeRetriever("other")], candidate_k=2)
    invalid.index([_chunk("a")])
    with pytest.raises(RetrieverContractError, match="duplicate"):
        invalid.search("query", top_k=1)


def test_search_rejects_conflicting_payloads_for_the_same_chunk() -> None:
    left = _FakeRetriever("left", [_result("shared", 1)])
    conflicting = SearchResult(
        chunk_id="shared",
        document_id="different-document",
        text="different text",
        score=1.0,
        rank=1,
    )
    right = _FakeRetriever("right", [conflicting])
    retriever = HybridRetriever([left, right], candidate_k=1)
    retriever.index([_chunk("shared")])

    with pytest.raises(RetrieverContractError, match="conflicting payloads"):
        retriever.search("query", top_k=1)


def test_hybrid_indexes_bm25_and_source_once() -> None:
    bm25 = BM25Retriever()
    dense_like = _FakeRetriever("dense", [_result("chunk-2", 1)])
    retriever = HybridRetriever([bm25, dense_like], candidate_k=2)
    chunks = [_chunk("chunk-1", "alpha"), _chunk("chunk-2", "beta")]
    retriever.index(chunks)

    assert bm25.search("alpha", 1)[0].chunk_id == "chunk-1"
    assert dense_like.indexed_chunks is chunks
    assert dense_like.search_calls == []
    assert retriever.search("alpha", top_k=1)[0].chunk_id == "chunk-1"


def test_search_requires_index_and_rejects_invalid_query() -> None:
    retriever = HybridRetriever([_FakeRetriever("first"), _FakeRetriever("second")])
    with pytest.raises(RetrieverContractError, match="not indexed"):
        retriever.search("query", top_k=1)
    retriever.index([])
    with pytest.raises(RetrieverContractError, match="query"):
        retriever.search(cast(str, object()), top_k=1)


def test_settings_reject_non_json_source_settings() -> None:
    retriever = HybridRetriever(
        [
            _FakeRetriever("first", settings={"bad": object()}),
            _FakeRetriever("second"),
        ]
    )
    with pytest.raises(RetrieverContractError, match="unsupported JSON"):
        _ = retriever.settings


class _CountingRetriever(_FakeRetriever):
    def __init__(
        self,
        strategy_name: str,
        results: Sequence[SearchResult] = (),
    ) -> None:
        super().__init__(strategy_name, results)
        self.index_calls = 0

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.index_calls += 1
        super().index(chunks)


class _RaisingNameRetriever(_FakeRetriever):
    @property
    def name(self) -> str:
        raise RuntimeError("name unavailable")


class _RaisingSettingsRetriever(_FakeRetriever):
    def __init__(self, strategy_name: str, error: Exception) -> None:
        super().__init__(strategy_name)
        self._error = error

    @property
    def settings(self) -> Mapping[str, object]:
        raise self._error


class _StaticSettingsRetriever(_FakeRetriever):
    def __init__(self, strategy_name: str, settings: object) -> None:
        super().__init__(strategy_name)
        self._static_settings = settings

    @property
    def settings(self) -> Mapping[str, object]:
        return cast("Mapping[str, object]", self._static_settings)


class _ExplodingMapping(Mapping):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def keys(self):
        raise RuntimeError("keys unavailable")


class _RawSearchRetriever(_FakeRetriever):
    def __init__(self, strategy_name: str, raw: object) -> None:
        super().__init__(strategy_name)
        self._raw = raw

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return cast("list[SearchResult]", self._raw)


class _RaisingSearchRetriever(_FakeRetriever):
    def search(self, query: str, top_k: int) -> list[SearchResult]:
        raise RuntimeError("backend down")


def test_constructor_wraps_source_name_failures() -> None:
    with pytest.raises(
        RetrieverContractError, match="could not provide a name"
    ) as captured:
        HybridRetriever([_RaisingNameRetriever("broken"), _FakeRetriever("second")])

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_settings_wrap_unreadable_source_settings() -> None:
    retriever = HybridRetriever(
        [
            _RaisingSettingsRetriever("first", RuntimeError("io failed")),
            _FakeRetriever("second"),
        ]
    )

    with pytest.raises(
        RetrieverContractError, match="settings could not be read"
    ) as captured:
        _ = retriever.settings

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_settings_propagate_retrieval_lab_errors_from_sources() -> None:
    retriever = HybridRetriever(
        [
            _RaisingSettingsRetriever(
                "first", RetrieverContractError("source settings failed")
            ),
            _FakeRetriever("second"),
        ]
    )

    with pytest.raises(RetrieverContractError, match=r"^source settings failed$"):
        _ = retriever.settings


def test_settings_wrap_mapping_failures_during_normalization() -> None:
    retriever = HybridRetriever(
        [
            _StaticSettingsRetriever("first", _ExplodingMapping()),
            _FakeRetriever("second"),
        ]
    )

    with pytest.raises(
        RetrieverContractError, match="settings must be JSON-compatible"
    ) as captured:
        _ = retriever.settings

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_settings_reject_non_mapping_source_settings() -> None:
    retriever = HybridRetriever(
        [
            _StaticSettingsRetriever("first", ["not", "a", "mapping"]),
            _FakeRetriever("second"),
        ]
    )

    with pytest.raises(RetrieverContractError, match="settings must be a mapping"):
        _ = retriever.settings


def test_settings_reject_non_string_keys() -> None:
    retriever = HybridRetriever(
        [
            _StaticSettingsRetriever("first", {1: "value"}),
            _FakeRetriever("second"),
        ]
    )

    with pytest.raises(RetrieverContractError, match="keys must be strings"):
        _ = retriever.settings


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_settings_reject_non_finite_numbers(value: float) -> None:
    retriever = HybridRetriever(
        [
            _StaticSettingsRetriever("first", {"score": value}),
            _FakeRetriever("second"),
        ]
    )

    with pytest.raises(RetrieverContractError, match="NaN or infinity"):
        _ = retriever.settings


def test_settings_normalize_nested_lists_and_mappings() -> None:
    retriever = HybridRetriever(
        [
            _StaticSettingsRetriever(
                "first", {"tags": ["a", "b"], "nested": {"y": 2, "x": 1}}
            ),
            _FakeRetriever("second"),
        ]
    )

    sources = retriever.settings["sources"]

    assert isinstance(sources, list)
    entry = sources[0]
    assert isinstance(entry, dict)
    settings = entry["settings"]
    assert isinstance(settings, dict)
    # Dict equality ignores order, so compare the key sequences explicitly:
    # normalization must sort keys at every mapping level.
    assert list(settings) == ["nested", "tags"]
    assert settings["tags"] == ["a", "b"]
    nested = settings["nested"]
    assert isinstance(nested, dict)
    assert list(nested) == ["x", "y"]
    assert nested == {"x": 1, "y": 2}


def test_reindexing_the_same_chunks_does_not_call_sources_again() -> None:
    first = _CountingRetriever("first")
    second = _CountingRetriever("second")
    retriever = HybridRetriever([first, second])
    chunks = [_chunk("a"), _chunk("b")]

    retriever.index(chunks)
    retriever.index(list(chunks))

    assert first.index_calls == 1
    assert second.index_calls == 1

    retriever.index([_chunk("c")])

    assert first.index_calls == 2
    assert second.index_calls == 2


def test_index_sources_once_records_and_skips_shared_sources() -> None:
    first = _CountingRetriever("first")
    second = _CountingRetriever("second")
    retriever = HybridRetriever([first, second])
    chunks = [_chunk("a")]
    indexed_ids: set[int] = set()

    retriever._index_sources_once(chunks, indexed_ids)

    assert first.index_calls == 1
    assert second.index_calls == 1
    assert indexed_ids == {id(first), id(second)}

    # A repeated call with the same chunks early-returns: no source is
    # indexed again, but the source ids are still recorded for the new
    # caller's set so it does not index them separately either.
    second_pass: set[int] = set()
    retriever._index_sources_once(chunks, second_pass)

    assert first.index_calls == 1
    assert second.index_calls == 1
    assert second_pass == {id(first), id(second)}


def test_index_sources_once_skips_sources_the_caller_already_indexed() -> None:
    first = _CountingRetriever("first")
    second = _CountingRetriever("second")
    retriever = HybridRetriever([first, second])

    retriever._index_sources_once([_chunk("a")], {id(first)})

    assert first.index_calls == 0
    assert second.index_calls == 1


@pytest.mark.parametrize("top_k", [0, -3, True, 2.5])
def test_search_rejects_invalid_top_k(top_k: object) -> None:
    retriever = HybridRetriever([_FakeRetriever("first"), _FakeRetriever("second")])
    retriever.index([_chunk("a")])

    with pytest.raises(RetrieverContractError, match="top_k must be a positive"):
        retriever.search("query", top_k)  # type: ignore[arg-type]


def test_search_rejects_non_sequence_source_ranking() -> None:
    retriever = HybridRetriever(
        [_RawSearchRetriever("first", "rankings"), _FakeRetriever("second")]
    )
    retriever.index([_chunk("a")])

    with pytest.raises(RetrieverContractError, match="must return a sequence"):
        retriever.search("query", top_k=1)


def test_search_rejects_non_result_source_items() -> None:
    retriever = HybridRetriever(
        [_RawSearchRetriever("first", [object()]), _FakeRetriever("second")]
    )
    retriever.index([_chunk("a")])

    with pytest.raises(RetrieverContractError, match="must return a sequence"):
        retriever.search("query", top_k=1)


def test_search_wraps_source_exceptions() -> None:
    retriever = HybridRetriever(
        [_RaisingSearchRetriever("first"), _FakeRetriever("second")]
    )
    retriever.index([_chunk("a")])

    with pytest.raises(
        RetrieverContractError, match="returned an invalid ranking"
    ) as captured:
        retriever.search("query", top_k=1)

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_index_rejects_non_sequence_chunks() -> None:
    first = _CountingRetriever("first")
    second = _CountingRetriever("second")
    retriever = HybridRetriever([first, second])

    with pytest.raises(RetrieverContractError, match="must be a sequence"):
        retriever.index(cast("Sequence[Chunk]", "chunks"))

    assert first.index_calls == 0
    assert second.index_calls == 0


def test_index_rejects_duplicate_chunk_identifiers() -> None:
    first = _CountingRetriever("first")
    second = _CountingRetriever("second")
    retriever = HybridRetriever([first, second])

    with pytest.raises(RetrieverContractError, match="duplicate chunk identifier"):
        retriever.index([_chunk("a"), _chunk("a")])

    assert first.index_calls == 0
    assert second.index_calls == 0
