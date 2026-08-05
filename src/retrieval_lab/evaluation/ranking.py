"""Deterministic validation and normalization of retriever rankings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

from retrieval_lab.domain import SearchResult
from retrieval_lab.exceptions import RetrieverContractError


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise RetrieverContractError("top_k must be a positive integer")


def validate_search_results(
    results: Sequence[SearchResult], *, top_k: int
) -> tuple[SearchResult, ...]:
    """Validate one retriever response and return an immutable shallow copy.

    Duplicate chunks are rejected rather than silently deduplicated. A retriever
    must return no more than ``top_k`` items with contiguous one-based ranks.
    Ordering by score is normalized separately by :func:`stable_rank_results`.
    """

    _validate_top_k(top_k)
    normalized = tuple(results)
    if len(normalized) > top_k:
        raise RetrieverContractError(
            f"retriever returned {len(normalized)} results for top_k={top_k}"
        )

    seen_chunk_ids: set[str] = set()
    for expected_rank, result in enumerate(normalized, start=1):
        if not isinstance(result.chunk_id, str) or not result.chunk_id:
            raise RetrieverContractError("search result chunk_id must not be empty")
        if not isinstance(result.document_id, str) or not result.document_id:
            raise RetrieverContractError("search result document_id must not be empty")
        if (
            isinstance(result.score, bool)
            or not isinstance(result.score, int | float)
            or not math.isfinite(result.score)
        ):
            raise RetrieverContractError("search result score must be finite")
        if (
            isinstance(result.rank, bool)
            or not isinstance(result.rank, int)
            or result.rank != expected_rank
        ):
            raise RetrieverContractError(
                "search result ranks must be contiguous and one-based; "
                f"expected {expected_rank}, got {result.rank}"
            )
        if result.chunk_id in seen_chunk_ids:
            raise RetrieverContractError(
                f"duplicate search result chunk_id: {result.chunk_id!r}"
            )
        seen_chunk_ids.add(result.chunk_id)
    return normalized


def stable_rank_results(
    results: Sequence[SearchResult], *, top_k: int
) -> tuple[SearchResult, ...]:
    """Validate, deterministically sort, and recompute ranks for search results."""

    validated = validate_search_results(results, top_k=top_k)
    ordered = sorted(validated, key=lambda result: (-result.score, result.chunk_id))
    return tuple(
        replace(result, rank=rank) for rank, result in enumerate(ordered, start=1)
    )


def collapse_to_documents(
    results: Sequence[SearchResult],
) -> tuple[SearchResult, ...]:
    """Keep the first result for each document and recompute contiguous ranks."""

    collapsed: list[SearchResult] = []
    seen_document_ids: set[str] = set()
    for result in results:
        if result.document_id in seen_document_ids:
            continue
        seen_document_ids.add(result.document_id)
        collapsed.append(replace(result, rank=len(collapsed) + 1))
    return tuple(collapsed)
