"""End-to-end dense retrieval tests against a real embedding model.

The unit tests drive :class:`DenseRetriever` through fake backends, so the
lazy sentence-transformers adapter itself is only exercised here.  The whole
module is skipped when the optional ``dense`` extra is not installed; when it
is installed, the first run downloads the model (~90 MB) and later runs reuse
the Hugging Face cache.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip(
    "sentence_transformers",
    reason="dense extra is not installed",
)

from retrieval_lab import DenseRetriever
from retrieval_lab.models import Chunk

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=f"document-{chunk_id}",
        text=text,
        start_offset=0,
        end_offset=len(text),
        metadata={"chunk": chunk_id},
    )


@pytest.fixture(scope="module")
def retriever() -> DenseRetriever:
    indexed = DenseRetriever(
        model_id=MODEL_ID,
        query_prompt="",
        document_prompt="",
        batch_size=2,
    )
    indexed.index(
        [
            _chunk(
                "feline",
                "The cat is a small domesticated feline mammal kept as a pet.",
            ),
            _chunk(
                "canine",
                "The dog is a domesticated canine mammal bred as a companion animal.",
            ),
            _chunk(
                "market",
                "Bond yields rose as investors weighed inflation data.",
            ),
            _chunk(
                "recipe",
                "Simmer the tomato sauce with garlic and basil for twenty minutes.",
            ),
        ]
    )
    return indexed


def test_real_model_ranks_the_matching_topic_first(
    retriever: DenseRetriever,
) -> None:
    results = retriever.search("a small domestic cat", top_k=4)

    assert [item.chunk_id for item in results] == [
        "feline",
        "canine",
        "recipe",
        "market",
    ]


def test_real_model_results_follow_the_search_contract(
    retriever: DenseRetriever,
) -> None:
    results = retriever.search("feline pet", top_k=2)

    assert len(results) == 2
    assert [item.rank for item in results] == [1, 2]
    scores = [item.score for item in results]
    assert scores == sorted(scores, reverse=True)
    assert all(math.isfinite(score) and -1.0 <= score <= 1.0 for score in scores)
    first = results[0]
    assert first.chunk_id == "feline"
    assert first.document_id == "document-feline"
    assert first.text.startswith("The cat is a small domesticated feline")
    assert first.metadata == {"chunk": "feline"}


def test_real_model_search_is_deterministic(retriever: DenseRetriever) -> None:
    first = retriever.search("domesticated feline", top_k=4)
    second = retriever.search("domesticated feline", top_k=4)

    assert [(item.chunk_id, item.score, item.rank) for item in first] == [
        (item.chunk_id, item.score, item.rank) for item in second
    ]


def test_real_model_settings_report_identity_and_resolved_revision(
    retriever: DenseRetriever,
) -> None:
    settings = retriever.settings

    assert settings["model_id"] == MODEL_ID
    assert settings["requested_revision"] is None
    resolved = settings["resolved_revision"]
    assert isinstance(resolved, str)
    assert len(resolved) == 40
    assert all(character in "0123456789abcdef" for character in resolved)
