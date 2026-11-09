"""Deterministic RAG quality metrics: faithfulness and context quality.

The formulas follow the RAGAS definitions, computed from structured
inputs the caller supplies: which context items were relevant, and
which answer claims are supported by the context. Claim extraction
and support judgments stay caller-controlled (an LLM judge from
:mod:`retrieval_lab.judging` can produce them); this module only does
the arithmetic, so every score is reproducible offline.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from retrieval_lab.exceptions import EvaluationError

__all__ = [
    "ClaimAssessment",
    "RagEvaluation",
    "context_precision",
    "context_recall",
    "evaluate_rag_answer",
    "faithfulness",
]


@dataclass(frozen=True)
class ClaimAssessment:
    """One answer claim and whether the retrieved context supports it."""

    claim: str
    supported: bool

    def __post_init__(self) -> None:
        if not isinstance(self.claim, str) or not self.claim.strip():
            raise EvaluationError("claim must be a non-empty string")
        if not isinstance(self.supported, bool):
            raise EvaluationError("supported must be a boolean")


def faithfulness(claims: Sequence[ClaimAssessment]) -> float:
    """Return the fraction of assessed claims supported by the context."""

    assessed = _validated_claims(claims)
    supported = sum(1 for item in assessed if item.supported)
    return supported / len(assessed)


def context_precision(
    context_ids: Sequence[str], relevant_ids: Collection[str]
) -> float:
    """Return rank-weighted context precision over the full context list.

    Following the RAGAS context precision definition, each rank
    contributes precision-at-that-rank when its item is relevant, and
    the sum is normalized by the number of relevant items present, so
    ranking relevant items earlier scores higher.
    """

    context, relevant = _validated_context(context_ids, relevant_ids)
    hits = 0
    weighted = 0.0
    for rank, identifier in enumerate(context, start=1):
        if identifier in relevant:
            hits += 1
            weighted += hits / rank
    if hits == 0:
        return 0.0
    return weighted / hits


def context_recall(context_ids: Sequence[str], relevant_ids: Collection[str]) -> float:
    """Return the fraction of relevant items present in the context list."""

    context, relevant = _validated_context(context_ids, relevant_ids)
    hits = sum(1 for identifier in context if identifier in relevant)
    return hits / len(relevant)


@dataclass(frozen=True)
class RagEvaluation:
    """RAG quality measurements for one answered query."""

    query_id: str
    faithfulness: float | None
    context_precision: float
    context_recall: float

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise EvaluationError("query_id must be a non-empty string")
        _validate_score(self.faithfulness, "faithfulness", allow_none=True)
        _validate_score(self.context_precision, "context_precision", allow_none=False)
        _validate_score(self.context_recall, "context_recall", allow_none=False)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible per-query RAG evaluation record."""

        return {
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "query_id": self.query_id,
        }


def evaluate_rag_answer(
    query_id: str,
    *,
    context_ids: Sequence[str],
    relevant_ids: Collection[str],
    claims: Sequence[ClaimAssessment] = (),
) -> RagEvaluation:
    """Evaluate one RAG answer's context quality and claim faithfulness.

    ``faithfulness`` is ``None`` when no claim assessments are given,
    mirroring how answer evaluation reports ``None`` for metrics whose
    inputs were not supplied.
    """

    if not isinstance(query_id, str) or not query_id.strip():
        raise EvaluationError("query_id must be a non-empty string")
    assessed = _validated_claims(claims, allow_empty=True)
    return RagEvaluation(
        query_id=query_id,
        faithfulness=faithfulness(assessed) if assessed else None,
        context_precision=context_precision(context_ids, relevant_ids),
        context_recall=context_recall(context_ids, relevant_ids),
    )


def _validated_claims(
    claims: Sequence[ClaimAssessment], *, allow_empty: bool = False
) -> tuple[ClaimAssessment, ...]:
    if isinstance(claims, (str, bytes)) or not isinstance(claims, Sequence):
        raise EvaluationError("claims must be a sequence of ClaimAssessment values")
    assessed = tuple(claims)
    if not assessed and not allow_empty:
        raise EvaluationError("at least one claim assessment is required")
    for position, item in enumerate(assessed):
        if not isinstance(item, ClaimAssessment):
            raise EvaluationError(f"claims[{position}] must be a ClaimAssessment")
    return assessed


def _validated_context(
    context_ids: Sequence[str], relevant_ids: Collection[str]
) -> tuple[tuple[str, ...], frozenset[str]]:
    if isinstance(context_ids, (str, bytes)) or not isinstance(context_ids, Sequence):
        raise EvaluationError("context_ids must be a sequence of document IDs")
    context = tuple(context_ids)
    seen: set[str] = set()
    for position, identifier in enumerate(context):
        if not isinstance(identifier, str) or not identifier.strip():
            raise EvaluationError(f"context_ids[{position}] must be a non-empty string")
        if identifier in seen:
            raise EvaluationError("context identifiers must be unique")
        seen.add(identifier)
    if isinstance(relevant_ids, (str, bytes)) or not isinstance(
        relevant_ids, Collection
    ):
        raise EvaluationError("relevant_ids must be a collection of document IDs")
    for identifier in relevant_ids:
        if not isinstance(identifier, str) or not identifier.strip():
            raise EvaluationError("relevant identifiers must be non-empty strings")
    relevant = frozenset(relevant_ids)
    if not relevant:
        raise EvaluationError("relevant identifiers must not be empty")
    return context, relevant


def _validate_score(value: object, name: str, *, allow_none: bool) -> None:
    if value is None and allow_none:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise EvaluationError(f"{name} must be a number between 0.0 and 1.0")
