"""Deterministic rule-based answer and citation evaluation."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence, Set as AbstractSet
from dataclasses import dataclass
from typing import Literal

from retrieval_lab.exceptions import EvaluationError
from retrieval_lab.generation import GenerationOutput, GenerationRequest

AnswerFailureType = Literal[
    "success",
    "retrieval_failure",
    "generation_failure",
    "citation_failure",
    "unscored",
]


@dataclass(frozen=True, init=False)
class AnswerReference:
    """Ground-truth information used by deterministic answer evaluation."""

    query_id: str
    relevant_document_ids: frozenset[str]
    reference_answer: str | None
    required_facts: tuple[str, ...]
    answerable: bool

    def __init__(
        self,
        query_id: str,
        *,
        relevant_document_ids: AbstractSet[str] = frozenset(),
        reference_answer: str | None = None,
        required_facts: Sequence[str] = (),
        answerable: bool = True,
    ) -> None:
        if not isinstance(query_id, str) or not query_id.strip():
            raise EvaluationError("AnswerReference.query_id must be non-empty")
        if isinstance(relevant_document_ids, (str, bytes)) or not isinstance(
            relevant_document_ids, AbstractSet
        ):
            raise EvaluationError(
                "AnswerReference.relevant_document_ids must be a set of IDs"
            )
        normalized_ids: set[str] = set()
        for identifier in relevant_document_ids:
            if not isinstance(identifier, str) or not identifier.strip():
                raise EvaluationError(
                    "AnswerReference relevant document IDs must be non-empty"
                )
            normalized_ids.add(unicodedata.normalize("NFC", identifier))
        if reference_answer is not None and (
            not isinstance(reference_answer, str) or not reference_answer.strip()
        ):
            raise EvaluationError(
                "AnswerReference.reference_answer must be non-empty or None"
            )
        if isinstance(required_facts, (str, bytes)) or not isinstance(
            required_facts, Sequence
        ):
            raise EvaluationError(
                "AnswerReference.required_facts must be a sequence of strings"
            )
        facts: list[str] = []
        seen_facts: set[str] = set()
        for position, fact in enumerate(required_facts):
            if not isinstance(fact, str) or not fact.strip():
                raise EvaluationError(
                    f"AnswerReference.required_facts[{position}] must be non-empty"
                )
            normalized = unicodedata.normalize("NFC", fact)
            key = _normalize_text(normalized)
            if key in seen_facts:
                raise EvaluationError(
                    f"AnswerReference required facts contain duplicate {normalized!r}"
                )
            seen_facts.add(key)
            facts.append(normalized)
        if not isinstance(answerable, bool):
            raise EvaluationError("AnswerReference.answerable must be a bool")
        object.__setattr__(self, "query_id", unicodedata.normalize("NFC", query_id))
        object.__setattr__(
            self,
            "relevant_document_ids",
            frozenset(normalized_ids),
        )
        object.__setattr__(
            self,
            "reference_answer",
            None
            if reference_answer is None
            else unicodedata.normalize("NFC", reference_answer),
        )
        object.__setattr__(self, "required_facts", tuple(facts))
        object.__setattr__(self, "answerable", answerable)


@dataclass(frozen=True)
class AnswerEvaluation:
    """Rule-based answer-quality measurements for one generated answer."""

    query_id: str
    exact_match: float | None
    token_f1: float | None
    citation_presence: float
    citation_validity: float
    citation_precision: float
    citation_recall: float
    citation_coverage: float
    empty_answer: bool
    required_fact_coverage: float | None
    latency_ms: float
    token_usage: int | None
    estimated_cost: float | None
    failure_type: AnswerFailureType

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible per-query evaluation record."""

        return {
            "citation_coverage": self.citation_coverage,
            "citation_precision": self.citation_precision,
            "citation_presence": self.citation_presence,
            "citation_recall": self.citation_recall,
            "citation_validity": self.citation_validity,
            "empty_answer": self.empty_answer,
            "estimated_cost": self.estimated_cost,
            "exact_match": self.exact_match,
            "failure_type": self.failure_type,
            "latency_ms": self.latency_ms,
            "query_id": self.query_id,
            "required_fact_coverage": self.required_fact_coverage,
            "token_f1": self.token_f1,
            "token_usage": self.token_usage,
        }


@dataclass(frozen=True)
class AnswerEvaluationResult:
    """Aggregate answer-evaluation result with per-query evidence."""

    evaluations: tuple[AnswerEvaluation, ...]

    def __post_init__(self) -> None:
        if not self.evaluations:
            raise EvaluationError("answer evaluations must not be empty")
        seen: set[str] = set()
        for position, evaluation in enumerate(self.evaluations):
            if not isinstance(evaluation, AnswerEvaluation):
                raise EvaluationError(
                    f"evaluations[{position}] must be an AnswerEvaluation"
                )
            if evaluation.query_id in seen:
                raise EvaluationError(
                    "answer evaluation query IDs must be unique; duplicate "
                    f"{evaluation.query_id!r}"
                )
            seen.add(evaluation.query_id)

    @property
    def mean_exact_match(self) -> float | None:
        """Return mean Exact Match over queries with reference answers."""

        return _mean_optional(item.exact_match for item in self.evaluations)

    @property
    def mean_token_f1(self) -> float | None:
        """Return mean Token F1 over queries with reference answers."""

        return _mean_optional(item.token_f1 for item in self.evaluations)

    @property
    def mean_required_fact_coverage(self) -> float | None:
        """Return mean required-fact coverage when facts were supplied."""

        return _mean_optional(
            item.required_fact_coverage for item in self.evaluations
        )

    @property
    def empty_answer_rate(self) -> float:
        """Return the share of generated answers that are empty after trimming."""

        return sum(item.empty_answer for item in self.evaluations) / len(
            self.evaluations
        )

    @property
    def total_token_usage(self) -> int | None:
        """Return total token usage when every answer reports usage."""

        values = [item.token_usage for item in self.evaluations]
        if any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)

    @property
    def total_estimated_cost(self) -> float | None:
        """Return total estimated cost when every answer reports a cost."""

        values = [item.estimated_cost for item in self.evaluations]
        if any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)

    def to_dict(self) -> dict[str, object]:
        """Return aggregate metrics and per-query evidence."""

        return {
            "empty_answer_rate": self.empty_answer_rate,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "mean_citation_coverage": _mean(
                item.citation_coverage for item in self.evaluations
            ),
            "mean_citation_precision": _mean(
                item.citation_precision for item in self.evaluations
            ),
            "mean_citation_presence": _mean(
                item.citation_presence for item in self.evaluations
            ),
            "mean_citation_recall": _mean(
                item.citation_recall for item in self.evaluations
            ),
            "mean_citation_validity": _mean(
                item.citation_validity for item in self.evaluations
            ),
            "mean_exact_match": self.mean_exact_match,
            "mean_generation_latency_ms": _mean(
                item.latency_ms for item in self.evaluations
            ),
            "mean_required_fact_coverage": self.mean_required_fact_coverage,
            "mean_token_f1": self.mean_token_f1,
            "query_count": len(self.evaluations),
            "total_estimated_cost": self.total_estimated_cost,
            "total_token_usage": self.total_token_usage,
        }


def exact_match(answer: str, reference: str) -> float:
    """Return normalized Exact Match as 0.0 or 1.0."""

    _require_text(answer, "answer")
    _require_text(reference, "reference", allow_empty=False)
    return float(_normalize_text(answer) == _normalize_text(reference))


def token_f1(answer: str, reference: str) -> float:
    """Return multiset Token F1 using deterministic multilingual tokenization."""

    _require_text(answer, "answer")
    _require_text(reference, "reference", allow_empty=False)
    answer_tokens = _answer_tokens(answer)
    reference_tokens = _answer_tokens(reference)
    if not answer_tokens:
        return 0.0
    common = sum((Counter(answer_tokens) & Counter(reference_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(answer_tokens)
    recall = common / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def evaluate_answer(
    reference: AnswerReference,
    request: GenerationRequest,
    output: GenerationOutput,
) -> AnswerEvaluation:
    """Evaluate one generated answer with deterministic rules and citations."""

    if not isinstance(reference, AnswerReference):
        raise EvaluationError("reference must be an AnswerReference")
    if not isinstance(request, GenerationRequest):
        raise EvaluationError("request must be a GenerationRequest")
    if not isinstance(output, GenerationOutput):
        raise EvaluationError("output must be a GenerationOutput")
    if reference.query_id != request.query_id:
        raise EvaluationError("reference and request query IDs must match")

    answer = output.answer
    context_document_ids = frozenset(
        context.document_id for context in request.contexts
    )
    citation_ids = frozenset(answer.citation_ids)
    relevant_ids = reference.relevant_document_ids
    valid_citations = citation_ids & context_document_ids
    relevant_citations = citation_ids & relevant_ids
    relevant_context_ids = relevant_ids & context_document_ids

    presence = float(bool(citation_ids))
    validity = _ratio(len(valid_citations), len(citation_ids))
    precision = _ratio(len(relevant_citations), len(citation_ids))
    recall = _ratio(len(relevant_citations), len(relevant_ids))
    coverage = _ratio(
        len(citation_ids & relevant_context_ids),
        len(relevant_context_ids),
    )

    exact: float | None = None
    f1: float | None = None
    if reference.reference_answer is not None:
        exact = exact_match(answer.text, reference.reference_answer)
        f1 = token_f1(answer.text, reference.reference_answer)

    fact_coverage: float | None = None
    if reference.required_facts:
        normalized_answer = _normalize_text(answer.text)
        matched = sum(
            _normalize_text(fact) in normalized_answer
            for fact in reference.required_facts
        )
        fact_coverage = matched / len(reference.required_facts)

    failure_type = _classify_failure(
        reference=reference,
        answer_text=answer.text,
        context_document_ids=context_document_ids,
        exact=exact,
        fact_coverage=fact_coverage,
        citation_validity=validity,
        citation_recall=recall,
    )
    return AnswerEvaluation(
        query_id=reference.query_id,
        exact_match=exact,
        token_f1=f1,
        citation_presence=presence,
        citation_validity=validity,
        citation_precision=precision,
        citation_recall=recall,
        citation_coverage=coverage,
        empty_answer=not bool(answer.text.strip()),
        required_fact_coverage=fact_coverage,
        latency_ms=output.latency_ms,
        token_usage=answer.token_usage,
        estimated_cost=answer.estimated_cost,
        failure_type=failure_type,
    )


def evaluate_answers(
    references: Sequence[AnswerReference],
    requests: Sequence[GenerationRequest],
    outputs: Sequence[GenerationOutput],
) -> AnswerEvaluationResult:
    """Evaluate a batch while preserving input order and query identity."""

    for name, values in (
        ("references", references),
        ("requests", requests),
        ("outputs", outputs),
    ):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise EvaluationError(f"{name} must be a sequence")
    if not references:
        raise EvaluationError("answer evaluation batch must not be empty")
    if not (len(references) == len(requests) == len(outputs)):
        raise EvaluationError(
            "references, requests, and outputs must have the same length"
        )
    evaluations = tuple(
        evaluate_answer(reference, request, output)
        for reference, request, output in zip(
            references,
            requests,
            outputs,
            strict=True,
        )
    )
    return AnswerEvaluationResult(evaluations)


def _classify_failure(
    *,
    reference: AnswerReference,
    answer_text: str,
    context_document_ids: frozenset[str],
    exact: float | None,
    fact_coverage: float | None,
    citation_validity: float,
    citation_recall: float,
) -> AnswerFailureType:
    if reference.relevant_document_ids and not (
        reference.relevant_document_ids & context_document_ids
    ):
        return "retrieval_failure"

    correctness: bool | None
    if not reference.answerable:
        correctness = not bool(answer_text.strip())
    elif fact_coverage is not None:
        correctness = fact_coverage == 1.0
    elif exact is not None:
        correctness = exact == 1.0
    else:
        correctness = None

    if correctness is None:
        return "unscored"
    if not correctness:
        return "generation_failure"
    if reference.relevant_document_ids and (
        citation_validity < 1.0 or citation_recall < 1.0
    ):
        return "citation_failure"
    return "success"


def _answer_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            tokens.append("".join(buffer))
            buffer.clear()

    for character in normalized:
        if character.isspace():
            flush()
        elif _is_cjk_or_kana(character):
            flush()
            tokens.append(character)
        elif character.isalnum() or character == "_":
            buffer.append(character)
        else:
            flush()
            tokens.append(character)
    flush()
    return tuple(tokens)


def _is_cjk_or_kana(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _require_text(value: object, name: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise EvaluationError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise EvaluationError(f"{name} must be non-empty")
    return value


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _mean(values: Iterable[float]) -> float:
    normalized = tuple(values)
    if not normalized:
        raise EvaluationError("cannot calculate mean of empty values")
    if any(not math.isfinite(value) for value in normalized):
        raise EvaluationError("mean values must be finite")
    return sum(normalized) / len(normalized)


def _mean_optional(values: Iterable[float | None]) -> float | None:
    normalized = tuple(value for value in values if value is not None)
    if not normalized:
        return None
    return _mean(normalized)


__all__ = [
    "AnswerEvaluation",
    "AnswerEvaluationResult",
    "AnswerFailureType",
    "AnswerReference",
    "evaluate_answer",
    "evaluate_answers",
    "exact_match",
    "token_f1",
]
