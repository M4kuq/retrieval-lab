"""Provider-independent answer generation contracts for Retrieval Lab."""

from __future__ import annotations

import math
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from retrieval_lab.domain._validation import normalize_json_mapping
from retrieval_lab.domain.json_types import JSONValue
from retrieval_lab.exceptions import EvaluationError

Clock = Callable[[], float]


@dataclass(frozen=True)
class GenerationContext:
    """One retrieved context item supplied to an answer generator."""

    document_id: str
    chunk_id: str
    text: str
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("document_id", "chunk_id", "text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EvaluationError(f"GenerationContext.{name} must be non-empty")
            object.__setattr__(self, name, unicodedata.normalize("NFC", value))
        object.__setattr__(
            self,
            "metadata",
            normalize_json_mapping(
                self.metadata,
                field_name=f"GenerationContext[{self.chunk_id!r}].metadata",
                error_type=EvaluationError,
            ),
        )


@dataclass(frozen=True, init=False)
class GenerationRequest:
    """A query plus an ordered context list for answer generation."""

    query_id: str
    query: str
    contexts: tuple[GenerationContext, ...]

    def __init__(
        self,
        query_id: str,
        query: str,
        contexts: Sequence[GenerationContext] = (),
    ) -> None:
        for name, value in (("query_id", query_id), ("query", query)):
            if not isinstance(value, str) or not value.strip():
                raise EvaluationError(f"GenerationRequest.{name} must be non-empty")
        if isinstance(contexts, (str, bytes)) or not isinstance(contexts, Sequence):
            raise EvaluationError(
                "GenerationRequest.contexts must be a sequence of GenerationContext"
            )
        normalized_contexts = tuple(contexts)
        seen_chunks: set[str] = set()
        for position, context in enumerate(normalized_contexts):
            if not isinstance(context, GenerationContext):
                raise EvaluationError(
                    f"GenerationRequest.contexts[{position}] must be GenerationContext"
                )
            if context.chunk_id in seen_chunks:
                raise EvaluationError(
                    f"GenerationRequest contexts contain duplicate chunk ID "
                    f"{context.chunk_id!r}"
                )
            seen_chunks.add(context.chunk_id)
        object.__setattr__(self, "query_id", unicodedata.normalize("NFC", query_id))
        object.__setattr__(self, "query", unicodedata.normalize("NFC", query))
        object.__setattr__(self, "contexts", normalized_contexts)


@dataclass(frozen=True, init=False)
class GeneratedAnswer:
    """An answer and explicit document citations returned by a generator."""

    text: str
    citation_ids: tuple[str, ...]
    token_usage: int | None
    estimated_cost: float | None
    metadata: Mapping[str, JSONValue]

    def __init__(
        self,
        text: str,
        *,
        citation_ids: Sequence[str] = (),
        token_usage: int | None = None,
        estimated_cost: float | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> None:
        if not isinstance(text, str):
            raise EvaluationError("GeneratedAnswer.text must be a string")
        if isinstance(citation_ids, (str, bytes)) or not isinstance(
            citation_ids, Sequence
        ):
            raise EvaluationError(
                "GeneratedAnswer.citation_ids must be a sequence of document IDs"
            )
        normalized_citations: list[str] = []
        seen: set[str] = set()
        for position, citation_id in enumerate(citation_ids):
            if not isinstance(citation_id, str) or not citation_id.strip():
                raise EvaluationError(
                    f"GeneratedAnswer.citation_ids[{position}] must be non-empty"
                )
            normalized = unicodedata.normalize("NFC", citation_id)
            if normalized in seen:
                raise EvaluationError(
                    f"GeneratedAnswer citations contain duplicate ID {normalized!r}"
                )
            seen.add(normalized)
            normalized_citations.append(normalized)
        if token_usage is not None and (
            isinstance(token_usage, bool)
            or not isinstance(token_usage, int)
            or token_usage < 0
        ):
            raise EvaluationError("GeneratedAnswer.token_usage must be an integer >= 0")
        if estimated_cost is not None:
            if isinstance(estimated_cost, bool) or not isinstance(
                estimated_cost, (int, float)
            ):
                raise EvaluationError(
                    "GeneratedAnswer.estimated_cost must be a finite number >= 0"
                )
            normalized_cost = float(estimated_cost)
            if not math.isfinite(normalized_cost) or normalized_cost < 0.0:
                raise EvaluationError(
                    "GeneratedAnswer.estimated_cost must be a finite number >= 0"
                )
            estimated_cost = normalized_cost
        object.__setattr__(self, "text", unicodedata.normalize("NFC", text))
        object.__setattr__(self, "citation_ids", tuple(normalized_citations))
        object.__setattr__(self, "token_usage", token_usage)
        object.__setattr__(self, "estimated_cost", estimated_cost)
        object.__setattr__(
            self,
            "metadata",
            normalize_json_mapping(
                {} if metadata is None else metadata,
                field_name="GeneratedAnswer.metadata",
                error_type=EvaluationError,
            ),
        )


@dataclass(frozen=True)
class GenerationOutput:
    """A validated generated answer plus observed generation latency."""

    generator: str
    answer: GeneratedAnswer
    latency_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.generator, str) or not self.generator.strip():
            raise EvaluationError("GenerationOutput.generator must be non-empty")
        if not isinstance(self.answer, GeneratedAnswer):
            raise EvaluationError("GenerationOutput.answer must be a GeneratedAnswer")
        if isinstance(self.latency_ms, bool) or not isinstance(
            self.latency_ms, (int, float)
        ):
            raise EvaluationError("GenerationOutput.latency_ms must be finite and >= 0")
        normalized = float(self.latency_ms)
        if not math.isfinite(normalized) or normalized < 0.0:
            raise EvaluationError("GenerationOutput.latency_ms must be finite and >= 0")
        object.__setattr__(self, "latency_ms", normalized)


class Generator(Protocol):
    """Minimal answer-generator contract; providers remain caller-controlled."""

    name: str

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        """Generate one answer for a validated request."""
        ...


GenerateCallable = Callable[[GenerationRequest], GeneratedAnswer]


@dataclass(frozen=True)
class CallableGenerator:
    """Adapt an arbitrary synchronous callable to the public Generator contract."""

    name: str
    callback: GenerateCallable

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise EvaluationError("CallableGenerator.name must be a non-empty string")
        if not callable(self.callback):
            raise EvaluationError("CallableGenerator.callback must be callable")

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        """Delegate to the caller-provided generation function."""

        return self.callback(request)


@dataclass(frozen=True)
class MockGenerator:
    """Deterministic offline generator for tests, examples, and pipelines."""

    answer: GeneratedAnswer
    name: str = "mock"

    def __post_init__(self) -> None:
        if not isinstance(self.answer, GeneratedAnswer):
            raise EvaluationError("MockGenerator.answer must be a GeneratedAnswer")
        if not isinstance(self.name, str) or not self.name.strip():
            raise EvaluationError("MockGenerator.name must be a non-empty string")

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        """Return the configured answer without network access."""

        if not isinstance(request, GenerationRequest):
            raise EvaluationError("request must be a GenerationRequest")
        return self.answer


def generate_answer(
    generator: Generator,
    request: GenerationRequest,
    *,
    clock: Clock = time.perf_counter,
) -> GenerationOutput:
    """Generate one answer and measure latency using a caller-controllable clock."""

    if not isinstance(request, GenerationRequest):
        raise EvaluationError("request must be a GenerationRequest")
    generator_name = _generator_name(generator)
    if not callable(clock):
        raise EvaluationError("clock must be callable")
    start = _read_clock(clock)
    try:
        answer = generator.generate(request)
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(
            f"generator {generator_name!r} failed: {exc}"
        ) from exc
    end = _read_clock(clock)
    if not isinstance(answer, GeneratedAnswer):
        raise EvaluationError(
            f"generator {generator_name!r} must return a GeneratedAnswer"
        )
    latency_ms = (end - start) * 1000.0
    if latency_ms < 0.0:
        raise EvaluationError("clock moved backwards during answer generation")
    return GenerationOutput(
        generator=generator_name,
        answer=answer,
        latency_ms=latency_ms,
    )


def _generator_name(generator: Generator) -> str:
    try:
        name = generator.name
        method = generator.generate
    except AttributeError as exc:
        raise EvaluationError(
            "generator must provide non-empty name and callable generate()"
        ) from exc
    if not isinstance(name, str) or not name.strip() or not callable(method):
        raise EvaluationError(
            "generator must provide non-empty name and callable generate()"
        )
    return name


def _read_clock(clock: Clock) -> float:
    try:
        value = float(clock())
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationError("clock must return a finite number") from exc
    if not math.isfinite(value):
        raise EvaluationError("clock must return a finite number")
    return value


__all__ = [
    "CallableGenerator",
    "Clock",
    "GenerateCallable",
    "GeneratedAnswer",
    "GenerationContext",
    "GenerationOutput",
    "GenerationRequest",
    "Generator",
    "MockGenerator",
    "generate_answer",
]
