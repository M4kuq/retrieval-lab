import math

import pytest

from retrieval_lab.exceptions import EvaluationError
from retrieval_lab.generation import (
    CallableGenerator,
    GeneratedAnswer,
    GenerationContext,
    GenerationOutput,
    GenerationRequest,
    MockGenerator,
    generate_answer,
)


class SequenceClock:
    def __init__(self, *values: object) -> None:
        self._values = iter(values)

    def __call__(self) -> object:
        return next(self._values)


def _request() -> GenerationRequest:
    return GenerationRequest(
        "q-1",
        "What stores secrets?",
        [
            GenerationContext(
                "doc-1",
                "chunk-1",
                "AWS Secrets Manager stores secrets.",
                {"source": "manual"},
            )
        ],
    )


def test_generation_context_normalizes_and_copies_metadata() -> None:
    metadata = {"source": "before"}
    context = GenerationContext("doc-1", "chunk-1", "text", metadata)
    metadata["source"] = "after"

    assert context.metadata == {"source": "before"}
    with pytest.raises(TypeError):
        context.metadata["source"] = "mutated"  # type: ignore[index]


@pytest.mark.parametrize("field", ["document_id", "chunk_id", "text"])
def test_generation_context_rejects_empty_required_fields(field: str) -> None:
    values = {"document_id": "doc", "chunk_id": "chunk", "text": "text"}
    values[field] = ""

    with pytest.raises(EvaluationError, match=field):
        GenerationContext(**values)


def test_generation_request_allows_empty_contexts_for_retrieval_failure_analysis() -> None:
    request = GenerationRequest("q-1", "query")

    assert request.contexts == ()


def test_generation_request_rejects_duplicate_chunk_ids() -> None:
    context = GenerationContext("doc-1", "chunk-1", "text")

    with pytest.raises(EvaluationError, match="duplicate chunk ID"):
        GenerationRequest("q-1", "query", [context, context])


def test_generated_answer_preserves_empty_answer_and_usage() -> None:
    answer = GeneratedAnswer(
        "",
        citation_ids=["doc-1"],
        token_usage=0,
        estimated_cost=0,
        metadata={"provider": "mock"},
    )

    assert answer.text == ""
    assert answer.citation_ids == ("doc-1",)
    assert answer.token_usage == 0
    assert answer.estimated_cost == 0.0
    assert answer.metadata == {"provider": "mock"}


def test_generated_answer_rejects_duplicate_citations() -> None:
    with pytest.raises(EvaluationError, match="duplicate ID"):
        GeneratedAnswer("answer", citation_ids=["doc-1", "doc-1"])


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_generated_answer_rejects_invalid_token_usage(value: object) -> None:
    with pytest.raises(EvaluationError, match="token_usage"):
        GeneratedAnswer("answer", token_usage=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan, True, "1"])
def test_generated_answer_rejects_invalid_estimated_cost(value: object) -> None:
    with pytest.raises(EvaluationError, match="estimated_cost"):
        GeneratedAnswer("answer", estimated_cost=value)  # type: ignore[arg-type]


def test_generate_answer_measures_latency_and_preserves_generator_name() -> None:
    answer = GeneratedAnswer("AWS Secrets Manager", citation_ids=["doc-1"])
    output = generate_answer(
        MockGenerator(answer, name="offline-mock"),
        _request(),
        clock=SequenceClock(10.0, 10.025),  # type: ignore[arg-type]
    )

    assert isinstance(output, GenerationOutput)
    assert output.generator == "offline-mock"
    assert output.answer is answer
    assert output.latency_ms == pytest.approx(25.0)


def test_callable_generator_delegates_to_callback() -> None:
    seen: list[str] = []

    def callback(request: GenerationRequest) -> GeneratedAnswer:
        seen.append(request.query_id)
        return GeneratedAnswer("answer")

    output = generate_answer(
        CallableGenerator("callable", callback),
        _request(),
        clock=SequenceClock(1.0, 1.0),  # type: ignore[arg-type]
    )

    assert output.answer.text == "answer"
    assert seen == ["q-1"]


def test_generate_answer_wraps_unexpected_provider_error() -> None:
    def failing(_request: GenerationRequest) -> GeneratedAnswer:
        raise RuntimeError("provider failed")

    with pytest.raises(EvaluationError, match="provider failed") as captured:
        generate_answer(
            CallableGenerator("failing", failing),
            _request(),
            clock=SequenceClock(1.0),  # type: ignore[arg-type]
        )

    assert isinstance(captured.value.__cause__, RuntimeError)


def test_generate_answer_preserves_evaluation_error() -> None:
    error = EvaluationError("known failure")

    def failing(_request: GenerationRequest) -> GeneratedAnswer:
        raise error

    with pytest.raises(EvaluationError, match="known failure") as captured:
        generate_answer(
            CallableGenerator("failing", failing),
            _request(),
            clock=SequenceClock(1.0),  # type: ignore[arg-type]
        )

    assert captured.value is error


def test_generate_answer_rejects_provider_returning_wrong_type() -> None:
    generator = CallableGenerator(
        "bad",
        lambda _request: object(),  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(EvaluationError, match="GeneratedAnswer"):
        generate_answer(
            generator,
            _request(),
            clock=SequenceClock(1.0, 1.0),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("clock_value", [math.inf, math.nan, "bad"])
def test_generate_answer_rejects_invalid_clock(clock_value: object) -> None:
    with pytest.raises(EvaluationError, match="finite"):
        generate_answer(
            MockGenerator(GeneratedAnswer("answer")),
            _request(),
            clock=SequenceClock(clock_value),  # type: ignore[arg-type]
        )


def test_generate_answer_rejects_clock_moving_backwards() -> None:
    with pytest.raises(EvaluationError, match="backwards"):
        generate_answer(
            MockGenerator(GeneratedAnswer("answer")),
            _request(),
            clock=SequenceClock(2.0, 1.0),  # type: ignore[arg-type]
        )


def test_generation_output_validates_latency() -> None:
    answer = GeneratedAnswer("answer")

    with pytest.raises(EvaluationError, match="latency_ms"):
        GenerationOutput("mock", answer, -1.0)
