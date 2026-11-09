import math

import pytest

from retrieval_lab.answer_evaluation import (
    AnswerEvaluation,
    AnswerEvaluationResult,
    AnswerReference,
    evaluate_answer,
    evaluate_answers,
    exact_match,
    token_f1,
)
from retrieval_lab.exceptions import EvaluationError
from retrieval_lab.generation import (
    GeneratedAnswer,
    GenerationContext,
    GenerationOutput,
    GenerationRequest,
)


def _request(
    *document_ids: str,
    query_id: str = "q-1",
) -> GenerationRequest:
    contexts = [
        GenerationContext(
            document_id,
            f"chunk-{index}",
            f"context from {document_id}",
        )
        for index, document_id in enumerate(document_ids)
    ]
    return GenerationRequest(query_id, "query", contexts)


def _output(
    text: str,
    citations: tuple[str, ...] = (),
    *,
    token_usage: int | None = 10,
    estimated_cost: float | None = 0.01,
    latency_ms: float = 20.0,
) -> GenerationOutput:
    return GenerationOutput(
        "mock",
        GeneratedAnswer(
            text,
            citation_ids=citations,
            token_usage=token_usage,
            estimated_cost=estimated_cost,
        ),
        latency_ms,
    )


def test_exact_match_normalizes_case_width_and_whitespace() -> None:
    assert (
        exact_match("  \uff21\uff37\uff33  Secrets Manager ", "aws secrets manager")
        == 1.0
    )
    assert exact_match("AWS S3", "AWS Secrets Manager") == 0.0


def test_token_f1_matches_hand_calculation_for_latin_text() -> None:
    assert token_f1("alpha beta", "alpha beta gamma") == pytest.approx(0.8)


def test_token_f1_is_deterministic_for_japanese_text() -> None:
    assert token_f1("設定方法", "設定手順") == pytest.approx(0.5)


def test_token_f1_is_zero_for_empty_or_disjoint_answer() -> None:
    assert token_f1("", "reference") == 0.0
    assert token_f1("alpha", "beta") == 0.0


def test_answer_evaluation_citation_metrics_match_hand_calculation() -> None:
    reference = AnswerReference(
        "q-1",
        relevant_document_ids={"doc-1", "doc-2"},
        required_facts=["expected fact"],
    )
    evaluation = evaluate_answer(
        reference,
        _request("doc-1", "doc-3"),
        _output("expected fact", ("doc-1", "doc-3")),
    )

    assert evaluation.citation_presence == 1.0
    assert evaluation.citation_validity == 1.0
    assert evaluation.citation_precision == pytest.approx(0.5)
    assert evaluation.citation_recall == pytest.approx(0.5)
    assert evaluation.citation_coverage == 1.0
    assert evaluation.required_fact_coverage == 1.0
    assert evaluation.failure_type == "citation_failure"


def test_invalid_citation_reduces_validity_and_precision() -> None:
    evaluation = evaluate_answer(
        AnswerReference(
            "q-1",
            relevant_document_ids={"doc-1"},
            reference_answer="answer",
        ),
        _request("doc-1"),
        _output("answer", ("doc-1", "not-in-context")),
    )

    assert evaluation.citation_validity == pytest.approx(0.5)
    assert evaluation.citation_precision == pytest.approx(0.5)
    assert evaluation.citation_recall == 1.0
    assert evaluation.failure_type == "citation_failure"


def test_no_citations_produce_zero_citation_metrics() -> None:
    evaluation = evaluate_answer(
        AnswerReference(
            "q-1",
            relevant_document_ids={"doc-1"},
            reference_answer="answer",
        ),
        _request("doc-1"),
        _output("answer"),
    )

    assert evaluation.citation_presence == 0.0
    assert evaluation.citation_validity == 0.0
    assert evaluation.citation_precision == 0.0
    assert evaluation.citation_recall == 0.0
    assert evaluation.citation_coverage == 0.0
    assert evaluation.failure_type == "citation_failure"


def test_retrieval_failure_precedes_generation_scoring() -> None:
    evaluation = evaluate_answer(
        AnswerReference(
            "q-1",
            relevant_document_ids={"doc-1"},
            reference_answer="correct answer",
        ),
        _request("doc-2"),
        _output("wrong answer", ("doc-2",)),
    )

    assert evaluation.exact_match == 0.0
    assert evaluation.failure_type == "retrieval_failure"


def test_generation_failure_when_relevant_context_exists_but_fact_is_missing() -> None:
    evaluation = evaluate_answer(
        AnswerReference(
            "q-1",
            relevant_document_ids={"doc-1"},
            required_facts=["AWS Secrets Manager"],
        ),
        _request("doc-1"),
        _output("Amazon S3", ("doc-1",)),
    )

    assert evaluation.required_fact_coverage == 0.0
    assert evaluation.failure_type == "generation_failure"


def test_success_requires_rule_based_correctness_and_complete_citations() -> None:
    evaluation = evaluate_answer(
        AnswerReference(
            "q-1",
            relevant_document_ids={"doc-1"},
            reference_answer="AWS Secrets Manager",
        ),
        _request("doc-1"),
        _output("aws secrets manager", ("doc-1",)),
    )

    assert evaluation.exact_match == 1.0
    assert evaluation.token_f1 == 1.0
    assert evaluation.failure_type == "success"


def test_unscored_when_no_reference_answer_or_required_facts_exist() -> None:
    evaluation = evaluate_answer(
        AnswerReference("q-1", relevant_document_ids={"doc-1"}),
        _request("doc-1"),
        _output("plausible answer", ("doc-1",)),
    )

    assert evaluation.exact_match is None
    assert evaluation.token_f1 is None
    assert evaluation.required_fact_coverage is None
    assert evaluation.failure_type == "unscored"


def test_unanswerable_query_succeeds_only_with_empty_answer() -> None:
    reference = AnswerReference("q-1", answerable=False)

    empty = evaluate_answer(reference, _request(), _output(""))
    nonempty = evaluate_answer(reference, _request(), _output("invented"))

    assert empty.empty_answer
    assert empty.failure_type == "success"
    assert nonempty.failure_type == "generation_failure"


def test_required_fact_coverage_counts_normalized_substrings() -> None:
    evaluation = evaluate_answer(
        AnswerReference(
            "q-1",
            required_facts=["AWS Secrets Manager", "rotation"],
        ),
        _request(),
        _output("aws secrets manager supports secrets"),
    )

    assert evaluation.required_fact_coverage == pytest.approx(0.5)
    assert evaluation.failure_type == "generation_failure"


def test_evaluate_answer_preserves_usage_latency_and_cost() -> None:
    evaluation = evaluate_answer(
        AnswerReference("q-1", answerable=False),
        _request(),
        _output("", token_usage=12, estimated_cost=0.25, latency_ms=7.5),
    )

    assert evaluation.token_usage == 12
    assert evaluation.estimated_cost == pytest.approx(0.25)
    assert evaluation.latency_ms == pytest.approx(7.5)


def test_batch_aggregates_measured_metrics_and_usage() -> None:
    references = [
        AnswerReference("q-1", reference_answer="a"),
        AnswerReference("q-2", reference_answer="b"),
    ]
    requests = [_request(query_id="q-1"), _request(query_id="q-2")]
    outputs = [
        _output("a", token_usage=4, estimated_cost=0.1, latency_ms=10.0),
        _output("x", token_usage=6, estimated_cost=0.2, latency_ms=30.0),
    ]

    result = evaluate_answers(references, requests, outputs)
    payload = result.to_dict()

    assert isinstance(result, AnswerEvaluationResult)
    assert result.mean_exact_match == pytest.approx(0.5)
    assert result.mean_token_f1 == pytest.approx(0.5)
    assert result.empty_answer_rate == 0.0
    assert result.total_token_usage == 10
    assert result.total_estimated_cost == pytest.approx(0.3)
    assert payload["mean_generation_latency_ms"] == pytest.approx(20.0)
    assert payload["query_count"] == 2


def test_batch_aggregates_optional_metrics_only_when_measured() -> None:
    result = evaluate_answers(
        [
            AnswerReference("q-1", reference_answer="answer"),
            AnswerReference("q-2"),
        ],
        [_request(query_id="q-1"), _request(query_id="q-2")],
        [
            _output("answer", token_usage=1, estimated_cost=0.1),
            _output("anything", token_usage=None, estimated_cost=None),
        ],
    )

    assert result.mean_exact_match == 1.0
    assert result.mean_token_f1 == 1.0
    assert result.total_token_usage is None
    assert result.total_estimated_cost is None


def test_evaluate_answer_requires_matching_query_ids() -> None:
    with pytest.raises(EvaluationError, match="query IDs"):
        evaluate_answer(
            AnswerReference("q-1"),
            _request(query_id="q-2"),
            _output("answer"),
        )


def test_evaluate_answers_rejects_empty_or_mismatched_batches() -> None:
    with pytest.raises(EvaluationError, match="must not be empty"):
        evaluate_answers([], [], [])
    with pytest.raises(EvaluationError, match="same length"):
        evaluate_answers(
            [AnswerReference("q-1")],
            [_request()],
            [],
        )


def test_result_rejects_duplicate_query_ids() -> None:
    evaluation = evaluate_answer(
        AnswerReference("q-1", answerable=False),
        _request(),
        _output(""),
    )

    with pytest.raises(EvaluationError, match="unique"):
        AnswerEvaluationResult((evaluation, evaluation))


@pytest.mark.parametrize("value", ["", "   ", None, 1])
def test_answer_reference_rejects_invalid_query_id(value: object) -> None:
    with pytest.raises(EvaluationError, match="query_id"):
        AnswerReference(value)  # type: ignore[arg-type]


def test_answer_reference_rejects_duplicate_required_facts_after_normalization() -> (
    None
):
    with pytest.raises(EvaluationError, match="duplicate"):
        AnswerReference(
            "q-1",
            required_facts=["\uff21\uff37\uff33", "aws"],
        )


def test_exact_and_token_f1_require_nonempty_reference() -> None:
    with pytest.raises(EvaluationError, match="reference"):
        exact_match("answer", "")
    with pytest.raises(EvaluationError, match="reference"):
        token_f1("answer", "")


def test_token_f1_treats_punctuation_as_its_own_token() -> None:
    assert token_f1("hello, world", "hello, world") == pytest.approx(1.0)
    assert token_f1("hello, world", "hello world") == pytest.approx(0.8)


@pytest.mark.parametrize("value", [["doc-1"], "doc-1", 42])
def test_answer_reference_rejects_non_set_document_ids(value: object) -> None:
    with pytest.raises(EvaluationError, match="must be a set"):
        AnswerReference("q-1", relevant_document_ids=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("identifier", ["", "   ", 42])
def test_answer_reference_rejects_invalid_document_id(identifier: object) -> None:
    with pytest.raises(EvaluationError, match="document IDs"):
        AnswerReference(
            "q-1",
            relevant_document_ids={identifier},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", ["", "   ", 42])
def test_answer_reference_rejects_invalid_reference_answer(value: object) -> None:
    with pytest.raises(EvaluationError, match="reference_answer"):
        AnswerReference("q-1", reference_answer=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["fact", 42])
def test_answer_reference_rejects_non_sequence_required_facts(
    value: object,
) -> None:
    with pytest.raises(EvaluationError, match="must be a sequence"):
        AnswerReference("q-1", required_facts=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("fact", ["", "   ", 42])
def test_answer_reference_rejects_invalid_required_fact(fact: object) -> None:
    with pytest.raises(EvaluationError, match=r"required_facts\[0\]"):
        AnswerReference("q-1", required_facts=[fact])  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, "yes", None])
def test_answer_reference_rejects_non_bool_answerable(value: object) -> None:
    with pytest.raises(EvaluationError, match="answerable"):
        AnswerReference("q-1", answerable=value)  # type: ignore[arg-type]


def test_result_rejects_empty_evaluations() -> None:
    with pytest.raises(EvaluationError, match="must not be empty"):
        AnswerEvaluationResult(())


def test_result_rejects_non_evaluation_items() -> None:
    with pytest.raises(EvaluationError, match="must be an AnswerEvaluation"):
        AnswerEvaluationResult(("not-an-evaluation",))  # type: ignore[arg-type]


def test_result_rejects_non_finite_metrics_when_serializing() -> None:
    evaluation = AnswerEvaluation(
        query_id="q-1",
        exact_match=None,
        token_f1=None,
        citation_presence=0.0,
        citation_validity=0.0,
        citation_precision=0.0,
        citation_recall=0.0,
        citation_coverage=0.0,
        empty_answer=True,
        required_fact_coverage=None,
        latency_ms=math.inf,
        token_usage=None,
        estimated_cost=None,
        failure_type="unscored",
    )

    with pytest.raises(EvaluationError, match="finite"):
        AnswerEvaluationResult((evaluation,)).to_dict()


def test_evaluate_answer_rejects_wrong_argument_types() -> None:
    reference = AnswerReference("q-1")
    request = _request()
    output = _output("answer")

    with pytest.raises(EvaluationError, match="reference must be"):
        evaluate_answer("q-1", request, output)  # type: ignore[arg-type]
    with pytest.raises(EvaluationError, match="request must be"):
        evaluate_answer(reference, "q-1", output)  # type: ignore[arg-type]
    with pytest.raises(EvaluationError, match="output must be"):
        evaluate_answer(reference, request, "answer")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("references", "requests", "outputs", "name"),
    [
        ("q-1", [], [], "references"),
        ([], "q-1", [], "requests"),
        ([], [], "q-1", "outputs"),
    ],
)
def test_evaluate_answers_rejects_non_sequence_batches(
    references: object,
    requests: object,
    outputs: object,
    name: str,
) -> None:
    with pytest.raises(EvaluationError, match=f"{name} must be a sequence"):
        evaluate_answers(references, requests, outputs)  # type: ignore[arg-type]


def test_exact_match_and_token_f1_reject_non_string_inputs() -> None:
    with pytest.raises(EvaluationError, match="answer must be a string"):
        exact_match(42, "reference")  # type: ignore[arg-type]
    with pytest.raises(EvaluationError, match="reference must be a string"):
        token_f1("answer", None)  # type: ignore[arg-type]
