import pytest

from retrieval_lab import (
    ClaimAssessment,
    EvaluationError,
    RagEvaluation,
    context_precision,
    context_recall,
    evaluate_rag_answer,
    faithfulness,
)


def test_faithfulness_counts_supported_claims() -> None:
    claims = (
        ClaimAssessment("The sky is blue.", True),
        ClaimAssessment("Water is wet.", True),
        ClaimAssessment("Fire is cold.", False),
        ClaimAssessment("Ice floats.", True),
    )
    assert faithfulness(claims) == pytest.approx(0.75)


def test_faithfulness_extremes() -> None:
    assert faithfulness((ClaimAssessment("a", True),)) == 1.0
    assert faithfulness((ClaimAssessment("a", False),)) == 0.0


def test_faithfulness_requires_claims() -> None:
    with pytest.raises(EvaluationError):
        faithfulness(())
    with pytest.raises(EvaluationError):
        faithfulness("not-a-sequence")  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        faithfulness((object(),))  # type: ignore[arg-type]


def test_claim_assessment_validates_its_fields() -> None:
    with pytest.raises(EvaluationError):
        ClaimAssessment(" ", True)
    with pytest.raises(EvaluationError):
        ClaimAssessment("claim", 1)  # type: ignore[arg-type]


def test_context_precision_weights_early_relevant_items() -> None:
    # Relevant at ranks 1 and 3: (1/1 + 2/3) / 2 = 5/6.
    assert context_precision(["a", "b", "c", "d"], {"a", "c"}) == pytest.approx(5 / 6)
    # Perfect ranking scores 1.0.
    assert context_precision(["a", "c", "b", "d"], {"a", "c"}) == 1.0
    # Relevant last: (1/3 + 2/4) / 2 = 5/12.
    assert context_precision(["b", "d", "a", "c"], {"a", "c"}) == pytest.approx(5 / 12)


def test_context_precision_without_relevant_items_is_zero() -> None:
    assert context_precision(["a", "b"], {"z"}) == 0.0
    assert context_precision([], {"z"}) == 0.0


def test_context_recall_counts_relevant_coverage() -> None:
    assert context_recall(["a", "b"], {"a", "c", "d"}) == pytest.approx(1 / 3)
    assert context_recall(["a", "c", "d"], {"a", "c", "d"}) == 1.0
    assert context_recall([], {"a"}) == 0.0


def test_context_metrics_validate_inputs() -> None:
    with pytest.raises(EvaluationError):
        context_precision(["a", "a"], {"a"})
    with pytest.raises(EvaluationError):
        context_recall(["a"], set())
    with pytest.raises(EvaluationError):
        context_recall(["a"], {" "})
    with pytest.raises(EvaluationError):
        context_precision("abc", {"a"})  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        context_recall([1], {"a"})  # type: ignore[list-item]


def test_evaluate_rag_answer_combines_all_metrics() -> None:
    evaluation = evaluate_rag_answer(
        "q-1",
        context_ids=["a", "b", "c", "d"],
        relevant_ids={"a", "c", "e"},
        claims=(
            ClaimAssessment("one", True),
            ClaimAssessment("two", False),
        ),
    )
    assert evaluation.faithfulness == pytest.approx(0.5)
    assert evaluation.context_precision == pytest.approx(5 / 6)
    assert evaluation.context_recall == pytest.approx(2 / 3)
    assert evaluation.to_dict() == {
        "context_precision": evaluation.context_precision,
        "context_recall": evaluation.context_recall,
        "faithfulness": evaluation.faithfulness,
        "query_id": "q-1",
    }


def test_evaluate_rag_answer_without_claims_reports_no_faithfulness() -> None:
    evaluation = evaluate_rag_answer("q-1", context_ids=["a"], relevant_ids={"a"})
    assert evaluation.faithfulness is None
    assert evaluation.context_precision == 1.0
    assert evaluation.context_recall == 1.0


def test_rag_evaluation_validates_scores_and_query() -> None:
    with pytest.raises(EvaluationError):
        RagEvaluation(
            query_id="q", faithfulness=None, context_precision=1.5, context_recall=0.5
        )
    with pytest.raises(EvaluationError):
        RagEvaluation(
            query_id=" ", faithfulness=None, context_precision=0.5, context_recall=0.5
        )
    with pytest.raises(EvaluationError):
        evaluate_rag_answer(" ", context_ids=[], relevant_ids={"a"})
    with pytest.raises(EvaluationError):
        evaluate_rag_answer(
            "q",
            context_ids=["a"],
            relevant_ids={"a"},
            claims=None,  # type: ignore[arg-type]
        )
