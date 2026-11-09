"""Tests for the caller-controlled LLM-as-judge contracts."""

from __future__ import annotations

import pytest

from retrieval_lab import (
    DEFAULT_JUDGE_PROMPT_TEMPLATE,
    CallableJudge,
    JudgeOutput,
    JudgePromptTemplate,
    JudgeRequest,
    JudgeUsage,
    JudgeVerdict,
    MockJudge,
    PanelJudgment,
    judge_answer,
    judge_answer_with_panel,
)
from retrieval_lab.exceptions import EvaluationError


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _request() -> JudgeRequest:
    return JudgeRequest(
        query_id="q-1",
        query="What is retrieval quality?",
        answer="It is the measured relevance of retrieved items.",
        context=("Relevant items should be ranked first.",),
        reference="The relevance of retrieved items, measured per query.",
    )


def test_default_template_renders_every_placeholder() -> None:
    prompt = DEFAULT_JUDGE_PROMPT_TEMPLATE.render(_request())

    assert "What is retrieval quality?" in prompt
    assert "It is the measured relevance of retrieved items." in prompt
    assert "Relevant items should be ranked first." in prompt
    assert "The relevance of retrieved items, measured per query." in prompt


def test_template_renders_missing_reference_and_context_as_empty() -> None:
    template = JudgePromptTemplate(
        "Q: {query}\nC: [{context}]\nR: [{reference}]\nA: {answer}"
    )
    rendered = template.render(JudgeRequest(query_id="q", query="query", answer="ans"))

    assert rendered == "Q: query\nC: []\nR: []\nA: ans"


@pytest.mark.parametrize(
    "template",
    [
        "",
        "no placeholders",
        "Query only: {query}",
        "Answer only: {answer}",
        "Unknown: {query} {answer} {score}",
        "Spec: {query:>10} {answer}",
        "Conversion: {query!r} {answer}",
        "Unclosed: {query} {answer",
    ],
)
def test_template_rejects_invalid_templates(template: str) -> None:
    with pytest.raises(EvaluationError):
        JudgePromptTemplate(template)


def test_judge_request_validation() -> None:
    with pytest.raises(EvaluationError):
        JudgeRequest(query_id="", query="q", answer="a")
    with pytest.raises(EvaluationError):
        JudgeRequest(query_id="q", query=" ", answer="a")
    with pytest.raises(EvaluationError):
        JudgeRequest(query_id="q", query="q", answer="a", context="not-a-sequence")  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        JudgeRequest(query_id="q", query="q", answer="a", context=(1,))  # type: ignore[arg-type]
    request = JudgeRequest(query_id="q", query="q", answer="", context=["c"])
    assert request.context == ("c",)
    assert request.answer == ""


def test_verdict_and_usage_validation() -> None:
    assert JudgeVerdict(1).score == 1.0
    assert JudgeVerdict(0, usage=JudgeUsage(10, 5, 0)).score == 0.0
    for bad_score in (-0.1, 1.1, float("nan"), True, "high"):
        with pytest.raises(EvaluationError):
            JudgeVerdict(bad_score)  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        JudgeVerdict(0.5, usage=object())  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        JudgeUsage(input_tokens=-1)
    with pytest.raises(EvaluationError):
        JudgeUsage(output_tokens=True)
    with pytest.raises(EvaluationError):
        JudgeUsage(cost_usd=-0.5)
    with pytest.raises(EvaluationError):
        JudgeUsage(cost_usd=float("inf"))


def test_judge_answer_records_prompt_latency_and_usage() -> None:
    seen: dict[str, object] = {}

    def callback(request: JudgeRequest, prompt: str) -> JudgeVerdict:
        seen["prompt"] = prompt
        return JudgeVerdict(
            0.8,
            rationale="well supported",
            usage=JudgeUsage(input_tokens=120, output_tokens=20, cost_usd=0.01),
        )

    output = judge_answer(
        CallableJudge("judge-a", callback),
        _request(),
        clock=SequenceClock(1.0, 1.25),
    )

    assert output.judge == "judge-a"
    assert output.verdict.score == 0.8
    assert output.latency_ms == 250.0
    assert output.prompt == seen["prompt"]
    assert "What is retrieval quality?" in output.prompt


def test_judge_answer_wraps_failures_and_rejects_bad_returns() -> None:
    def failing(request: JudgeRequest, prompt: str) -> JudgeVerdict:
        raise RuntimeError("provider timeout")

    with pytest.raises(EvaluationError, match="judge 'bad' failed"):
        judge_answer(
            CallableJudge("bad", failing),
            _request(),
            clock=SequenceClock(1.0, 1.0),
        )

    def invalid(request: JudgeRequest, prompt: str) -> JudgeVerdict:
        return object()  # type: ignore[return-value]

    with pytest.raises(EvaluationError, match="must return a JudgeVerdict"):
        judge_answer(
            CallableJudge("bad", invalid),
            _request(),
            clock=SequenceClock(1.0, 1.0),
        )

    def typed_failure(request: JudgeRequest, prompt: str) -> JudgeVerdict:
        raise EvaluationError("typed failure")

    with pytest.raises(EvaluationError, match="typed failure"):
        judge_answer(
            CallableJudge("bad", typed_failure),
            _request(),
            clock=SequenceClock(1.0, 1.0),
        )

    with pytest.raises(EvaluationError, match="moved backwards"):
        judge_answer(
            MockJudge(JudgeVerdict(0.5)),
            _request(),
            clock=SequenceClock(2.0, 1.0),
        )


def test_panel_aggregates_scores_and_costs() -> None:
    judges = [
        CallableJudge(
            "judge-a",
            lambda request, prompt: JudgeVerdict(
                0.4, usage=JudgeUsage(input_tokens=100, output_tokens=10, cost_usd=0.02)
            ),
        ),
        CallableJudge(
            "judge-b",
            lambda request, prompt: JudgeVerdict(
                0.8, usage=JudgeUsage(input_tokens=90)
            ),
        ),
        MockJudge(JudgeVerdict(0.6), name="judge-c"),
    ]

    panel = judge_answer_with_panel(
        judges,
        _request(),
        clock=SequenceClock(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    )

    assert panel.query_id == "q-1"
    assert [output.judge for output in panel.outputs] == [
        "judge-a",
        "judge-b",
        "judge-c",
    ]
    assert panel.mean_score == pytest.approx(0.6)
    assert panel.score_spread == pytest.approx(0.4)
    assert panel.total_input_tokens == 190
    assert panel.total_output_tokens == 10
    assert panel.total_cost_usd == pytest.approx(0.02)


def test_panel_without_usage_reports_none_totals() -> None:
    panel = judge_answer_with_panel(
        [MockJudge(JudgeVerdict(0.5))],
        _request(),
        clock=SequenceClock(0.0, 0.1),
    )

    assert panel.total_input_tokens is None
    assert panel.total_output_tokens is None
    assert panel.total_cost_usd is None


def test_panel_rejects_empty_and_duplicate_judges() -> None:
    with pytest.raises(EvaluationError, match="must not be empty"):
        judge_answer_with_panel([], _request())
    with pytest.raises(EvaluationError, match="unique"):
        judge_answer_with_panel(
            [MockJudge(JudgeVerdict(0.5)), MockJudge(JudgeVerdict(0.6))],
            _request(),
            clock=SequenceClock(0.0, 0.1, 0.2, 0.3),
        )
    with pytest.raises(EvaluationError, match="sequence"):
        judge_answer_with_panel("judge", _request())  # type: ignore[arg-type]


def test_panel_judgment_rejects_inconsistent_records() -> None:
    output = JudgeOutput(
        judge="judge-a",
        verdict=JudgeVerdict(0.5),
        latency_ms=10.0,
        prompt="prompt",
    )
    assert PanelJudgment(query_id="q", outputs=(output,)).mean_score == 0.5
    with pytest.raises(EvaluationError, match="must not be empty"):
        PanelJudgment(query_id="q", outputs=())
    with pytest.raises(EvaluationError, match="unique"):
        PanelJudgment(query_id="q", outputs=(output, output))
    with pytest.raises(EvaluationError):
        JudgeOutput(judge="", verdict=JudgeVerdict(0.5), latency_ms=0.0, prompt="p")
    with pytest.raises(EvaluationError):
        JudgeOutput(judge="j", verdict=JudgeVerdict(0.5), latency_ms=-1.0, prompt="p")
