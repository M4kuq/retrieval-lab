"""Caller-controlled LLM-as-judge contracts for Retrieval Lab.

Judges are supplied by the caller, exactly like answer generators: the
library renders a validated prompt template, invokes the judge,
measures latency, and records the verdict together with the token and
cost usage the judge reports. The library ships no provider client
and never parses free-text judge output; a judge callable returns a
typed :class:`JudgeVerdict`.
"""

from __future__ import annotations

import math
import string
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from retrieval_lab.exceptions import EvaluationError
from retrieval_lab.generation import Clock

_PLACEHOLDERS = ("query", "answer", "context", "reference")


@dataclass(frozen=True)
class JudgeUsage:
    """Token and cost usage reported by one judge invocation."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("input_tokens", "output_tokens"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise EvaluationError(
                    f"JudgeUsage.{field_name} must be a non-negative integer or None"
                )
        if self.cost_usd is not None:
            if isinstance(self.cost_usd, bool) or not isinstance(
                self.cost_usd, (int, float)
            ):
                raise EvaluationError("JudgeUsage.cost_usd must be finite and >= 0")
            normalized = float(self.cost_usd)
            if not math.isfinite(normalized) or normalized < 0.0:
                raise EvaluationError("JudgeUsage.cost_usd must be finite and >= 0")
            object.__setattr__(self, "cost_usd", normalized)


@dataclass(frozen=True)
class JudgeRequest:
    """One generated answer awaiting judgment, with its evidence."""

    query_id: str
    query: str
    answer: str
    context: tuple[str, ...] = ()
    reference: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("query_id", "query"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise EvaluationError(f"JudgeRequest.{field_name} must be non-empty")
        if not isinstance(self.answer, str):
            raise EvaluationError("JudgeRequest.answer must be a string")
        if isinstance(self.context, (str, bytes)) or not isinstance(
            self.context, Sequence
        ):
            raise EvaluationError("JudgeRequest.context must be a sequence of strings")
        context = tuple(self.context)
        if not all(isinstance(item, str) for item in context):
            raise EvaluationError("JudgeRequest.context must be a sequence of strings")
        object.__setattr__(self, "context", context)
        if self.reference is not None and not isinstance(self.reference, str):
            raise EvaluationError("JudgeRequest.reference must be a string or None")


@dataclass(frozen=True)
class JudgePromptTemplate:
    """A judge prompt template with a closed set of placeholders."""

    template: str

    def __post_init__(self) -> None:
        if not isinstance(self.template, str) or not self.template.strip():
            raise EvaluationError("JudgePromptTemplate.template must be non-empty")
        seen: set[str] = set()
        try:
            parsed = string.Formatter().parse(self.template)
            for _, field_name, format_spec, conversion in parsed:
                if field_name is None:
                    continue
                if conversion is not None or format_spec:
                    raise EvaluationError(
                        "JudgePromptTemplate placeholders must not use "
                        "conversions or format specs"
                    )
                if field_name not in _PLACEHOLDERS:
                    raise EvaluationError(
                        f"JudgePromptTemplate placeholder {field_name!r} is not "
                        f"one of {', '.join(_PLACEHOLDERS)}"
                    )
                seen.add(field_name)
        except ValueError as exc:
            raise EvaluationError(
                "JudgePromptTemplate.template is not a valid format string"
            ) from exc
        if not {"query", "answer"} <= seen:
            raise EvaluationError(
                "JudgePromptTemplate must contain {query} and {answer}"
            )

    def render(self, request: JudgeRequest) -> str:
        """Render the prompt for one validated judge request."""

        if not isinstance(request, JudgeRequest):
            raise EvaluationError("request must be a JudgeRequest")
        return self.template.format(
            query=request.query,
            answer=request.answer,
            context="\n\n".join(request.context),
            reference=request.reference or "",
        )


DEFAULT_JUDGE_PROMPT_TEMPLATE = JudgePromptTemplate(
    "You are evaluating a generated answer for a search system.\n\n"
    "Query:\n{query}\n\n"
    "Retrieved context:\n{context}\n\n"
    "Reference answer:\n{reference}\n\n"
    "Generated answer:\n{answer}\n\n"
    "Score the generated answer from 0.0 to 1.0: 1.0 means it fully and "
    "accurately answers the query using the retrieved context, and 0.0 "
    "means it is wrong, unsupported, or unresponsive. Judge only the "
    "answer, not its length or style."
)


@dataclass(frozen=True)
class JudgeVerdict:
    """One judge's validated score for one answer."""

    score: float
    rationale: str | None = None
    usage: JudgeUsage | None = None

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise EvaluationError("JudgeVerdict.score must be between 0 and 1")
        normalized = float(self.score)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise EvaluationError("JudgeVerdict.score must be between 0 and 1")
        object.__setattr__(self, "score", normalized)
        if self.rationale is not None and not isinstance(self.rationale, str):
            raise EvaluationError("JudgeVerdict.rationale must be a string or None")
        if self.usage is not None and not isinstance(self.usage, JudgeUsage):
            raise EvaluationError("JudgeVerdict.usage must be a JudgeUsage or None")


@dataclass(frozen=True)
class JudgeOutput:
    """A validated verdict plus the prompt and latency that produced it."""

    judge: str
    verdict: JudgeVerdict
    latency_ms: float
    prompt: str

    def __post_init__(self) -> None:
        if not isinstance(self.judge, str) or not self.judge.strip():
            raise EvaluationError("JudgeOutput.judge must be non-empty")
        if not isinstance(self.verdict, JudgeVerdict):
            raise EvaluationError("JudgeOutput.verdict must be a JudgeVerdict")
        if isinstance(self.latency_ms, bool) or not isinstance(
            self.latency_ms, (int, float)
        ):
            raise EvaluationError("JudgeOutput.latency_ms must be finite and >= 0")
        normalized = float(self.latency_ms)
        if not math.isfinite(normalized) or normalized < 0.0:
            raise EvaluationError("JudgeOutput.latency_ms must be finite and >= 0")
        object.__setattr__(self, "latency_ms", normalized)
        if not isinstance(self.prompt, str):
            raise EvaluationError("JudgeOutput.prompt must be a string")


class Judge(Protocol):
    """Minimal judge contract; providers remain caller-controlled."""

    name: str

    def judge(self, request: JudgeRequest, prompt: str) -> JudgeVerdict:
        """Score one answer given its rendered judge prompt."""
        ...


JudgeCallable = Callable[[JudgeRequest, str], JudgeVerdict]


@dataclass(frozen=True)
class CallableJudge:
    """Adapt an arbitrary synchronous callable to the public Judge contract."""

    name: str
    callback: JudgeCallable

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise EvaluationError("CallableJudge.name must be a non-empty string")
        if not callable(self.callback):
            raise EvaluationError("CallableJudge.callback must be callable")

    def judge(self, request: JudgeRequest, prompt: str) -> JudgeVerdict:
        """Delegate to the caller-provided judge function."""

        return self.callback(request, prompt)


@dataclass(frozen=True)
class MockJudge:
    """Deterministic offline judge for tests, examples, and pipelines."""

    verdict: JudgeVerdict
    name: str = "mock"

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, JudgeVerdict):
            raise EvaluationError("MockJudge.verdict must be a JudgeVerdict")
        if not isinstance(self.name, str) or not self.name.strip():
            raise EvaluationError("MockJudge.name must be a non-empty string")

    def judge(self, request: JudgeRequest, prompt: str) -> JudgeVerdict:
        """Return the configured verdict without network access."""

        if not isinstance(request, JudgeRequest):
            raise EvaluationError("request must be a JudgeRequest")
        return self.verdict


@dataclass(frozen=True)
class PanelJudgment:
    """Verdicts from several judges for one answer, with panel totals."""

    query_id: str
    outputs: tuple[JudgeOutput, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise EvaluationError("PanelJudgment.query_id must be non-empty")
        outputs = tuple(self.outputs)
        if not outputs:
            raise EvaluationError("PanelJudgment.outputs must not be empty")
        if not all(isinstance(output, JudgeOutput) for output in outputs):
            raise EvaluationError("PanelJudgment.outputs must contain JudgeOutput")
        if len({output.judge for output in outputs}) != len(outputs):
            raise EvaluationError("PanelJudgment judge names must be unique")
        object.__setattr__(self, "outputs", outputs)

    @property
    def mean_score(self) -> float:
        """Return the mean of the panel's scores."""

        return math.fsum(o.verdict.score for o in self.outputs) / len(self.outputs)

    @property
    def score_spread(self) -> float:
        """Return the difference between the highest and lowest scores."""

        scores = [o.verdict.score for o in self.outputs]
        return max(scores) - min(scores)

    @property
    def total_input_tokens(self) -> int | None:
        """Return summed input tokens, or None when no judge reported usage."""

        return _total_usage(self.outputs, "input_tokens")

    @property
    def total_output_tokens(self) -> int | None:
        """Return summed output tokens, or None when no judge reported usage."""

        return _total_usage(self.outputs, "output_tokens")

    @property
    def total_cost_usd(self) -> float | None:
        """Return summed cost, or None when no judge reported a cost."""

        values = [
            o.verdict.usage.cost_usd
            for o in self.outputs
            if o.verdict.usage is not None and o.verdict.usage.cost_usd is not None
        ]
        return math.fsum(values) if values else None


def judge_answer(
    judge: Judge,
    request: JudgeRequest,
    *,
    template: JudgePromptTemplate = DEFAULT_JUDGE_PROMPT_TEMPLATE,
    clock: Clock = time.perf_counter,
) -> JudgeOutput:
    """Judge one answer and measure latency using a caller-controllable clock."""

    if not isinstance(request, JudgeRequest):
        raise EvaluationError("request must be a JudgeRequest")
    if not isinstance(template, JudgePromptTemplate):
        raise EvaluationError("template must be a JudgePromptTemplate")
    judge_name = _judge_name(judge)
    if not callable(clock):
        raise EvaluationError("clock must be callable")
    prompt = template.render(request)
    start = _read_clock(clock)
    try:
        verdict = judge.judge(request, prompt)
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"judge {judge_name!r} failed: {exc}") from exc
    end = _read_clock(clock)
    if not isinstance(verdict, JudgeVerdict):
        raise EvaluationError(f"judge {judge_name!r} must return a JudgeVerdict")
    latency_ms = (end - start) * 1000.0
    if latency_ms < 0.0:
        raise EvaluationError("clock moved backwards during judging")
    return JudgeOutput(
        judge=judge_name,
        verdict=verdict,
        latency_ms=latency_ms,
        prompt=prompt,
    )


def judge_answer_with_panel(
    judges: Sequence[Judge],
    request: JudgeRequest,
    *,
    template: JudgePromptTemplate = DEFAULT_JUDGE_PROMPT_TEMPLATE,
    clock: Clock = time.perf_counter,
) -> PanelJudgment:
    """Judge one answer with several judges and aggregate their verdicts."""

    if isinstance(judges, (str, bytes)) or not isinstance(judges, Sequence):
        raise EvaluationError("judges must be a sequence of Judge values")
    panel = tuple(judges)
    if not panel:
        raise EvaluationError("judges must not be empty")
    names = [_judge_name(judge) for judge in panel]
    if len(set(names)) != len(names):
        raise EvaluationError("judge names must be unique")
    outputs = tuple(
        judge_answer(judge, request, template=template, clock=clock) for judge in panel
    )
    return PanelJudgment(query_id=request.query_id, outputs=outputs)


def _total_usage(outputs: tuple[JudgeOutput, ...], field_name: str) -> int | None:
    values = [
        getattr(o.verdict.usage, field_name)
        for o in outputs
        if o.verdict.usage is not None
        and getattr(o.verdict.usage, field_name) is not None
    ]
    return sum(values) if values else None


def _judge_name(judge: Judge) -> str:
    try:
        name = judge.name
        method = judge.judge
    except AttributeError as exc:
        raise EvaluationError(
            "judge must provide non-empty name and callable judge()"
        ) from exc
    if not isinstance(name, str) or not name.strip() or not callable(method):
        raise EvaluationError("judge must provide non-empty name and callable judge()")
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
    "DEFAULT_JUDGE_PROMPT_TEMPLATE",
    "CallableJudge",
    "Judge",
    "JudgeCallable",
    "JudgeOutput",
    "JudgePromptTemplate",
    "JudgeRequest",
    "JudgeUsage",
    "JudgeVerdict",
    "MockJudge",
    "PanelJudgment",
    "judge_answer",
    "judge_answer_with_panel",
]
