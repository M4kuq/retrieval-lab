# Answer generation and evaluation

Retrieval Lab v0.4 extends retrieval evaluation with a separate, provider-independent answer-generation and rule-based answer-evaluation layer. Retrieval metrics and answer metrics remain distinct so a run can tell whether retrieval failed, generation failed, or citations were incorrect.

## Generator contract

The core package does not require an LLM provider or network access. A generator implements the public `Generator` contract:

```python
from retrieval_lab import (
    GeneratedAnswer,
    GenerationContext,
    GenerationRequest,
    MockGenerator,
    generate_answer,
)

request = GenerationRequest(
    "q-1",
    "Which AWS service stores secrets?",
    [
        GenerationContext(
            "security-doc",
            "security-doc:0",
            "AWS Secrets Manager stores and rotates secrets.",
        )
    ],
)

output = generate_answer(
    MockGenerator(
        GeneratedAnswer(
            "AWS Secrets Manager",
            citation_ids=["security-doc"],
            token_usage=12,
            estimated_cost=0.0,
        )
    ),
    request,
)
```

`CallableGenerator` can wrap an existing local model client, Ollama integration, OpenAI-compatible client, or application function. Retrieval Lab itself does not make an external request unless the caller-provided generator does so.

## Rule-based metrics

`evaluate_answer()` computes deterministic metrics without an LLM judge:

- Exact Match after Unicode NFKC, case, and whitespace normalization;
- multilingual Token F1;
- Citation Presence;
- Citation Validity: cited document IDs that are present in the supplied context divided by all citations;
- Citation Precision: relevant cited document IDs divided by all citations;
- Citation Recall: relevant cited document IDs divided by all known relevant document IDs;
- Citation Coverage: cited relevant documents divided by relevant documents that were actually present in the supplied context;
- Required Fact Coverage when `required_facts` are supplied;
- Empty Answer;
- answer-generation latency;
- token usage and estimated cost when the generator reports them.

When there are no citations, citation ratios are reported as `0.0`; the library does not treat missing citations as perfect validity. Exact Match and Token F1 are `None` when no reference answer exists rather than being reported as zero.

## Failure attribution

The rule-based baseline reports one conservative failure category per query:

- `retrieval_failure`: none of the known relevant documents were supplied as generation context;
- `generation_failure`: relevant context was available, but the deterministic correctness rule failed;
- `citation_failure`: the answer passed the deterministic correctness rule, but citations were invalid or incomplete;
- `success`: deterministic correctness and required citation checks passed;
- `unscored`: no reference answer, required facts, or unanswerable condition was available to determine correctness.

If `required_facts` are supplied, all facts must be present after normalization for rule-based correctness. Otherwise, when a `reference_answer` is supplied, Exact Match is the conservative correctness rule. These failure categories are diagnostic rules, not semantic or human-equivalent judgments.

## Unanswerable queries

`AnswerReference(answerable=False)` treats an empty answer as the deterministic success condition. A non-empty answer is classified as a generation failure. This rule does not attempt to detect unsupported claims inside a non-empty answer.

## Batch evaluation

`evaluate_answers()` aggregates per-query evidence while preserving query order. Optional reference-based metrics are averaged only over queries where they were measured. Total token usage and estimated cost are reported only when every answer provides the corresponding value.

## LLM-as-a-Judge

LLM-as-a-Judge is intentionally not part of this deterministic baseline. A later optional judge integration must record the judge model, prompt, temperature, scoring rubric, repeat count, variance, human-agreement evidence, and cost. Judge output must remain explicitly experimental and must not replace rule-based evidence or human review.
