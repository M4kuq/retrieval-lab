# Saved-run retrieval regression gates

Retrieval Lab can evaluate a saved candidate result against a saved baseline
without running a retriever or downloading a model. The checked-in files under
`examples/ci/` are a small, offline schema-version 1 contract for this flow.

```console
retrieval-lab gate \
  --config examples/ci/retrieval-lab.yaml \
  --baseline examples/ci/baseline.json \
  examples/ci/candidate-improved.json \
  --json
```

The configuration uses both `max_absolute_drop` and `max_relative_drop` for
`bm25` Recall@1. A higher retrieval score is better. An improvement has zero
drop; a regression is measured as baseline minus candidate. Relative drop uses
the comparison API's baseline-relative semantics, including its explicit
baseline-zero status.

## Exit statuses

The command is suitable for a CI step:

- `0`: every configured constraint passed.
- `1`: the runs were comparable and at least one quality constraint failed.
- `2`: invalid configuration or result input, including incomparable runs.
- `3`: unexpected runtime failure.

Comparison must complete before regression amounts are calculated. A
comparability failure is therefore an input failure (`2`), not a quality
failure (`1`). The Python API raises the typed `IncomparableRunError` for the
same condition.

## Baseline and artifact policy

The baseline is the last accepted result for the same dataset/query contract.
It must be updated deliberately when the dataset, relevance definition,
metric implementation, or evaluation cutoffs intentionally change. Keep the
baseline and candidate as result JSON artifacts and pass their paths to the
gate; result loading rejects malformed, duplicate-key, and non-finite JSON.

Retriever, corpus, chunk, configuration, seed, runtime, and run identity are
reported as experimental variables. They do not make otherwise matching runs
incomparable, but changes should be recorded in the pull request that updates
the baseline. The strict comparison fields are dataset hash, relevance level,
metric version, top-k, query IDs, and per-query metric/cutoff shapes.

The public workflow template at
`examples/github-actions/retrieval-quality-gate.yml` assumes that a calling
workflow has already placed baseline and candidate artifacts at the documented
paths. It has read-only repository permissions and performs no secrets,
external API calls, or model downloads. The active CI workflow runs one passing
and one intentionally failing static fixture to keep exit-code behavior
visible.

For the core wheel smoke test, CI builds and installs the wheel in an isolated
environment, then runs `init`, `validate`, `run`, and `inspect`. A separate
install matrix checks the core package and the `dense` extra; the dense check
imports the class only and never constructs a model backend.
