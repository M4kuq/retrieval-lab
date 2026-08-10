# Retrieval Lab

Retrieval Lab is a local-first Python toolkit for comparing and
regression-testing RAG retrieval strategies on your own corpus.

The supported local pipeline evaluates Keyword, BM25, Dense, and RRF Hybrid
retrieval using deterministic shared metrics. Core installation has no model
download or external service requirement; Dense execution is an optional
feature and can use an injected local embedding backend.

## Start here

- Follow the [offline tutorial](https://github.com/M4kuq/retrieval-lab/blob/main/docs/tutorial.md) for a complete local evaluation.
- Read the [minimum API](https://github.com/M4kuq/retrieval-lab/blob/main/docs/api-minimum.md) for typed Python contracts.
- Use the [CI regression guide](https://github.com/M4kuq/retrieval-lab/blob/main/docs/ci-regression.md) for saved-run quality gates.
- Check the [FAQ](https://github.com/M4kuq/retrieval-lab/blob/main/docs/faq.md) for relevance, latency, and optional Dense behavior.

All examples and documentation builds are offline-safe. Saved results are
schema-versioned JSON and can be inspected or compared without rerunning
retrieval.

## Install

```console
python -m pip install retrieval-lab-sdk==0.1.0rc1
```

The import package is `retrieval_lab`. The optional Dense and documentation
extras are `retrieval-lab-sdk[dense]` and `retrieval-lab-sdk[docs]`.

## First CLI run

```console
retrieval-lab init ./my-evaluation
retrieval-lab validate --config ./my-evaluation/retrieval-lab.yaml
retrieval-lab run --config ./my-evaluation/retrieval-lab.yaml
```

The [tutorial](https://github.com/M4kuq/retrieval-lab/blob/main/docs/tutorial.md),
[FAQ](https://github.com/M4kuq/retrieval-lab/blob/main/docs/faq.md),
[API minimum](https://github.com/M4kuq/retrieval-lab/blob/main/docs/api-minimum.md),
and [CI regression guide](https://github.com/M4kuq/retrieval-lab/blob/main/docs/ci-regression.md)
provide the next level of detail.
