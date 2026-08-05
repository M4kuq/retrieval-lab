# Retrieval Lab development instructions

## Product contract

- Retrieval Lab is a local-first Python SDK for comparing and regression-testing RAG retrieval strategies on a user's own corpus.
- The public Python API is the primary application interface. Any CLI must be a thin adapter over that API and must not contain evaluation logic.
- v0.1 covers retrieval evaluation only. Do not add answer generation, LLM judges, web crawling, cloud upload, or provider-specific vector database dependencies.
- Core behavior must work without network access, paid APIs, or heavyweight ML dependencies.
- Keep domain models and metric logic independent from optional external libraries.

## Implementation rules

- Support Python 3.11 and newer.
- Use the `src` package layout and expose supported APIs from `retrieval_lab.__init__`.
- Add complete type annotations and concise public docstrings. Avoid broad `Any` types.
- Preserve deterministic ordering, stable identifiers, and explicit schema versions.
- Raise Retrieval Lab exceptions at public boundaries; do not silently swallow errors.
- Never use `print` inside library code. Callers own presentation and logging.
- Add or update tests for every behavior change. Metric tests must include hand-calculated examples.
- Do not disable tests or type/lint checks to make a change pass.
- Do not commit, push, force-push, or modify `main`.

## Source-of-truth documents

- `docs/product-plan.md`: product scope and roadmap.
- `docs/technical-design.md`: precise v0.1 contracts.
- `docs/api-minimum.md`: current public API boundary.
- `docs/evaluation-spec.md`: metric and relevance semantics.
- `docs/adr/`: accepted architectural decisions.

When documents conflict, use this priority: `docs/technical-design.md`, accepted ADRs, `docs/product-plan.md`.

## Verification

Run these commands from the repository root:

```bash
pytest
ruff check .
ruff format --check .
mypy src
python -m build
```
