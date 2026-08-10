# Retrieval Lab 0.1.0rc1

This release candidate packages the local retrieval evaluation workflow for
review before publication. It is not a statement that the package has been
published.

The distribution name is `retrieval-lab-sdk`: the unqualified `retrieval-lab`
name is already occupied by another PyPI project. The import package remains
`retrieval_lab`, the console command remains `retrieval-lab`, and the repository
name remains `retrieval-lab`. The exact distribution name must be checked again
against PyPI immediately before any public upload.

## Highlights

- Deterministic Keyword, BM25, Dense, and RRF Hybrid retrieval over shared
  chunks.
- Typed evaluation results with strict JSON reload and deterministic reports.
- Saved-run comparison with absolute/relative metric deltas and typed quality
  gates.
- Thin `init`, `validate`, `run`, `inspect`, `compare`, and `gate` CLI commands.
- Offline examples, a strict documentation site, and release-candidate
  wheel/sdist validation.

## Constraints

The core package remains local-first and lightweight. Dense retrieval is
optional, and the default Dense backend requires the separate `dense` extra and
an available model cache. CI never loads or downloads a model. Result
comparisons reject mismatched dataset/query/evaluation contracts before
calculating regression values.

## Upgrade and install

For a clean candidate check:

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install retrieval-lab-sdk==0.1.0rc1
```

Optional feature sets are available as `retrieval-lab-sdk[dense]` and
`retrieval-lab-sdk[docs]`. Existing schema-version 1 result files remain the
compatibility boundary; review the comparison contract before using a saved
baseline for a changed dataset or cutoff.

## Verification

The candidate is checked with `uv lock --check`, Ruff, mypy, pytest, strict
MkDocs, sdist/wheel metadata and content inspection, clean-environment installs,
and the wheel CLI smoke path. These checks use no LLM, external retrieval API,
secret, or model download.

## Before publication

Publication is intentionally not included in the release-candidate workflow.
TestPyPI, PyPI, tag creation, GitHub Release creation, and Pages deployment each
require the user's explicit approval at the time of action. Use
`docs/release-checklist.md` as the review record.
