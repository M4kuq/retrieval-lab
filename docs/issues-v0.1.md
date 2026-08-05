# v0.1 implementation issues

## Milestone M1 - evaluation core

- [x] Freeze the first public API and evaluation semantics.
- [x] Implement validated domain models and result schema.
- [x] Implement hand-verified ranking metrics.
- [x] Evaluate precomputed ranked results.
- [x] Add document-level collapse and retriever contract validation.

## Milestone M2 - built-in retrieval

- [x] Implement deterministic corpus loaders.
- [x] Implement deterministic chunk identifiers.
- [x] Complete the keyword baseline vertical slice.
- [x] Add BM25 with replaceable tokenization.
- [ ] Add optional Dense exact search.
- [ ] Add RRF Hybrid over a shared chunk artifact.
- [ ] Add content-addressed index caching.

## Milestone M3 - formal Python API and reports

- [ ] Add strict versioned configuration models and safe YAML loading.
- [ ] Add synchronous callable and asynchronous retriever adapters.
- [ ] Add JSON, CSV, and standalone HTML reports.
- [ ] Add manifest and reproducibility metadata.

## Milestone M4 - CLI and CI regression gates

- [ ] Add `init`, `validate`, `run`, `compare`, `inspect`, and `gate`.
- [ ] Enforce exit codes 0, 1, 2, and 3.
- [ ] Reject incomparable runs before calculating regressions.
- [ ] Add a GitHub Actions regression-gate example.

## Milestone M5 - publication quality

- [ ] Complete API reference, tutorials, notebook, and Japanese sample dataset.
- [ ] Verify wheel and sdist in clean environments.
- [ ] Check the final PyPI and GitHub name immediately before publication.
- [ ] Publish TestPyPI, PyPI, documentation, and a tagged GitHub release.
