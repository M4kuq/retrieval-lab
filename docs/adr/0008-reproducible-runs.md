# ADR 0008: Content-addressed reproducibility

- Status: Accepted
- Date: 2026-08-03

## Decision

Corpora, chunks, indexes, datasets, and run inputs use canonical content hashes.
Manifests record normalized configuration, implementation versions, model revision,
Python/OS/dependency versions, seed, timing, and artifact hashes. Secrets and
environment-variable values are never recorded.

## Consequences

Cache reuse is explainable and stale artifacts are detected. Run comparison rejects
different dataset hashes, query IDs, relevance levels, metric versions, or cutoffs.

