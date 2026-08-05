# ADR 0004: Explicit document and chunk relevance

- Status: Accepted
- Date: 2026-08-03

## Decision

Datasets declare `document` or `chunk` relevance. Document evaluation collapses a
chunk ranking to the first occurrence of each parent document. Chunk evaluation uses
chunk identifiers directly and requires a matching chunk-artifact hash.

## Consequences

Fine-grained chunking cannot inflate document metrics. Results from incompatible
chunk definitions are rejected rather than compared as if they were equivalent.

