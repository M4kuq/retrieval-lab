# ADR 0007: Optional dependency boundaries

- Status: Accepted
- Date: 2026-08-03

## Decision

The core uses the minimum dependencies needed for validated datasets and evaluation.
Dense, Japanese tokenization, reports, notebooks, and future PDF support use separate
extras. Missing extras raise a clear `OptionalDependencyError` with an install hint.

## Consequences

Users pay installation and security costs only for selected features. CI must verify
both core-only and extra-enabled environments.

