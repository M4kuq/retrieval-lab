# ADR 0001: Python API first

- Status: Accepted
- Date: 2026-08-03

## Decision

The supported Python API is Retrieval Lab's primary application interface. Public
types are re-exported from the package root and carry type annotations and semantic
versioning commitments.

## Consequences

Notebooks, tests, services, and future UIs can embed the same evaluation engine.
API design must be completed before a CLI command is considered finished.

