# ADR 0002: Thin CLI adapter

- Status: Accepted
- Date: 2026-08-03

## Decision

CLI commands validate presentation-level arguments, invoke public Python services,
render their results, and map documented exceptions to exit codes. They contain no
retrieval, metric, comparison, or report business logic.

## Consequences

Equivalent Python and CLI inputs must produce the same result schema. CLI tests can
focus on argument and exit-code behavior while core correctness stays in API tests.

