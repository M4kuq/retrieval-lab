# ADR 0006: Optional, revision-pinned Dense retrieval

- Status: Accepted
- Date: 2026-08-03

## Decision

Dense exact search is an optional feature. The initial multilingual preset may use
`intfloat/multilingual-e5-small`, with explicit query/document prompts. The resolved
model commit revision is recorded in the run manifest.

## Consequences

Installing the core does not install PyTorch. Model download and network use occur
only after the user selects Dense retrieval, and reproducible runs can pin a revision.

