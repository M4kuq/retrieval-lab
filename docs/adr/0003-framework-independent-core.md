# ADR 0003: Framework-independent core

- Status: Accepted
- Date: 2026-08-03

## Decision

The core does not depend on LangChain, LlamaIndex, a vector database SDK, or a cloud
provider. Existing systems integrate through typed retriever/result contracts.

## Consequences

Core installation stays small and local. Provider-specific convenience adapters may
be optional packages, but the canonical evaluation path accepts plain Python values.

