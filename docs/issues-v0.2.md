# v0.2 implementation plan

v0.2 adds evaluation-dataset authoring assistance while preserving the local-first, Python-API-first contract.

## Required outcomes

1. Dataset inspection for duplicate queries, relevance concentration, and low-difficulty candidates.
2. Explicit dataset origin and human-review state.
3. Reusable Python APIs for creating and editing relevance judgments.
4. A thin interactive CLI over the Python authoring API.
5. Synthetic dataset generation behind an explicit provider boundary.
6. Synthetic, unreviewed data must remain clearly experimental and must never be silently promoted to trusted ground truth.

## Planned PR sequence

- PR 1: dataset inspection foundation.
- PR 2: provenance and human-review state.
- PR 3: dataset authoring and deterministic JSONL round trips.
- PR 4: interactive relevance selection via application API and thin CLI.
- PR 5: synthetic generation provider boundary and deterministic test provider.
- PR 6: integration, documentation, packaging, and release hardening.

## Non-goals

Answer generation, LLM-as-a-Judge, web crawling, cloud upload, and mandatory external APIs remain outside v0.2.
