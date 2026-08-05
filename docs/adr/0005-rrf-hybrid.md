# ADR 0005: Reciprocal Rank Fusion for v0.1 Hybrid

- Status: Accepted
- Date: 2026-08-03

## Decision

The first Hybrid retriever combines BM25 and Dense rankings with Reciprocal Rank
Fusion. Each source retrieves a shared `candidate_k`; RRF uses an explicit `rrf_k`.

## Consequences

BM25 and cosine scores are never added as though they shared a scale. Weighted score
fusion remains a later, separately evaluated strategy.

