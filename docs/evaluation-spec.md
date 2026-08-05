# Evaluation specification

## Relevance levels

The default level is `document`. A ranked chunk list is collapsed to parent
documents by keeping only the first occurrence of each document. This prevents a
retriever from gaining artificial credit by returning many chunks from one relevant
document.

Chunk relevance compares `SearchResult.chunk_id` directly and is introduced after
the initial vertical slice. Evaluation must never infer the relevance level from
which fields happen to be populated.

## Stable ranking

Retrievers order by descending score and then ascending chunk identifier. Ranks are
recomputed after stable ordering and after document collapse. NaN, infinity,
duplicate chunk identifiers, invalid ranks, and empty identifiers violate the
retriever contract.

## Metrics

For a ranked identifier sequence `R`, positive relevant set `G`, and cutoff `k`:

- HitRate@k is 1 when `R[:k]` intersects `G`, otherwise 0.
- Recall@k is `|R[:k] intersect G| / |G|`.
- Precision@k is `|R[:k] intersect G| / k`; missing positions count as non-relevant.
- MRR@k is the reciprocal rank of the first relevant item in `R[:k]`, or 0.
- nDCG@k uses gain `2**relevance - 1` and logarithmic discount `log2(rank + 1)`.
- AP@k averages precision at each relevant hit and divides by `min(|G|, k)`.

The dataset contract requires at least one positive relevant item, so public metric
functions reject an empty relevant set instead of inventing a denominator.

Aggregate quality values are macro averages across queries. A query is never
silently removed from the denominator because retrieval failed.

## Top-k behavior

Cutoffs are unique positive integers and are normalized into ascending order. A
retriever is called once per query with `max(top_k)`; smaller cutoffs slice that
same ranking.

## Keyword baseline

The initial deterministic keyword baseline normalizes query and text with Unicode
NFC and `casefold()`. Query terms are split on whitespace. Each distinct term that
appears as a substring of a chunk contributes one point. Zero-score chunks are not
returned. This is a transparent baseline, not a replacement for BM25.

## BM25 baseline

BM25 uses `log(1 + (N - df + 0.5) / (df + 0.5))`, term-frequency saturation,
and document-length normalization. Scores are ordered descending with chunk ID as
the deterministic tie-break.

The default tokenizer applies Unicode NFC normalization and `casefold()`. Latin
letters and numbers form word tokens. Contiguous CJK text emits character unigrams
and adjacent bigrams so Japanese text without spaces is not reduced to one
sentence-sized token. Callers may replace the tokenizer.

## Shared evaluation engine

Runner-based and precomputed-result evaluation calculate HitRate, Recall,
Precision, MRR, nDCG, and AP through the same implementation. Their dataset hash
uses canonical query text, metadata, and graded relevance, allowing the same qrels
to be identified across both entry points.
