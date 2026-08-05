# Glossary

- **Corpus**: the normalized source documents used by a retrieval experiment.
- **Document**: one source record with a stable document identifier and body text.
- **Chunk**: a contiguous portion of a document with its own stable identifier and
  parent document identifier.
- **Query case**: one evaluation query plus its known positive relevance judgments.
- **Qrels**: query-to-item relevance judgments, optionally with graded relevance.
- **Retriever**: a component that returns a ranked sequence of chunks for a query.
- **Run**: one immutable retrieval ranking per query plus its evaluation metadata.
- **Relevance level**: whether qrels identify whole documents or exact chunks.
- **Document collapse**: removing later chunks from an already-seen parent document
  before document-level metrics are calculated.
- **Macro average**: the arithmetic mean of a metric calculated independently for
  each query.
- **RRF**: Reciprocal Rank Fusion, which combines rankings using rank positions
  rather than incomparable raw scores.
- **Quality gate**: a CI rule that fails when an absolute threshold or allowed
  regression is violated.
- **Manifest**: reproducibility metadata describing normalized inputs, versions,
  environment, and content hashes for a run.

