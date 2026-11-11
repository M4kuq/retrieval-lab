# Retrieval Lab

Retrieval Lab is a local-first Python toolkit for comparing and regression-testing
RAG retrieval strategies on your own corpus.

The project is in alpha. Install the distribution package
`retrieval-lab-sdk` (`pip install retrieval-lab-sdk`); the import package is
`retrieval_lab`. The supported local pipeline loads TXT, Markdown, and JSONL,
validates graded relevance data, and compares deterministic Keyword, BM25,
Dense, and RRF Hybrid retrieval without network access or paid services.
Exact dense retrieval is available as an optional extra.

## Quick start

```python
from retrieval_lab import Document, EvaluationQuery, EvaluationRunner

result = EvaluationRunner.quick_evaluate(
    documents=[
        Document(
            id="doc-1",
            text="RAGでは検索品質の評価が重要です。",
        ),
        Document(
            id="doc-2",
            text="Pythonではpytestを利用できます。",
        ),
    ],
    queries=[
        EvaluationQuery(
            id="q-1",
            query="RAG 検索品質",
            relevant_document_ids={"doc-1"},
        ),
    ],
    strategies=["keyword", "bm25"],
    top_k=[1],
)

assert result.metrics["keyword"].recall_at(1) == 1.0
assert result.metrics["bm25"].recall_at(1) == 1.0
result.save_json("artifacts/result.json")
```

## Evaluate local files

The canonical dataset format is one JSON object per line. Each query has one or
more positive relevance judgments; integer grades greater than one are used by
nDCG.

```json
{"query_id":"q-1","query":"AWSで機密情報を管理するサービスは？","relevant":[{"id":"security.md","relevance":3}]}
```

```python
from retrieval_lab import EvaluationDataset, EvaluationRunner, load_documents

documents = load_documents("examples/corpus")
dataset = EvaluationDataset.from_jsonl("examples/evaluation.jsonl")

result = EvaluationRunner.from_dataset(
    documents=documents,
    dataset=dataset,
    strategies=["keyword", "bm25"],
    top_k=[1, 3],
).run()
```

Loading and validation reject duplicate IDs, blank records, invalid UTF-8 or JSON,
invalid relevance grades, and gold IDs that do not exist in the corpus. Errors
include the source path and line when available.

## Evaluate existing search results

Existing search APIs and vector databases can be evaluated without building an
index or running retrieval:

```python
from retrieval_lab import RetrievedQueryResult, evaluate_results

result = evaluate_results(
    dataset=dataset,
    retrieved_results=[
        RetrievedQueryResult(
            query_id="q-1",
            retrieved_document_ids=["security.md", "storage.md"],
        )
    ],
    top_k=[1, 3],
)
```

Every evaluation path reports HitRate, Recall, Precision, MRR, nDCG, and AP using
the same shared metric engine and deterministic dataset hash.

## Compare saved runs

Saved result JSON files can be compared without indexing or searching again:

```python
from retrieval_lab import compare_runs, load_result

comparison = compare_runs(
    load_result("baseline/result.json"),
    load_result("candidate/result.json"),
)
```

Comparisons require matching dataset/query/relevance/metric/top-k contracts and
report retriever, corpus, chunk, configuration, and latency differences
separately. A comparison can also be rendered as a standalone HTML diff report
with `comparison.to_html()` or saved with `comparison.save_html(path)`.
`bootstrap_metric_delta()` resamples the paired per-query deltas of a metric
comparison and reports a bootstrap confidence interval, a two-sided p-value,
and a significance flag, deterministically under a caller seed.

To evaluate a synchronous search callable directly, return `RetrievedItem`
records in the provider's best-first order:

```python
from retrieval_lab import CallableRetriever, RetrievedItem, evaluate_retrievers

search = CallableRetriever(
    "production",
    lambda query, *, top_k: [
        RetrievedItem(id=row.chunk_id, parent_document_id=row.document_id)
        for row in existing_search(query=query, limit=top_k)
    ],
)
result = evaluate_retrievers(
    dataset=dataset,
    retrievers={"production": search},
    top_k=[1, 3, 5],
)
```

This adapter has no provider SDK dependency and requires no corpus or index.
Each query is called once at `max(top_k)`; the returned sequence order is the
ranking order.

For an async provider callable, use the separate async entry point from an
existing event loop:

```python
from retrieval_lab import (
    AsyncCallableRetriever,
    RetrievedItem,
    evaluate_async_retrievers,
)


async def search(query, *, top_k):
    return await provider_search(query, limit=top_k)


async_search = AsyncCallableRetriever("production", search)
result = await evaluate_async_retrievers(
    dataset=dataset,
    retrievers={"production": async_search},
    top_k=[1, 3, 5],
    concurrency=8,
)
```

`evaluate_async_retrievers()` is awaitable and does not create or replace an
event loop. The synchronous `evaluate_retrievers()` entry point remains
separate. Repeated async calls must return the same ranking; a timeout or one
failed query fails the whole evaluation. Caller cancellation is propagated
after child tasks are cleaned up.

## Dense retrieval

Install the optional dependency when you want multilingual E5 dense retrieval:

```bash
pip install 'retrieval-lab-sdk[dense]'
```

`DenseRetriever` uses `intfloat/multilingual-e5-small` by default and lazily
loads the model on its first encode operation. You can inject an
`EmbeddingBackend` to use a custom embedding service without adding a vector
database dependency.

## Hybrid retrieval

`HybridRetriever` combines two or more indexed `BaseRetriever` instances with
Reciprocal Rank Fusion. Every source receives the same chunks and candidate
cutoff; fusion uses source ranks rather than raw scores, with deterministic
chunk-ID tie-breaking.

```python
from retrieval_lab import BM25Retriever, DenseRetriever, HybridRetriever

hybrid = HybridRetriever(
    [BM25Retriever(), DenseRetriever(backend=my_backend)],
    rrf_k=60,
    candidate_k=100,
)
```

## Content-addressed cache

Pass `cache_dir` to reuse validated chunk artifacts and Dense embeddings across
identical runs. Cache files are schema-versioned canonical JSON, and corrupt or
incompatible entries are rebuilt rather than used.

```python
runner = EvaluationRunner.from_dataset(
    documents=documents,
    dataset=dataset,
    retrievers=[hybrid],
    top_k=[1, 3, 5],
    cache_dir=".retrieval-lab/cache",
)
result = runner.run()
```

Cache hit, miss, and rebuild events are recorded under `manifest["runtime"]`
without changing the deterministic `run_id`.

## Strict YAML configuration

Installations include safe, schema-versioned YAML configuration support. Paths
are resolved relative to the configuration file; unknown fields, duplicate keys,
unsafe YAML tags, unsupported retriever settings, and invalid cross-references are
rejected. Environment variables are never expanded implicitly.

```yaml
schema_version: 1
corpus:
  path: ./examples/corpus
  include: ["**/*.md"]
  chunker:
    type: recursive_characters
    size: 512
    overlap: 64
dataset:
  path: ./examples/evaluation.jsonl
  format: native_jsonl
  relevance_level: document
retrievers:
  - name: keyword
    type: keyword
  - name: bm25
    type: bm25
evaluation:
  top_k: [1, 3, 5, 10]
  metrics: [hit_rate, recall, precision, mrr, ndcg, ap]
  repetitions: 1
  concurrency: 1
```

```python
from retrieval_lab import EvaluationRunner

result = EvaluationRunner.from_config("retrieval-lab.yaml").run()
```

Built-in names are canonical (`keyword`, `bm25`, `dense`, and `hybrid`) in
v0.1. YAML uses the dependency-free BM25 tokenizer; custom tokenizers remain
available through the Python API. Quality-gate and report blocks are validated
and recorded now, with execution added by their dedicated APIs.

## Result reports

Results can be safely reloaded and rendered without exposing query or document
text. JSON loading validates schema version 1, rejects duplicate keys and
non-finite numbers, and limits file loads to 64 MiB by default:

```python
from retrieval_lab import EvaluationResult, load_result

same_result = load_result("artifacts/result.json")
# Equivalent: EvaluationResult.load_json("artifacts/result.json")
print(same_result.summary())
same_result.save_csv("artifacts/reports")
same_result.save_html("artifacts/reports/report.html")
```

`save_summary_csv()` and `save_per_query_csv()` are also available when separate
paths are preferred. CSV output is UTF-8 long-form data with spreadsheet formula
injection protection. HTML is a standalone UTF-8 document with inline CSS only;
it shows metrics, latency, warnings, and a safe allowlist of normalized config
fields. All save methods use atomic same-directory replacement.

## Benchmark datasets

Standard benchmark layouts load from disk into the native dataset models:

- `load_beir_dataset()` reads a BEIR-format directory (`corpus.jsonl`,
  `queries.jsonl`, `qrels/<split>.tsv`).
- `load_msmarco_dataset()` reads an MS MARCO passage-ranking directory
  (`collection.tsv`, `queries.<split>.tsv`, TREC-format `qrels.<split>.tsv`).
- `load_natural_questions_dataset()` reads a simplified Natural Questions
  JSONL file, using each example's page as a corpus document and its
  long-answer annotation as a grade-1 judgment.

Queries without positive judgments in the split are reported as
`excluded_query_ids` instead of being silently dropped.

## Evaluation-side fusion

`fuse_rankings()` and `fuse_results()` combine saved rankings from several
retrievers with reciprocal-rank fusion, using exact rational arithmetic and
identifier tie-breaking. `evaluate_fused_results()` scores the fused ranking
through the shared metric engine and records the fusion parameters in the
result manifest, so a hybrid strategy can be regression-tested from saved
runs without re-running retrieval.

## Checkpoints for large evaluations

`run_with_checkpoint()` records each finished query's ranking to a
fingerprint-pinned JSONL checkpoint (`EvaluationCheckpoint`), flushed and
fsynced per record. Re-running after an interruption evaluates only the
queries still missing, so long evaluations over large corpora can resume
instead of starting over.

## Answer evaluation and judges

Generated answers can be scored without any provider dependency:
`evaluate_answer()` and `evaluate_answers()` compute deterministic
`exact_match` and `token_f1`, answers come from a caller-supplied
`CallableGenerator`, and `evaluate_rag_answer()` bundles the RAGAS-style
`faithfulness()`, `context_precision()`, and `context_recall()` formulas over
caller-supplied claim assessments. `judge_answer()` and
`judge_answer_with_panel()` run caller-controlled LLM judges through a
validated prompt template and return typed verdicts with per-judge usage
records; no provider client is bundled.

## Scope

Retrieval evaluation is the core: local-first, deterministic, and usable
without network access, paid APIs, or heavyweight ML dependencies. The CLI is
a thin adapter over the public Python API and supports project scaffolding,
validation, configured runs, inspection, comparison, and quality gates for CI
regression checks. Answer evaluation and judging stay provider-independent —
callers supply the generator, the claim assessments, and the judges.

See `docs/product-plan.md` and `docs/technical-design.md` for the full roadmap and
contracts.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run python -m build
```

## License

Apache-2.0.
