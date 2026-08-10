# Retrieval Lab

Retrieval Lab is a local-first Python toolkit for comparing and regression-testing
RAG retrieval strategies on your own corpus.

The project is currently pre-alpha. The supported local pipeline loads TXT,
Markdown, and JSONL, validates graded relevance data, and compares deterministic
Keyword, BM25, Dense, and RRF Hybrid retrieval without network access or paid
services. Exact dense retrieval is available as an optional extra.

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

## Dense retrieval

Install the optional dependency when you want multilingual E5 dense retrieval:

```bash
pip install 'retrieval-lab[dense]'
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

## Scope

v0.1 focuses on retrieval evaluation. CSV/HTML reports, configuration files, CLI
commands, and CI regression gates remain under development. Answer generation and
LLM-based judging are planned only after retrieval evaluation is stable.

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
