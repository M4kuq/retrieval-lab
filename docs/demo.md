# Public demo

Retrieval Lab keeps demo behavior on top of the same public Python API used by notebooks, tests, and applications. The web layer is presentation only.

## Reusable comparison API

Index any supported `BaseRetriever` implementations against the same chunks, then compare one query with a shared cutoff:

```python
from retrieval_lab import (
    BM25Retriever,
    Document,
    FixedSizeChunker,
    KeywordRetriever,
    compare_retrievers_for_query,
)

documents = [
    Document("doc-1", "AWS Secrets Manager stores secrets."),
    Document("doc-2", "Amazon S3 stores objects."),
]
chunks = FixedSizeChunker(size=256, overlap=0).chunk(documents)
retrievers = [KeywordRetriever(), BM25Retriever()]
for retriever in retrievers:
    retriever.index(chunks)

comparison = compare_retrievers_for_query(
    retrievers,
    "AWS secret storage",
    top_k=3,
)

for view in comparison.views:
    print(view.retriever, view.latency_ms)
    for hit in view.results:
        print(hit.rank, hit.document_id, hit.score, hit.text)
```

The returned `DemoComparison` is typed and JSON-compatible through `to_dict()`. Raw scores remain retriever-specific and should not be interpreted as comparable scales across different retrieval algorithms.

## Metric explanations

`retrieval_metric_explanations()` returns short descriptions for Hit Rate, Recall, Precision, MRR, nDCG, and MAP. It is intended for notebooks and presentation layers; metric computation continues to live in the evaluation engine.

## Optional Streamlit example

The repository includes `examples/streamlit_demo.py`. Streamlit is deliberately not a core dependency.

```bash
python -m pip install streamlit
streamlit run examples/streamlit_demo.py
```

The example uses only local sample documents, Keyword retrieval, and BM25. It does not download a model or send documents to an external service.

The demo shows:

- one shared query and Top-K setting;
- side-by-side retriever results;
- rank, document ID, chunk ID, score, and chunk text;
- observed per-retriever search latency;
- retrieval metric explanations.

For real evaluation metrics, use an `EvaluationDataset` with explicit relevance judgments. An ad-hoc query comparison without ground truth is a search-result and latency comparison, not a retrieval-accuracy measurement.
