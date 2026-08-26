"""Optional Streamlit UI over Retrieval Lab's reusable demo comparison API.

Run after installing Streamlit separately:

    python -m pip install streamlit
    streamlit run examples/streamlit_demo.py
"""

from retrieval_lab import (
    BM25Retriever,
    Document,
    FixedSizeChunker,
    KeywordRetriever,
    compare_retrievers_for_query,
    retrieval_metric_explanations,
)


def _sample_retrievers():
    documents = [
        Document(
            "secrets",
            "AWS Secrets Manager stores, rotates, and retrieves secrets securely.",
        ),
        Document(
            "s3",
            "Amazon S3 is an object-storage service for durable data storage.",
        ),
        Document(
            "rag",
            "RAG systems should measure retrieval quality separately from generation.",
        ),
    ]
    chunks = FixedSizeChunker(size=256, overlap=0).chunk(documents)
    retrievers = [KeywordRetriever(), BM25Retriever()]
    for retriever in retrievers:
        retriever.index(chunks)
    return retrievers


def main() -> None:
    """Render the optional web demo without adding web logic to the SDK core."""

    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is optional. Install it with `python -m pip install streamlit`."
        ) from exc

    st.set_page_config(page_title="Retrieval Lab demo", layout="wide")
    st.title("Retrieval Lab")
    st.caption("One query, shared corpus, side-by-side retrieval results")

    query = st.text_input("Query", value="AWS secret storage")
    top_k = st.slider("Top K", min_value=1, max_value=5, value=3)
    comparison = compare_retrievers_for_query(
        _sample_retrievers(),
        query,
        top_k=top_k,
    )

    columns = st.columns(len(comparison.views))
    for column, view in zip(columns, comparison.views, strict=True):
        with column:
            st.subheader(view.retriever)
            st.metric("Search latency", f"{view.latency_ms:.3f} ms")
            if not view.results:
                st.info("No matching chunks")
            for result in view.results:
                st.markdown(f"**#{result.rank} · {result.document_id}**")
                st.caption(f"score={result.score:.6g} · chunk={result.chunk_id}")
                st.write(result.text)

    with st.expander("Retrieval metric glossary"):
        for name, explanation in retrieval_metric_explanations().items():
            st.markdown(f"**{name}** — {explanation}")


if __name__ == "__main__":
    main()
