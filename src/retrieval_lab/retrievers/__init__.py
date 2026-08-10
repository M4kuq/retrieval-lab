"""Built-in retrieval strategies."""

from retrieval_lab.retrievers.base import BaseRetriever
from retrieval_lab.retrievers.bm25 import BM25Retriever, Tokenizer
from retrieval_lab.retrievers.callable import (
    CallableRetriever,
    RetrievedItem,
    Retriever,
    evaluate_retrievers,
)
from retrieval_lab.retrievers.dense import (
    DenseRetriever,
    EmbeddingBackend,
    EmbeddingModelMetadata,
)
from retrieval_lab.retrievers.hybrid import HybridRetriever
from retrieval_lab.retrievers.keyword import KeywordRetriever

__all__ = [
    "BM25Retriever",
    "BaseRetriever",
    "CallableRetriever",
    "DenseRetriever",
    "EmbeddingBackend",
    "EmbeddingModelMetadata",
    "HybridRetriever",
    "KeywordRetriever",
    "RetrievedItem",
    "Retriever",
    "Tokenizer",
    "evaluate_retrievers",
]
