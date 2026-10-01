"""Vector & embeddings layer: providers, vector stores, chunking, RAG."""

from .providers import (
    EmbeddingProvider,
    LocalHashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    CohereEmbeddingProvider,
)
from .vector_store import (
    VectorRecord,
    VectorStore,
    InMemoryVectorStore,
    PGVectorStore,
)
from .chunking import semantic_chunk
from .rag import RAGPipeline, RetrievedChunk

__all__ = [
    "EmbeddingProvider",
    "LocalHashEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "CohereEmbeddingProvider",
    "VectorRecord",
    "VectorStore",
    "InMemoryVectorStore",
    "PGVectorStore",
    "semantic_chunk",
    "RAGPipeline",
    "RetrievedChunk",
]
