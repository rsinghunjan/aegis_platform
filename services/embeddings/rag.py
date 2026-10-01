"""RAG pipeline: retrieval, ranking, and prompt augmentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .chunking import semantic_chunk
from .providers import EmbeddingProvider
from .vector_store import VectorStore


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: Dict[str, Any]


class RAGPipeline:
    """Ties together chunking, embedding, vector storage, and retrieval."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        chunks = semantic_chunk(document, max_chars=self.chunk_size, overlap_chars=self.chunk_overlap)
        if not chunks:
            return []
        vectors = self.embedding_provider.embed(chunks)
        metadatas = [dict(metadata or {}, chunk_index=i) for i in range(len(chunks))]
        return self.vector_store.upsert(vectors, chunks, metadatas=metadatas)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        query_vector = self.embedding_provider.embed_one(query)
        results = self.vector_store.query(query_vector, top_k=top_k)
        return [
            RetrievedChunk(text=record.text, score=score, metadata=record.metadata)
            for record, score in results
        ]

    def augment_prompt(self, query: str, top_k: int = 5, template: Optional[str] = None) -> str:
        """Build a retrieval-augmented prompt for a downstream LLM call."""
        chunks = self.retrieve(query, top_k=top_k)
        context = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))
        template = template or (
            "Use the following context to answer the question. If the context "
            "does not contain the answer, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        return template.format(context=context, query=query)
