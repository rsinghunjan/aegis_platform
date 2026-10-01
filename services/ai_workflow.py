"""Tenant-scoped reference workflow connecting retrieval and model inference."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

from services.embeddings import (
    InMemoryVectorStore,
    LocalHashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    RAGPipeline,
)
from services.inference import (
    EchoProvider,
    InferenceRequest,
    ModelRouter,
    OpenAICompatibleProvider,
)


@dataclass
class _TenantKnowledge:
    pipeline: RAGPipeline
    document_ids: set[str]
    indexed_chars: int


class AIWorkflowError(RuntimeError):
    """Raised when an external embedding or inference provider fails."""


class AIWorkflow:
    """Small local workflow for indexing tenant documents and answering questions.

    This reference implementation keeps its vector index in process memory.
    Production deployments should inject a durable vector store with tenant
    filtering rather than share this local implementation across workers.
    """

    max_document_chars = 64_000
    max_query_chars = 8_000
    max_documents_per_tenant = 100
    max_indexed_chars_per_tenant = 256_000
    max_tenants = 100

    def __init__(
        self,
        inference_router: ModelRouter | None = None,
        embedding_provider: Any = None,
    ):
        self.model = os.getenv("AEGIS_LLM_MODEL", "gpt-4o-mini")
        if inference_router is None:
            embedding_provider = embedding_provider or self._default_embeddings()
            provider = OpenAICompatibleProvider(
                api_key=os.getenv("AEGIS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("AEGIS_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            )
            inference_router = ModelRouter(
                [provider] if provider.is_available() else [EchoProvider()]
            )
        else:
            embedding_provider = embedding_provider or LocalHashEmbeddingProvider()
        self.inference_router = inference_router
        self.embedding_provider = embedding_provider
        self._tenants: dict[str, _TenantKnowledge] = {}

    @staticmethod
    def _default_embeddings():
        if os.getenv("AEGIS_EMBEDDING_PROVIDER", "local-hash").lower() == "openai":
            provider = OpenAIEmbeddingProvider(
                api_key=os.getenv("AEGIS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            )
            if provider.is_available():
                return provider
        return LocalHashEmbeddingProvider()

    def _knowledge(self, tenant_id: str) -> _TenantKnowledge:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if tenant_id not in self._tenants:
            if len(self._tenants) >= self.max_tenants:
                raise ValueError("AI workflow tenant limit reached")
            self._tenants[tenant_id] = _TenantKnowledge(
                pipeline=RAGPipeline(
                    self.embedding_provider,
                    InMemoryVectorStore(),
                ),
                document_ids=set(),
                indexed_chars=0,
            )
        return self._tenants[tenant_id]

    def ingest(self, tenant_id: str, document: str) -> dict[str, Any]:
        if not document or not document.strip():
            raise ValueError("document must not be empty")
        if len(document) > self.max_document_chars:
            raise ValueError("document exceeds the maximum length")
        knowledge = self._knowledge(tenant_id)
        if len(knowledge.document_ids) >= self.max_documents_per_tenant:
            raise ValueError("tenant document limit reached")
        if knowledge.indexed_chars + len(document) > self.max_indexed_chars_per_tenant:
            raise ValueError("tenant indexed text limit reached")
        document_id = str(uuid.uuid4())
        chunk_ids = knowledge.pipeline.ingest(
            document, metadata={"tenant_id": tenant_id, "document_id": document_id}
        )
        if not chunk_ids:
            raise ValueError("document produced no indexable text")
        knowledge.document_ids.add(document_id)
        knowledge.indexed_chars += len(document)
        return {
            "document_id": document_id,
            "chunks_indexed": len(chunk_ids),
            "embedding_provider": self.embedding_provider.name,
        }

    def answer(self, tenant_id: str, query: str, max_tokens: int = 512) -> dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if len(query) > self.max_query_chars:
            raise ValueError("query exceeds the maximum length")
        knowledge = self._knowledge(tenant_id)
        try:
            retrieved = knowledge.pipeline.retrieve(query, top_k=5)
        except Exception as exc:
            raise AIWorkflowError("AI workflow provider is unavailable") from exc
        context = "\n\n".join(
            f"[{index + 1}] {chunk.text}" for index, chunk in enumerate(retrieved)
        )
        prompt = (
            "Answer the question using only the supplied context. If it is "
            "insufficient, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        try:
            result = self.inference_router.generate(
                InferenceRequest(prompt=prompt, model=self.model, max_tokens=max_tokens)
            )
        except Exception as exc:
            raise AIWorkflowError("AI workflow provider is unavailable") from exc
        return {
            "answer": result.text,
            "model": result.model,
            "provider": result.provider,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "citations": [
                {
                    "document_id": chunk.metadata["document_id"],
                    "chunk_index": chunk.metadata["chunk_index"],
                    "score": chunk.score,
                }
                for chunk in retrieved
            ],
        }
