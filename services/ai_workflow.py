"""Tenant-scoped reference workflow connecting retrieval and model inference."""
from __future__ import annotations

import hashlib
import logging
import math
import os
import uuid
from dataclasses import dataclass
from typing import Any

from services.ai_storage import AIDataStore
from services.embeddings import (
    LocalHashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    RAGPipeline,
    semantic_chunk,
)
from services.inference import (
    InferenceRequest,
    ModelRouter,
    estimate_cost,
)
from services.inference.gateway import build_inference_router

logger = logging.getLogger(__name__)


@dataclass
class _TenantKnowledge:
    pipeline: RAGPipeline


class AIWorkflowError(RuntimeError):
    """Raised when an external embedding or inference provider fails."""


class AIWorkflow:
    """Tenant-scoped knowledge retrieval and metered model inference."""

    max_document_chars = 64_000
    max_query_chars = 8_000
    max_documents_per_tenant = 100
    max_indexed_chars_per_tenant = 256_000
    max_tenants = 100

    def __init__(
        self,
        inference_router: ModelRouter | None = None,
        embedding_provider: Any = None,
        data_store: AIDataStore | None = None,
    ):
        self.model = os.getenv("AEGIS_LLM_MODEL", "gpt-4o-mini")
        self.inference_router = inference_router or build_inference_router()
        self.embedding_provider = embedding_provider or self._default_embeddings()
        self.data_store = data_store or AIDataStore()
        self.input_cost_per_1k = self._cost_rate("AEGIS_LLM_INPUT_COST_PER_1K")
        self.output_cost_per_1k = self._cost_rate("AEGIS_LLM_OUTPUT_COST_PER_1K")
        self._tenants: dict[str, _TenantKnowledge] = {}

    @staticmethod
    def _cost_rate(setting: str) -> float:
        try:
            rate = float(os.getenv(setting, "0"))
        except ValueError as exc:
            raise ValueError(f"{setting} must be a non-negative number") from exc
        if not math.isfinite(rate) or rate < 0:
            raise ValueError(f"{setting} must be a non-negative number")
        return rate

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
                    self.data_store.vector_store(tenant_id),
                ),
            )
        return self._tenants[tenant_id]

    def ingest(self, tenant_id: str, document: str) -> dict[str, Any]:
        if not document or not document.strip():
            raise ValueError("document must not be empty")
        if len(document) > self.max_document_chars:
            raise ValueError("document exceeds the maximum length")
        document_id = str(uuid.uuid4())
        chunks = semantic_chunk(document)
        if not chunks:
            raise ValueError("document produced no indexable text")
        vectors = self.embedding_provider.embed(chunks)
        self.data_store.ingest_document(
            tenant_id,
            document_id,
            len(document),
            self.max_documents_per_tenant,
            self.max_indexed_chars_per_tenant,
            chunks,
            vectors,
        )
        return {
            "document_id": document_id,
            "chunks_indexed": len(chunks),
            "embedding_provider": self.embedding_provider.name,
        }

    def answer(self, tenant_id: str, query: str, max_tokens: int = 512) -> dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if len(query) > self.max_query_chars:
            raise ValueError("query exceeds the maximum length")
        knowledge = self._knowledge(tenant_id)
        request_id = str(uuid.uuid4())
        request_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        try:
            retrieved = knowledge.pipeline.retrieve(query, top_k=5)
            context = "\n\n".join(
                f"[{index + 1}] {chunk.text}" for index, chunk in enumerate(retrieved)
            )
            prompt = (
                "Answer the question using only the supplied context. If it is "
                "insufficient, say you do not know.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
            request_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        except Exception as exc:
            self._record_failed_request(
                request_id=request_id,
                tenant_id=tenant_id,
                model=self.model,
                request_sha256=request_sha256,
                status="retrieval_failed",
                error_type=type(exc).__name__,
                provider="not_invoked",
            )
            raise AIWorkflowError("AI workflow retrieval is unavailable") from exc

        try:
            result = self.inference_router.generate(
                InferenceRequest(prompt=prompt, model=self.model, max_tokens=max_tokens)
            )
        except Exception as exc:
            attempts = getattr(self.inference_router, "last_attempts", [])
            provider = attempts[-1].provider if attempts else "unavailable"
            self._record_failed_request(
                request_id=request_id,
                tenant_id=tenant_id,
                provider=provider,
                model=self.model,
                request_sha256=request_sha256,
                status="failed",
                error_type=type(exc).__name__,
            )
            raise AIWorkflowError("AI workflow provider is unavailable") from exc
        cost_usd = estimate_cost(
            result.input_tokens,
            result.output_tokens,
            self.input_cost_per_1k,
            self.output_cost_per_1k,
        )
        response_sha256 = hashlib.sha256(result.text.encode("utf-8")).hexdigest()
        try:
            self.data_store.record_inference(
                request_id=request_id,
                tenant_id=tenant_id,
                provider=result.provider,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                cost_usd=cost_usd,
                status="succeeded",
                request_sha256=request_sha256,
                response_sha256=response_sha256,
            )
        except Exception as exc:
            logger.critical(
                "AI response audit persistence failed request_id=%s provider=%s model=%s "
                "input_tokens=%s output_tokens=%s cost_usd=%s error_type=%s",
                request_id,
                result.provider,
                result.model,
                result.input_tokens,
                result.output_tokens,
                cost_usd,
                type(exc).__name__,
            )
            raise AIWorkflowError("AI usage audit could not be persisted") from exc
        return {
            "request_id": request_id,
            "answer": result.text,
            "model": result.model,
            "provider": result.provider,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "cost_usd": cost_usd,
            "citations": [
                {
                    "document_id": chunk.metadata["document_id"],
                    "chunk_index": chunk.metadata["chunk_index"],
                    "score": chunk.score,
                }
                for chunk in retrieved
            ],
        }

    def list_usage(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        return self.data_store.list_usage(tenant_id, limit)

    def delete_document(self, tenant_id: str, document_id: str) -> bool:
        if not tenant_id or not document_id:
            raise ValueError("tenant_id and document_id are required")
        return self.data_store.remove_document(tenant_id, document_id)

    def _record_failed_request(
        self,
        *,
        request_id: str,
        tenant_id: str,
        provider: str,
        model: str,
        request_sha256: str,
        status: str,
        error_type: str,
    ) -> None:
        try:
            self.data_store.record_inference(
                request_id=request_id,
                tenant_id=tenant_id,
                provider=provider,
                model=model,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0,
                cost_usd=0,
                status=status,
                request_sha256=request_sha256,
                error_type=error_type,
            )
        except Exception as exc:
            logger.error(
                "AI failure audit persistence failed request_id=%s status=%s error_type=%s",
                request_id,
                status,
                type(exc).__name__,
            )
