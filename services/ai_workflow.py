"""Tenant-scoped reference workflow connecting retrieval and model inference."""
from __future__ import annotations

import hashlib
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, StrictStr, ValidationError

from safety_checker import SafetyChecker
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
from services.inference.gateway import build_inference_router, dev_providers_allowed
from services.embeddings.rerank import LLMReranker, LexicalReranker
from services.grounding import check_grounding
from services.prompt_registry import PromptRegistry, render
from services.observability.ai_metrics import record_ai_request
from services.observability.tracing import get_trace_id, start_span

logger = logging.getLogger(__name__)


class _ModelAnswer(BaseModel):
    answer: StrictStr = Field(min_length=1)


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
    max_chunks_per_tenant = 512
    max_tenants = 100

    def __init__(
        self,
        inference_router: ModelRouter | None = None,
        embedding_provider: Any = None,
        data_store: AIDataStore | None = None,
    ):
        self.embedding_provider = (
            embedding_provider
            if embedding_provider is not None
            else self._default_embeddings()
        )
        if (
            not dev_providers_allowed()
            and self.embedding_provider.name.lower() == "local-hash"
        ):
            raise RuntimeError(
                "local-hash embeddings are only allowed when "
                "AEGIS_ALLOW_DEV_PROVIDERS=true"
            )
        self.model = os.getenv("AEGIS_LLM_MODEL", "gpt-4o-mini")
        self.inference_router = inference_router or build_inference_router()
        self.data_store = data_store or AIDataStore()
        self.input_cost_per_1k = self._cost_rate("AEGIS_LLM_INPUT_COST_PER_1K")
        self.output_cost_per_1k = self._cost_rate("AEGIS_LLM_OUTPUT_COST_PER_1K")
        self._tenants: dict[str, _TenantKnowledge] = {}
        self._configure_capacity()
        self.reranker = self._build_reranker()
        self.grounding_mode = os.getenv("AEGIS_GROUNDING_MODE", "warn").strip().lower()
        if self.grounding_mode not in {"off", "warn", "enforce"}:
            raise ValueError("AEGIS_GROUNDING_MODE must be off, warn, or enforce")
        self.safety_checker = SafetyChecker()
        self.prompts = PromptRegistry.from_environment()

    def _configure_capacity(self) -> None:
        """PostgreSQL uses an HNSW (pgvector) index, so it can hold far larger corpora
        than the bounded SQLite/dev scan. Each limit is overridable by environment."""
        indexed = self.data_store.engine.dialect.name == "postgresql"
        defaults = {
            "AEGIS_AI_MAX_DOCUMENTS_PER_TENANT": (100, 5_000),
            "AEGIS_AI_MAX_INDEXED_CHARS_PER_TENANT": (256_000, 50_000_000),
            "AEGIS_AI_MAX_CHUNKS_PER_TENANT": (512, 100_000),
        }
        values = {}
        for name, (dev, prod) in defaults.items():
            try:
                values[name] = int(os.getenv(name, str(prod if indexed else dev)))
            except ValueError as exc:
                raise ValueError(f"{name} must be a positive integer") from exc
            if values[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_documents_per_tenant = values["AEGIS_AI_MAX_DOCUMENTS_PER_TENANT"]
        self.max_indexed_chars_per_tenant = values["AEGIS_AI_MAX_INDEXED_CHARS_PER_TENANT"]
        self.max_chunks_per_tenant = values["AEGIS_AI_MAX_CHUNKS_PER_TENANT"]

    def _build_reranker(self):
        mode = os.getenv("AEGIS_RERANKER", "lexical").strip().lower()
        if mode == "none":
            return None
        if mode == "lexical":
            return LexicalReranker()
        if mode == "llm":
            def generate(prompt: str) -> str:
                return self.inference_router.generate(
                    InferenceRequest(prompt=prompt, model=self.model, max_tokens=256)
                ).text

            return LLMReranker(generate)
        raise ValueError("AEGIS_RERANKER must be none, lexical, or llm")

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
    def _is_production() -> bool:
        environment = (
            os.getenv("AEGIS_ENV")
            or os.getenv("ENVIRONMENT")
            or os.getenv("ENV")
            or ""
        )
        production_flag = os.getenv("AEGIS_PRODUCTION", "").lower()
        return environment.strip().lower() in {"prod", "production"} or (
            production_flag in {"1", "true", "yes"}
        )

    @staticmethod
    def _default_embeddings():
        dev = dev_providers_allowed()
        provider_name = os.getenv(
            "AEGIS_EMBEDDING_PROVIDER",
            "local-hash" if dev else "openai",
        ).strip().lower()
        if provider_name == "local-hash":
            if not dev:
                raise RuntimeError(
                    "local-hash embeddings are only allowed when "
                    "AEGIS_ALLOW_DEV_PROVIDERS=true"
                )
            return LocalHashEmbeddingProvider()
        if provider_name == "openai":
            provider = OpenAIEmbeddingProvider(
                api_key=os.getenv("AEGIS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            )
        else:
            raise ValueError(f"unsupported embedding provider: {provider_name}")
        if not provider.is_available():
            raise RuntimeError(f"{provider_name} embedding provider is unavailable")
        return provider

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
        try:
            vectors = self.embedding_provider.embed(chunks)
        except Exception as exc:
            raise AIWorkflowError(
                "AI workflow embedding provider is unavailable"
            ) from exc
        self.data_store.ingest_document(
            tenant_id,
            document_id,
            len(document),
            self.max_documents_per_tenant,
            self.max_indexed_chars_per_tenant,
            self.max_chunks_per_tenant,
            chunks,
            vectors,
        )
        return {
            "document_id": document_id,
            "chunks_indexed": len(chunks),
            "embedding_provider": self.embedding_provider.name,
        }

    def answer(
        self,
        tenant_id: str,
        query: str,
        max_tokens: int = 512,
        prompt_template: str | None = None,
    ) -> dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if len(query) > self.max_query_chars:
            raise ValueError("query exceeds the maximum length")
        if self.safety_checker.is_unsafe(query):
            raise ValueError("query rejected by safety policy")
        knowledge = self._knowledge(tenant_id)
        request_id = str(uuid.uuid4())
        request_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        started_at = time.perf_counter()
        try:
            with start_span("ai.retrieval", {"ai.query.length": len(query)}):
                retrieved = knowledge.pipeline.retrieve(
                    query, top_k=20 if self.reranker else 5
                )
                if self.reranker:
                    retrieved = self.reranker.rerank(query, retrieved, 5)
            context = "\n\n".join(
                f"[{index + 1}] {chunk.text}" for index, chunk in enumerate(retrieved)
            )
            prompt = render(
                prompt_template or self.prompts.active()["template"], context, query
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
            record_ai_request("retrieval_failed", time.perf_counter() - started_at)
            raise AIWorkflowError("AI workflow retrieval is unavailable") from exc

        if self.safety_checker.is_unsafe(context):
            self._record_failed_request(
                request_id=request_id,
                tenant_id=tenant_id,
                model=self.model,
                request_sha256=request_sha256,
                status="safety_blocked",
                error_type="UnsafeRetrievedContext",
                provider="not_invoked",
            )
            raise ValueError("retrieved context rejected by safety policy")

        try:
            with start_span(
                "ai.provider",
                {"ai.provider": "router", "ai.model": self.model},
            ) as span:
                result = self.inference_router.generate(
                    InferenceRequest(
                        prompt=prompt, model=self.model, max_tokens=max_tokens
                    )
                )
                span.set_attribute("ai.provider", result.provider)
        except Exception as exc:
            attempts = getattr(exc, "attempts", ())
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
            record_ai_request("failed", time.perf_counter() - started_at)
            raise AIWorkflowError("AI workflow provider is unavailable") from exc
        provider = getattr(result, "provider", "unknown")
        model = getattr(result, "model", self.model)
        try:
            answer = _ModelAnswer(answer=result.text).answer
        except (ValidationError, AttributeError, TypeError) as exc:
            self._record_failed_request(
                request_id=request_id,
                tenant_id=tenant_id,
                provider=provider,
                model=model,
                request_sha256=request_sha256,
                status="invalid_output",
                error_type="InvalidModelOutput",
            )
            raise AIWorkflowError("AI workflow returned invalid model output") from exc
        if not answer.strip():
            self._record_failed_request(
                request_id=request_id,
                tenant_id=tenant_id,
                provider=provider,
                model=model,
                request_sha256=request_sha256,
                status="invalid_output",
                error_type="InvalidModelOutput",
            )
            raise AIWorkflowError("AI workflow returned invalid model output")
        if self.safety_checker.is_unsafe(answer):
            self._record_failed_request(
                request_id=request_id,
                tenant_id=tenant_id,
                provider=provider,
                model=model,
                request_sha256=request_sha256,
                status="safety_blocked",
                error_type="UnsafeModelAnswer",
            )
            raise ValueError("model answer rejected by safety policy")
        grounding = (
            check_grounding(answer, context) if self.grounding_mode != "off" else None
        )
        if grounding and not grounding["grounded"] and self.grounding_mode == "enforce":
            self._record_failed_request(
                request_id=request_id,
                tenant_id=tenant_id,
                provider=provider,
                model=model,
                request_sha256=request_sha256,
                status="ungrounded",
                error_type="UngroundedModelAnswer",
            )
            raise AIWorkflowError("AI workflow answer is not grounded in the context")
        cost_usd = estimate_cost(
            result.input_tokens,
            result.output_tokens,
            self.input_cost_per_1k,
            self.output_cost_per_1k,
        )
        response_sha256 = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        try:
            with start_span(
                "ai.audit.persist",
                {"ai.request_id": request_id, "ai.audit.status": "succeeded"},
            ):
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
                    trace_id=get_trace_id(),
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
            record_ai_request(
                "audit_failed",
                time.perf_counter() - started_at,
                cost_usd=cost_usd,
            )
            raise AIWorkflowError("AI usage audit could not be persisted") from exc
        record_ai_request(
            "succeeded", time.perf_counter() - started_at, cost_usd=cost_usd
        )
        return {
            "request_id": request_id,
            "trace_id": get_trace_id(),
            "answer": answer,
            "model": result.model,
            "provider": result.provider,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "cost_usd": cost_usd,
            "grounding": grounding,
            "citations": [
                {
                    "document_id": chunk.metadata["document_id"],
                    "chunk_index": chunk.metadata["chunk_index"],
                    "score": chunk.score,
                }
                for chunk in retrieved
            ],
        }

    def generate(
        self, tenant_id: str, prompt: str, max_tokens: int = 512
    ) -> dict[str, Any]:
        if not tenant_id or not prompt:
            raise ValueError("tenant_id and prompt are required")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        request_id = str(uuid.uuid4())
        request_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        try:
            result = self.inference_router.generate(
                InferenceRequest(prompt=prompt, model=self.model, max_tokens=max_tokens)
            )
        except Exception as exc:
            attempts = getattr(exc, "attempts", ())
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
            "text": result.text,
            "model": result.model,
            "provider": result.provider,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "cost_usd": cost_usd,
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
            with start_span(
                "ai.audit.persist",
                {"ai.request_id": request_id, "ai.audit.status": status},
            ):
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
                    trace_id=get_trace_id(),
                )
        except Exception as exc:
            logger.error(
                "AI failure audit persistence failed request_id=%s status=%s error_type=%s",
                request_id,
                status,
                type(exc).__name__,
            )
