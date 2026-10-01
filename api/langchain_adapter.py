  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
"""
Aegis LLM adapter and a tiny "LangChain-like" SDK surface for app composition.

Provides:
- AegisLLM: simple client wrapper for Aegis predict endpoints (sync, httpx).
- register_tool decorator + Tool interface used by the AgentRunner.
- AgentRunner: basic JSON-action loop pattern enabling model-driven tool invocation,
  RAG-like context injection (via a simple retriever function), and conversational memory.

Notes / design decisions:
- This is intentionally small and dependency-light: no direct LangChain dependency.
- The AgentRunner expects the model to respond with a JSON object in the top-level of the
  text response. The instruction template included below asks the model to emit:
    {"action":"tool","name":"<tool_name>","input":"..."} OR {"action":"final","output":"..."}
  This keeps the tool loop deterministic and easy to parse.
- For production-grade usage, replace the retriever and memory with durable stores (vector DB),
  add streaming support, backoff/retry, input sanitization, and stronger schema validation.
"""
from typing import Callable, Dict, Any, Optional, List
import os
import json
import logging
import time

import httpx

logger = logging.getLogger("aegis.langchain_adapter")

API_BASE = os.environ.get("AEGIS_API_BASE", "http://localhost:8081")
API_TOKEN = os.environ.get("AEGIS_API_TOKEN", "")
DEFAULT_MODEL = os.environ.get("AEGIS_DEFAULT_MODEL", "multimodal_demo")
DEFAULT_MODEL_VERSION = os.environ.get("AEGIS_DEFAULT_MODEL_VERSION", "v1")
DEFAULT_TIMEOUT = int(os.environ.get("AEGIS_PREDICT_TIMEOUT_SEC", "30"))


class AegisError(RuntimeError):
    pass


class AegisLLM:
    """
    Lightweight wrapper for Aegis predict endpoint.

    Usage:
      llm = AegisLLM(api_base=..., api_token=..., model="foo", version="v1")
      resp_text = llm.generate("summarize this...", tenant_id="tenant-123")
    """

    def __init__(self, api_base: str = API_BASE, api_token: str = API_TOKEN, model: str = DEFAULT_MODEL, version: str = DEFAULT_MODEL_VERSION, timeout: int = DEFAULT_TIMEOUT):
        self.api_base = api_base.rstrip("/")
        self.api_token = api_token
        self.model = model
        self.version = version
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)

    def _predict_url(self) -> str:
        # Path pattern matches earlier examples; adapt if your server uses a different route.
        return f"{self.api_base}/v1/models/{self.model}/versions/{self.version}/predict"

    def generate(self, prompt: str, tenant_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> str:
        """
        Send prompt to Aegis predict endpoint and return the textual output.

        extra: optional dict that will be merged into request body (e.g., {"max_tokens": 512})
        """
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id

        body = {"input": prompt}
        if extra:
api/langchain_adapter.py
