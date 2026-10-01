"""
Hosted inference proxy runtime for Aegis (Hugging Face Inference Endpoints + generic HTTP model endpoints).

Usage:
 - Construct HfProxyRuntime(model_endpoint="google/flan-t5-small" or full URL)
 - call run(payload, tenant_id=..., headers=...)

Config (env):
 - HF_API_TOKEN (recommended to be injected from Vault; do NOT store in repo)
 - HF_PROXY_TIMEOUT_SEC (default 60)
 - HF_PROXY_MAX_RETRIES (default 2)

Notes:
 - This is intentionally generic: model_endpoint may be a full URL (https://api-inference.huggingface.co/models/owner/model)
   or a short model id (the runtime will prepend HF inference base URL if model_endpoint has no scheme).
 - Billing/usage is recorded via billing_meter() hook; implement collector to persist/invoice.
"""
from typing import Any, Dict, Optional
import os
import time
import logging

import httpx

from api.billing_metering import billing_meter_hf_call

logger = logging.getLogger("aegis.hf_proxy_runtime")

HF_API_INFERENCE_BASE = os.environ.get("HF_API_INFERENCE_BASE", "https://api-inference.huggingface.co/models")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_PROXY_TIMEOUT = int(os.environ.get("HF_PROXY_TIMEOUT_SEC", "60"))
HF_PROXY_MAX_RETRIES = int(os.environ.get("HF_PROXY_MAX_RETRIES", "2"))
# optional per-tenant mapping env var: JSON map of tenant -> hf_token (if you want per-tenant HF accounts)
# not implemented in this simple runtime; prefer Vault per-tenant secrets

class HfProxyError(RuntimeError):
    pass


class HfProxyRuntime:
    def __init__(self, model_endpoint: str, name: Optional[str] = None, timeout: Optional[int] = None, max_retries: Optional[int] = None):
        """
        model_endpoint: either a full URL or model id (owner/model) that the runtime will call.
        """
        self.model_endpoint = model_endpoint
        self.name = name or model_endpoint
        self.timeout = timeout or HF_PROXY_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else HF_PROXY_MAX_RETRIES
        self._client = httpx.Client(timeout=self.timeout)

    def _resolve_url(self) -> str:
        if self.model_endpoint.startswith("http://") or self.model_endpoint.startswith("https://"):
            return self.model_endpoint
        return f"{HF_API_INFERENCE_BASE.rstrip('/')}/{self.model_endpoint.lstrip('/')}"

    def _auth_headers(self, tenant_id: Optional[str] = None) -> Dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        token = os.environ.get("HF_API_TOKEN", HF_API_TOKEN)
        if not token:
            # fail fast; recommend injecting token via Vault
            logger.error("HF_API_TOKEN not configured in environment")
            raise HfProxyError("HF API token not configured")
        headers["Authorization"] = f"Bearer {token}"
        if tenant_id:
            headers["X-Tenant-ID"] = tenant_id
        return headers

    def run(self, payload: Dict[str, Any], tenant_id: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Forward the payload to the remote inference endpoint and return parsed JSON response.
        Raises HfProxyError on repeated failure.

        Also calls billing_meter_hf_call() after success (non-blocking).
        """
        url = self._resolve_url()
        headers = self._auth_headers(tenant_id=tenant_id)
        if extra_headers:
            headers.update(extra_headers)

        start = time.time()
        last_exc = None
        attempt = 0
        while attempt <= self.max_retries:
            attempt += 1
            try:
                logger.debug("HF proxy request -> %s (attempt %d)", url, attempt)
                resp = self._client.post(url, json=payload, headers=headers, timeout=self.timeout)
                # HF returns 200 for success, 403 or 429 for rate-limit/auth errors, 503 for overload
                if resp.status_code == 200:
                    elapsed = time.time() - start
                    # attempt to parse JSON
                    try:
                        data = resp.json()
                    except Exception:
                        data = {"raw": resp.text}
                    # record billing/usage in background (non-blocking ideally)
                    try:
                        billing_meter_hf_call(model=self.model_endpoint, tenant_id=tenant_id, duration_s=elapsed, request_bytes=len(resp.request.content or b""), response_bytes=len(resp.content or b""))
                    except Exception:
                        logger.exception("billing meter failed (non-fatal)")
                    return {"status": 200, "data": data}
                else:
                    # non-200: for some codes we may retry
                    txt = resp.text
                    logger.warning("HF proxy non-200 status %s: %s", resp.status_code, txt[:500])
                    if resp.status_code in (429, 503, 502) and attempt <= self.max_retries:
                        backoff = 0.5 * (2 ** (attempt - 1))
                        time.sleep(backoff)
                        continue
                    raise HfProxyError(f"remote status {resp.status_code}: {txt}")
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning("HF proxy HTTPError attempt %d: %s", attempt, str(exc))
                if attempt <= self.max_retries:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                    continue
                raise HfProxyError(str(exc)) from exc
        # if we reach here, we failed
        raise HfProxyError(f"proxy failed after {self.max_retries+1} attempts: {last_exc}")
