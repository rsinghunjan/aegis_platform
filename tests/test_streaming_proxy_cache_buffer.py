"""
Tests for ops/llm_support/gateway/streaming_proxy_cached.py in-memory
buffer capping during response streaming.

Focus: once the accumulated (would-be-cached) response text exceeds
CACHE_MAX_BUFFER_CHARS, the handler must stop accumulating it for caching
purposes (bounding per-request memory growth) while still streaming every
chunk through to the client unaffected.

Note: this module imports `ops.llm_support.cache.cache.LocalCache`, which
does not exist in this checkout (pre-existing, unrelated to this change).
We install a lightweight stand-in module in sys.modules before import so the
buffering behavior we're actually testing can be exercised in isolation.

Run: pytest tests/test_streaming_proxy_cache_buffer.py -q
"""
import asyncio
import os
import sys
import types

os.environ.setdefault("METRICS_PORT", "0")  # avoid binding a real port twice


def _install_stub_local_cache():
    if "ops.llm_support.cache.cache" in sys.modules:
        return
    mod = types.ModuleType("ops.llm_support.cache.cache")

    class LocalCache:
        def __init__(self, maxsize=1024, ttl=3600):
            self._store = {}

        def get(self, key):
            return self._store.get(key)

        def set(self, key, value):
            self._store[key] = value

        def __len__(self):
            return len(self._store)

    mod.LocalCache = LocalCache
    sys.modules["ops.llm_support.cache.cache"] = mod


_install_stub_local_cache()

import ops.llm_support.gateway.streaming_proxy_cached as proxy  # noqa: E402


class _FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self):
        return self._body


async def _run_generate(body: dict):
    resp = await proxy.generate(_FakeRequest(body))
    streamed = []
    async for piece in resp.body_iterator:
        streamed.append(piece)
    return streamed


def test_buffer_capped_but_streaming_unaffected(monkeypatch):
    monkeypatch.setattr(proxy, "CACHE_MAX_BUFFER_CHARS", 10)

    long_text_chunks = ["hello ", "world ", "this ", "is ", "a ", "long ", "response "]

    async def fake_backend_stream(payload):
        for c in long_text_chunks:
            yield {"text": c, "score": 0.0}

    monkeypatch.setattr(proxy, "backend_stream", fake_backend_stream)
    monkeypatch.setattr(proxy, "cacheable_request", lambda body: True)

    set_calls = []
    monkeypatch.setattr(proxy._local_cache, "set", lambda key, value: set_calls.append(value))

    body = {
        "client_id": "c1",
        "model": "m",
        "model_version": "v1",
        "system_prompt": "",
        "payload": {"inputs": "prompt", "params": {}},
    }

    streamed = asyncio.run(_run_generate(body))

    # Every chunk was still streamed to the client, unaffected by the cap.
    assert len(streamed) == len(long_text_chunks)
    full_streamed_text = "".join(long_text_chunks)
    assert len(full_streamed_text) > 10  # confirms the cap was actually exceeded

    # But since the accumulated text exceeded CACHE_MAX_BUFFER_CHARS, caching
    # was skipped entirely (no local-cache .set call).
    assert set_calls == []


def test_short_response_is_still_cached(monkeypatch):
    monkeypatch.setattr(proxy, "CACHE_MAX_BUFFER_CHARS", 1000)

    short_chunks = ["hi", " there"]

    async def fake_backend_stream(payload):
        for c in short_chunks:
            yield {"text": c, "score": 0.0}

    monkeypatch.setattr(proxy, "backend_stream", fake_backend_stream)
    monkeypatch.setattr(proxy, "cacheable_request", lambda body: True)

    set_calls = []
    monkeypatch.setattr(proxy._local_cache, "set", lambda key, value: set_calls.append(value))

    body = {
        "client_id": "c1",
        "model": "m",
        "model_version": "v1",
        "system_prompt": "",
        "payload": {"inputs": "prompt", "params": {}},
    }

    asyncio.run(_run_generate(body))
    assert len(set_calls) == 1
    assert set_calls[0]["response"] == "hi there"
