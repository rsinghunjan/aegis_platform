"""
Unit tests for HfProxyRuntime. Mocks httpx.Client.post to avoid network calls.
Run: pytest tests/test_hf_proxy_runtime.py -q
"""
import json
from unittest import mock
import pytest

from api.hf_proxy_runtime import HfProxyRuntime, HfProxyError

class DummyResp:
    def __init__(self, status_code=200, json_obj=None, text="ok"):
        self.status_code = status_code
        self._json = json_obj or {"generated": "hello"}
        self.text = text.encode() if isinstance(text, str) else text
        self.content = self.text
        class Req: content = b"{}"
        self.request = Req()
    def json(self):
        return self._json

@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("HF_API_TOKEN", "fake-token")

def test_hf_proxy_success(monkeypatch):
    runtime = HfProxyRuntime("owner/model")
    dummy = DummyResp(status_code=200, json_obj={"out":"ok"})
    m = mock.MagicMock()
    m.post.return_value = dummy
    runtime._client = m
    res = runtime.run({"inputs":"hi"}, tenant_id="t1")
    assert res["status"] == 200
    assert res["data"] == {"out":"ok"}

def test_hf_proxy_retry_then_fail(monkeypatch):
    runtime = HfProxyRuntime("owner/model", timeout=1, max_retries=1)
    m = mock.MagicMock()
    # first call raises HTTPError, second returns 503 then final raises
    from httpx import HTTPError
    m.post.side_effect = [HTTPError("conn"), DummyResp(status_code=503, json_obj={"err":"slow"}, text="slow")]
    runtime._client = m
    with pytest.raises(HfProxyError):
        runtime.run({"inputs":"x"}, tenant_id="t2")
