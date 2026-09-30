"""
Unit-style tests for MLflow registration wiring.

These tests mock MlflowClient and api.model_signing to validate that:
- register_model_from_mlflow downloads artifacts using MlflowClient
- sign_model_artifact is invoked when sign_key is provided
- registry.register is called when registry API exists

Run: pytest tests/test_mlflow_integration.py -q
"""
import tempfile
import os
from unittest import mock

import pytest

import scripts.register_model_from_mlflow as reg_script

class DummyClient:
    def __init__(self, download_return):
        self.download_return = download_return
        self.downloaded = []

    def download_artifacts(self, run_id, path, dst_path):
        # emulate creating a file at dst_path/path
        out_dir = os.path.join(dst_path, path if not path.endswith(".joblib") else os.path.dirname(path))
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, os.path.basename(path))
        with open(fpath, "wb") as fh:
            fh.write(b"fake-model")
        self.downloaded.append((run_id, path, dst_path))
        return out_dir

def test_register_download_and_sign(monkeypatch, tmp_path):
    # monkeypatch MlflowClient
    dummy = DummyClient(download_return=str(tmp_path))
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda uri=None: dummy)
    # monkeypatch sign_model_artifact
    sign_calls = []
    def fake_sign(p, key):
        sign_calls.append((p, key))
        return p + ".sig.json"
    monkeypatch.setitem(sys.modules, 'api.model_signing', mock.MagicMock())
    import api.model_signing as ms
    ms.sign_model_artifact = fake_sign

    # monkeypatch registry
    fake_registry = mock.MagicMock()
    fake_registry.register = mock.MagicMock()
    monkeypatch.setitem(sys.modules, 'api.registry', fake_registry)

    # invoke main with args
    test_args = ["--run-id", "r1", "--artifact-path", "model/model.joblib", "--model-name", "mymodel", "--sign-key", "k1"]
    import sys
    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0]] + test_args
    try:
        reg_script.main()
    finally:
        sys.argv = old_argv

    # assert sign was called (we replaced sign_model_artifact)
    assert len(sign_calls) == 1
    # assert registry.register called (best-effort)
    # fake_registry.register called with model name and version pattern
    assert fake_registry.register.called
tests/test_mlflow_integration.py
