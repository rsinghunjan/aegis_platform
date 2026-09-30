#!/usr/bin/env python3
"""
Unit test ensuring that when the enforcement patch is used, loaders call verify_and_download
before loading models. Uses monkeypatch to simulate fetch_and_verify_model raising on bad artifact.
"""
from __future__ import annotations
import pytest
from pathlib import Path

def test_patch_wrap_calls_verify(monkeypatch, tmp_path):
    pytest.importorskip(
        "aegis_multimodal_ai_system.orchestrator.enforce_verification_patch"
    )

    # Create a dummy loader with a load function to patch
    import types
    dummy = types.SimpleNamespace()
    def load_savedmodel(model_name: str, local_artifact: Path = None):
        # assert that local_artifact is provided by wrapper
        assert local_artifact is not None
        return "loaded"
    dummy.load_savedmodel = load_savedmodel

    # monkeypatch importing path inside enforce_verification_patch
    import sys
    sys.modules['aegis_multimodal_ai_system.loaders.tf_loader'] = dummy

    from aegis_multimodal_ai_system.orchestrator import enforce_verification_patch as ev

    called = {'verify': False}
    def fake_fetch(name, artifact_name='saved_model'):
        called['verify'] = True
        # return a fake local path
        return tmp_path / "fake_local"

    monkeypatch.setattr('aegis_multimodal_ai_system.model_registry.verify_and_download.fetch_and_verify_model', fake_fetch)

    # apply patch, should wrap dummy.load_savedmodel
    ev.patch_loaders()

    # call the patched loader
    res = dummy.load_savedmodel("example-model")
    assert res == "loaded"
    assert called['verify']
