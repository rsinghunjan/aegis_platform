#!/usr/bin/env python3
"""
Unit test to ensure enforcement patch calls fetch_and_verify_model before loader runs.
"""
from __future__ import annotations
import pytest
from pathlib import Path

def test_enforcement_patch_calls_verify(monkeypatch, tmp_path):
    pytest.importorskip(
        "aegis_multimodal_ai_system.orchestrator.enforce_verification_patch"
    )

    import types, sys
    dummy = types.SimpleNamespace()
    def load_savedmodel(model_name: str, local_artifact: Path = None):
        assert local_artifact is not None
        return "loaded"
    dummy.load_savedmodel = load_savedmodel
    loaders = types.ModuleType('aegis_multimodal_ai_system.loaders')
    loaders.tf_loader = dummy
    sys.modules['aegis_multimodal_ai_system.loaders'] = loaders
    sys.modules['aegis_multimodal_ai_system.loaders.tf_loader'] = dummy

    called = {'verified': False}
    def fake_fetch(name, artifact_name='saved_model'):
        called['verified'] = True
        return tmp_path / "fake_local"

    monkeypatch.setattr('model_registry.verify_and_download.fetch_and_verify_model', fake_fetch)

    import orchestrator_enforce_verification_patch as ev
    ev.patch_loaders()

    res = dummy.load_savedmodel("example-model")
    assert res == "loaded"
