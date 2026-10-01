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
#!/usr/bin/env python3
"""
Runtime enforcement patch (incremental safety layer).

Import early in your app (e.g., orchestrator entrypoint) to wrap common loader entrypoints
so fetch_and_verify_model() is called before an artifact is loaded. This is temporary
while callers are migrated to explicit fetch_and_prepare_model_for_load() calls.
"""
from __future__ import annotations
import logging
from functools import wraps
from typing import Callable, Any

from aegis_multimodal_ai_system.model_registry.verify_and_download import fetch_and_verify_model

LOG = logging.getLogger("aegis.enforce_verify")

def _wrap_loader(fn: Callable[..., Any], model_name_arg_index: int = 0, artifact_name: str = "model.onnx"):
    @wraps(fn)
    def _wrapped(*args, **kwargs):
        try:
            if len(args) > model_name_arg_index:
                model_name = args[model_name_arg_index]
            else:
                model_name = kwargs.get("model_name")
            if model_name:
                LOG.debug("Enforcing verification for model %s", model_name)
                local_path = fetch_and_verify_model(str(model_name), artifact_name=artifact_name)
                kwargs.setdefault("local_artifact", local_path)
        except Exception as e:
            LOG.exception("Model verification failed; refusing to load: %s", e)
            raise
        return fn(*args, **kwargs)
    return _wrapped

def patch_loaders():
    """
    Patch known loader functions. Add additional loader modules as you migrate callsites.
    """
    try:
        import aegis_multimodal_ai_system.loaders.tf_loader as tf_loader  # may not exist
        if hasattr(tf_loader, "load_savedmodel"):
            tf_loader.load_savedmodel = _wrap_loader(tf_loader.load_savedmodel, model_name_arg_index=0, artifact_name="saved_model")
            LOG.info("Patched tf_loader.load_savedmodel for verification enforcement")
    except Exception:
        LOG.debug("tf_loader not available to patch")

    try:
        import aegis_multimodal_ai_system.loaders.onnx_loader as onnx_loader  # may not exist
        if hasattr(onnx_loader, "load_onnx_model"):
            onnx_loader.load_onnx_model = _wrap_loader(onnx_loader.load_onnx_model, model_name_arg_index=0, artifact_name="model.onnx")
            LOG.info("Patched onnx_loader.load_onnx_model for verification enforcement")
    except Exception:
        LOG.debug("onnx_loader not available to patch")
aegis_multimodal_ai_system/orchestrator/enforce_verification_patch.py
