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
#!/usr/bin/env python3
"""
Orchestrator helper: fetch+verify model before load, fail-closed if verification fails.
Usage: call fetch_and_verify_model(model_name) and only load models that return a local path.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging

from aegis_multimodal_ai_system.model_registry.verify_and_download import fetch_and_verify_model

LOG = logging.getLogger("aegis.orchestrator.verify_loader")

def fetch_and_prepare_model_for_load(model_name: str, artifact_name: str = "model.onnx", dest_dir: Optional[Path] = None) -> Path:
    """
    Download and verify the model artifact. Raises RuntimeError if verification fails.
    Callers MUST catch exceptions and treat them as "do not load".
    """
    LOG.info("Fetching and verifying model %s/%s", model_name, artifact_name)
    local = fetch_and_verify_model(model_name, artifact_name=artifact_name, dest_dir=dest_dir)
    LOG.info("Model %s verified and available at %s", model_name, local)
    return Path(local)
aegis_multimodal_ai_system/orchestrator_verify_loader.py
