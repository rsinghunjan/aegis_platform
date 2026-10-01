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
#!/usr/bin/env python3
"""
TPU model loader — updated to use model_registry.loader.download_artifact
rather than calling cloud SDKs directly.

This module focuses on resolving a model artifact URI/path to a local file
and returns the local path for downstream TPU loading logic.

Behavior:
- Accepts model_uri (s3://, gs://, azure://, file://, or relative path under model_registry/)
- Uses model_registry.loader.download_artifact to fetch remote artifacts using the StorageClient abstraction
- Returns a Path to the local artifact file (caller is responsible for loading/parsing)
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from aegis_multimodal_ai_system.model_registry import loader as registry_loader

def resolve_and_fetch_model(model_uri_or_path: str, dest_dir: Optional[Path] = None) -> Path:
    """
    Resolve an artifact URI or a local path and return a local file path.
    - If model_uri_or_path is a local path (exists or file://), it will be copied/returned.
    - If it's a remote URI (s3://, gs://, azure://), the artifact is downloaded via the storage adapters.
    """
    # registry_loader.download_artifact handles local and remote URIs and returns a Path
    local = registry_loader.download_artifact(model_uri_or_path, dest_dir=dest_dir)
    return local

def load_for_tpu(model_uri_or_path: str, dest_dir: Optional[Path] = None):
    """
    High-level entry point: fetch artifact and (optionally) perform TPU-specific preparation.
    This function intentionally returns the local artifact path. TPU-specific loading
    (e.g., loading a compiled TPU artifact or sending file to TPU filesystem) should be
    implemented by the caller or a small wrapper that knows your TPU runtime.
    """
    local_model = resolve_and_fetch_model(model_uri_or_path, dest_dir=dest_dir)
    # Example placeholder: if you need to convert to TPU-friendly format, do it here.
    # For safety, we do not attempt to load with torch/xla or TF/XLA here to avoid adding cloud SDKs.
    return local_model
aegis_multimodal_ai_system/model_loader_tpu.py
