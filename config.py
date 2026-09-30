#!/usr/bin/env python3
"""
Aegis config adapter: unify env + optional YAML config.

Usage:
  from aegis_multimodal_ai_system.config import cfg
  print(cfg.OBJECT_STORE_TYPE, cfg.OBJECT_STORE_ENDPOINT)
"""
from __future__ import annotations
import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

_CONFIG_PATHS = [
    Path(os.environ.get("AEGIS_CONFIG", "")) if os.environ.get("AEGIS_CONFIG") else None,
    Path("config.yaml"),
    Path("config.yml"),
]
# filter None
_CONFIG_PATHS = [p for p in _CONFIG_PATHS if p is not None]

def _load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

@dataclass
class Config:
    # object store
    OBJECT_STORE_TYPE: str = os.environ.get("OBJECT_STORE_TYPE", "s3")  # 's3' or 'minio' or 'gcs' etc.
    OBJECT_STORE_ENDPOINT: Optional[str] = os.environ.get("OBJECT_STORE_ENDPOINT")
    OBJECT_STORE_REGION: Optional[str] = os.environ.get("OBJECT_STORE_REGION", "us-east-1")
    OBJECT_STORE_BUCKET: Optional[str] = os.environ.get("OBJECT_STORE_BUCKET", "aegis-models")
    OBJECT_STORE_ACCESS_KEY: Optional[str] = os.environ.get("OBJECT_STORE_ACCESS_KEY")
    OBJECT_STORE_SECRET_KEY: Optional[str] = os.environ.get("OBJECT_STORE_SECRET_KEY")
    # database
    DATABASE_URL: Optional[str] = os.environ.get("DATABASE_URL", "sqlite:///./aegis.db")
    # MLflow (if used)
    MLFLOW_TRACKING_URI: Optional[str] = os.environ.get("MLFLOW_TRACKING_URI")
    # general
    ENV: str = os.environ.get("ENV", "development")
    # raw loaded yaml (if any)
    _raw: dict = None

def _merge_from_yaml(cfg: Config) -> Config:
    for p in _CONFIG_PATHS:
        if p.exists():
            raw = _load_yaml(p)
            # map known keys
            for k, v in raw.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            cfg._raw = raw
            break
    return cfg

# Single shared config object
cfg = _merge_from_yaml(Config())
