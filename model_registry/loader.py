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
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
#!/usr/bin/env python3
"""
Model artifact loader that uses StorageClient abstraction.

Functions:
- resolve_artifact_uri(uri) -> (scheme, bucket/container, key)
- download_artifact(uri, dest_dir) -> Path to local artifact file
- load_model_from_registry(model_dir, artifact_name="model.onnx") -> local path

Supports:
 - s3://bucket/key
 - gs://bucket/key
 - azure://container/key
 - file:///absolute/path or relative paths under model_registry/
 - local model_registry/<model>/artifact
"""
from __future__ import annotations
import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, Optional

from aegis_multimodal_ai_system.storage.factory import create_storage_client
from aegis_multimodal_ai_system import config

def resolve_artifact_uri(uri: str) -> Tuple[str, str, str]:
    """
    Returns (scheme, bucket_or_container, key)
    """
    if uri.startswith("s3://"):
        rest = uri[len("s3://"):]
        parts = rest.split("/", 1)
        return "s3", parts[0], parts[1] if len(parts) > 1 else ""
    if uri.startswith("gs://"):
        rest = uri[len("gs://"):]
        parts = rest.split("/", 1)
        return "gs", parts[0], parts[1] if len(parts) > 1 else ""
    if uri.startswith("azure://") or uri.startswith("az://"):
        rest = uri.split("://",1)[1]
        parts = rest.split("/", 1)
        return "azure", parts[0], parts[1] if len(parts) > 1 else ""
    if uri.startswith("file://"):
        path = Path(uri[len("file://"):])
        return "file", "", str(path)
    # fallback: treat as relative path under model_registry
    if os.path.exists(uri):
        return "file", "", uri
    # try model_registry relative path
    if uri.startswith("model_registry/") or uri.startswith("./model_registry/"):
        p = Path(uri)
        return "file", "", str(p)
    # Unknown; treat as s3-like if pattern contains :
    raise ValueError(f"Unrecognized artifact URI: {uri}")

def download_artifact(uri: str, dest_dir: Optional[Path] = None) -> Path:
    scheme, container, key = resolve_artifact_uri(uri)
    dest_dir = Path(dest_dir or tempfile.mkdtemp(prefix="aegis-art-"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    if scheme in ("s3", "gs", "azure"):
        # create storage client of appropriate type by overriding cfg temporarily if needed
        # choose explicit adapter by scheme if cfg default doesn't match
        orig_type = config.cfg.OBJECT_STORE_TYPE
        try:
            # if scheme doesn't match default, temporarily set it
            if scheme == "gs":
                config.cfg.OBJECT_STORE_TYPE = "gcs"
            elif scheme == "azure":
                config.cfg.OBJECT_STORE_TYPE = "azure"
            elif scheme == "s3":
                config.cfg.OBJECT_STORE_TYPE = "s3"
            client = create_storage_client(bucket=container)
            local_path = dest_dir / Path(key).name
            client.download(key, local_path)
            return local_path
        finally:
            config.cfg.OBJECT_STORE_TYPE = orig_type
    elif scheme == "file":
        src = Path(key)
        if not src.exists():
            raise FileNotFoundError(f"Local artifact not found: {src}")
        dest = dest_dir / src.name
        shutil.copy2(str(src), str(dest))
        return dest
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

def load_model_from_registry(model_name: str, artifact_name: str = "model.onnx") -> Path:
    """
    Convenience: attempts to find artifact under model_registry/<model_name>/<artifact_name>
    or uses metadata.yaml -> artifact.uri if present.
    """
    base = Path.cwd() / "model_registry" / model_name
    # 1) explicit file
    cand = base / artifact_name
    if cand.exists():
        return cand
    # 2) metadata.yaml
    meta_yaml = base / "metadata.yaml"
    if meta_yaml.exists():
        try:
            import yaml
            meta = yaml.safe_load(meta_yaml.read_text(encoding="utf-8"))
            art = meta.get("artifact", {})
            if isinstance(art, dict):
                uri = art.get("uri") or art.get("path")
            else:
                uri = art
            if uri:
                return download_artifact(uri)
        except Exception:
            pass
    raise FileNotFoundError(f"Artifact {artifact_name} not found for model {model_name}")
