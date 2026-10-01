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
#!/usr/bin/env python3
"""
MLflow helper: fetches model artifacts using model_registry.loader.download_artifact
so MLflow-related code does not need to call cloud SDKs directly.

Functions:
- download_artifact_from_mlflow(run_id, artifact_path=None) -> Path
  attempts to download artifacts from MLflow; if the artifact metadata contains
  an external URI, it delegates to model_registry.loader.download_artifact.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:
    mlflow = None
    MlflowClient = None

from aegis_multimodal_ai_system.model_registry import loader as registry_loader

def download_artifact_from_mlflow(run_id: str, artifact_path: Optional[str] = None, dest_dir: Optional[Path] = None) -> Path:
    """
    Download artifact from MLflow run. If MLflow is not available or the artifact
    is an externally-hosted URI referenced in metadata, fall back to registry_loader.download_artifact.

    Returns Path to local artifact.
    """
    dest_dir = Path(dest_dir) if dest_dir else None

    # Try MLflow direct download first (fast path)
    if mlflow is not None and MlflowClient is not None:
        client = MlflowClient()
        try:
            # If artifact_path is provided, try to download that artifact.
            # If not provided, download all artifacts and try to find a model file.
            if artifact_path:
                local = client.download_artifacts(run_id, artifact_path, dst_path=str(dest_dir or Path(".mlflow_tmp") / run_id))
                return Path(local)
            else:
                # download root artifacts
                local_root = client.download_artifacts(run_id, ".", dst_path=str(dest_dir or Path(".mlflow_tmp") / run_id))
                # heuristics: prefer typical model filenames
                for cand in ("model.onnx", "model.pt", "model.pth", "model.zip"):
                    path = Path(local_root) / cand
                    if path.exists():
                        return path
                # fallback: return first file found
                for p in Path(local_root).rglob("*"):
                    if p.is_file():
                        return p
        except Exception:
            # fall through to registry loader fallback
            pass

    # Fallback: check run metadata or model registry metadata (if structured)
    # If caller supplied a registry-style URI in artifact_path, delegate:
    if artifact_path and (artifact_path.startswith("s3://") or artifact_path.startswith("gs://") or artifact_path.startswith("azure://") or artifact_path.startswith("file://") or artifact_path.startswith("model_registry/")):
        return registry_loader.download_artifact(artifact_path, dest_dir=dest_dir)

    # Last resort: if run has an artifact URI recorded in metadata, try to parse it
    try:
        client = MlflowClient() if MlflowClient else None
        if client:
            run = client.get_run(run_id)
            data = run.data.tags
            # convention: some pipelines store artifact_uri as tag
            artifact_uri = data.get("artifact_uri") or data.get("model_artifact_uri")
            if artifact_uri:
                # artifact_uri may be an http(s) mlflow server path or s3://...
                return registry_loader.download_artifact(artifact_uri, dest_dir=dest_dir)
    except Exception:
        pass

    raise RuntimeError("Unable to download artifact from MLflow or registry for run_id=" + run_id)
