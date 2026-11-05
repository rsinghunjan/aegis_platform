"""Promote an MLflow run artifact into the Aegis model registry directory.

Layout: $AEGIS_MODEL_REGISTRY_DIR/<name>/<version>/{<artifact>, <artifact>.sig, model_signature.json}.
Signing is mandatory when sign_key is given; any signing failure aborts before registration.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _download_mlflow_artifact(run_id: str, artifact_path: str, dst_dir: str) -> str:
    from mlflow.tracking import MlflowClient

    return MlflowClient().download_artifacts(run_id, artifact_path, dst_dir)


def _find_artifact_file(downloaded: str, basename: str) -> Optional[str]:
    path = Path(downloaded)
    if path.is_file():
        return str(path)
    for candidate in path.rglob(basename):
        if candidate.is_file():
            return str(candidate)
    return None


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _register(model_name: str, version: str, artifact: str, signatures: list, extra: Dict[str, Any]) -> str:
    if not _SAFE.match(model_name) or not _SAFE.match(version):
        raise ValueError("model name and version must be simple identifiers")
    root = Path(os.getenv("AEGIS_MODEL_REGISTRY_DIR", "model_registry/models")).resolve()
    target = (root / model_name / version).resolve()
    if root not in target.parents:
        raise ValueError("invalid registry target")
    if target.exists():
        raise FileExistsError(f"{model_name}:{version} is already registered")
    target.mkdir(parents=True)
    shutil.copy2(artifact, target / Path(artifact).name)
    for sig in signatures:
        shutil.copy2(sig, target / Path(sig).name)
    metadata = {
        "sha256": _sha256(artifact),
        "artifact": Path(artifact).name,
        "signatures": [Path(s).name for s in signatures],
        **extra,
    }
    (target / "model_signature.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return str(target)


def promote_run_to_registry(
    run_id: str,
    artifact_path: str,
    model_name: str,
    version: Optional[str] = None,
    sign_key: Optional[str] = None,
    registry_module: Any = None,
) -> Dict[str, Any]:
    """Returns {run_id, artifact_local_path, signed_paths, registered, registry_path, model_name, version}."""
    tmpdir = tempfile.mkdtemp(prefix="aegis_mlflow_promote_")
    signed_paths: list = []
    downloaded = _download_mlflow_artifact(run_id, artifact_path, tmpdir)
    candidate = _find_artifact_file(downloaded, os.path.basename(artifact_path))
    if not candidate:
        raise FileNotFoundError(f"Could not locate artifact file in downloaded path: {downloaded}")
    if sign_key:
        from api.model_signing import sign_model_artifact

        signed_paths.append(sign_model_artifact(candidate, sign_key))
    ver = version or f"mlflow-{run_id}"
    registry_path = _register(model_name, ver, candidate, signed_paths, {"run_id": run_id})
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        tags = {
            "aegis.registry.registered": "True",
            "aegis.registry.model_name": model_name,
            "aegis.registry.model_version": ver,
        }
        if signed_paths:
            tags["aegis.artifact.signed"] = ",".join(Path(p).name for p in signed_paths)
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
    except Exception:
        logger.exception("Failed to annotate MLflow run with registry tags (non-fatal)")
    return {
        "run_id": run_id,
        "artifact_local_path": candidate,
        "signed_paths": signed_paths,
        "registered": True,
        "registry_path": registry_path,
        "model_name": model_name,
        "version": ver,
    }
