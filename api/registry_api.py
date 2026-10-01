"""
FastAPI router for model registry lifecycle management.

Endpoints (require admin scope):
- POST /v1/registry/register
- GET  /v1/registry/{model_name}/versions
- POST /v1/registry/{model_name}/{version}/promote
- GET  /v1/registry/{model_name}/{version}/validate
"""

import os
import tempfile
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException

from api import auth
from api.model_registry import (
    register_model_artifact,
    get_model_versions,
    promote_model_version,
    validate_model_signature,
)
from api.db import SessionLocal

logger = logging.getLogger("aegis_registry_api")
router = APIRouter(prefix="/v1/registry", tags=["registry"])

admin_scope = auth.require_scopes(["admin"])


@router.post("/register")
async def register(
    model_name: str = Form(...),
    version: str = Form(...),
    stage: str = Form("staging"),
    description: Optional[str] = Form(None),
    artifact_file: Optional[UploadFile] = File(None),
    artifact_path: Optional[str] = Form(None),
    current_user=Depends(admin_scope),
):
    """
    Register a model version.
    - Accepts multipart upload (artifact_file) or a pre-uploaded artifact_path (S3 URL / path)
    - Stores checksum/signature in metadata._registry and returns metadata.
    """
    if stage not in ("staging", "canary", "prod"):
        raise HTTPException(status_code=400, detail="invalid stage")
    # If file upload provided, write to a temp path and register
    if artifact_file:
        tmpdir = tempfile.mkdtemp()
        target = os.path.join(tmpdir, artifact_file.filename)
        with open(target, "wb") as f:
            content = await artifact_file.read()
            f.write(content)
        artifact_path = target

    if not artifact_path:
        raise HTTPException(status_code=400, detail="artifact_file or artifact_path required")

    metadata = {"description": description}
    try:
        out = register_model_artifact(model_name=model_name, version=version, artifact_path=artifact_path, metadata=metadata, stage=stage)
        return out
    except Exception as e:
        logger.exception("register error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_name}/versions")
async def list_versions(model_name: str, current_user=Depends(admin_scope)):
    try:
        return get_model_versions(model_name)
    except KeyError:
        raise HTTPException(status_code=404, detail="model not found")


@router.post("/{model_name}/{version}/promote")
async def promote(model_name: str, version: str, target_stage: str = Form(...), current_user=Depends(admin_scope)):
    try:
        return promote_model_version(model_name, version, target_stage, actor=getattr(current_user, "username", "system"))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{model_name}/{version}/validate")
async def validate(model_name: str, version: str, current_user=Depends(admin_scope)):
    try:
        ok = validate_model_signature(model_name, version)
        return {"model": model_name, "version": version, "signature_valid": ok}
    except KeyError:
        raise HTTPException(status_code=404, detail="model or version not found")
