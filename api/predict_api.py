"""
Predict router (Option A): scope-protected predict endpoints.

This is extracted into a router so api/api_server4.py can include it cleanly.
"""
import base64
import time
import uuid
import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends

from api.auth import require_scopes
from api.model_registry import registry  # or whichever module exposes the model registry in your repo
from api.schemas import PredictionRequest, PredictionResponse  # adjust import path to where these pydantic models live
from api.utils import decode_base64_file  # adjust to your actual helper location

logger = logging.getLogger("aegis.predict_api")

router = APIRouter(prefix="/v1/models", tags=["predict"])

predict_scope = require_scopes(["predict"])


def _make_request_id() -> str:
    return str(uuid.uuid4())


@router.post("/{model_name}/versions/{version}/predict", response_model=PredictionResponse)
async def predict(model_name: str, version: str, req: PredictionRequest, auth=Depends(predict_scope)):
    request_id = _make_request_id()
    start = time.time()

    try:
        model_obj = registry.load(model_name, version)
    except KeyError:
        raise HTTPException(status_code=404, detail="model or version not found")

    # basic input validation: require at least one modality
    if not (req.text or req.image_base64 or req.image_url or req.audio_base64):
        raise HTTPException(status_code=400, detail="No input provided; supply text, image or audio")

    # decode base64 examples (don't store in memory in prod for large files)
    if req.image_base64:
        try:
            _ = decode_base64_file(req.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid image_base64")

    if req.audio_base64:
        try:
            _ = decode_base64_file(req.audio_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid audio_base64")

    result = registry.predict(model_obj, req)
    elapsed_ms = (time.time() - start) * 1000.0
    resp = PredictionResponse(
        request_id=request_id,
        model=model_name,
        version=version,
        result=result,
        metrics={"inference_ms": elapsed_ms},
    )
    logger.info("predict %s %s request_id=%s elapsed=%.2fms", model_name, version, request_id, elapsed_ms)
    return resp


@router.post("/{model_name}/versions/{version}/predict-multipart", response_model=PredictionResponse)
async def predict_multipart(
    model_name: str,
    version: str,
    text: Optional[str] = None,
    image_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    auth=Depends(predict_scope),
):
    request_id = _make_request_id()
    start = time.time()

    try:
        model_obj = registry.load(model_name, version)
    except KeyError:
        raise HTTPException(status_code=404, detail="model or version not found")

    image_b64 = None
    audio_b64 = None
    if image_file:
        image_bytes = await image_file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    if audio_file:
        audio_bytes = await audio_file.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    req = PredictionRequest(text=text, image_base64=image_b64, audio_base64=audio_b64, parameters={})
    result = registry.predict(model_obj, req)
    elapsed_ms = (time.time() - start) * 1000.0
    return PredictionResponse(
        request_id=request_id,
        model=model_name,
        version=version,
        result=result,
        metrics={"inference_ms": elapsed_ms},
    )
