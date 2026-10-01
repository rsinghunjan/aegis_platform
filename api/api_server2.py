# (updated) Reference FastAPI model-serving server for Aegis with auth integration
# - uses api.auth.require_scopes to protect prediction endpoints

import base64
import time
import uuid
import logging
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# import auth router and dependency generator
from api import auth

logger = logging.getLogger("aegis_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Aegis Model Serving", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount auth router
app.include_router(auth.router)


# --- Pydantic models for request/response -------------------------------------------------

class PredictionRequest(BaseModel):
    text: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    audio_base64: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    request_id: str
    model: str
    version: str
    result: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None


class ModelSummary(BaseModel):
    name: str
    latest_version: str


class ModelMetadata(BaseModel):
    name: str
    version: str
    description: Optional[str] = None
    inputs: Optional[List[Dict[str, Any]]] = None
    outputs: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[str] = None


# --- Simple ModelRegistry (placeholder) ---------------------------------------------------

class ModelRegistry:
    def __init__(self):
        self._registry = {
            "example_text_model": {
                "v1": {"metadata": {"name": "example_text_model", "version": "v1", "description": "Demo text model"}},
            },
            "multimodal_demo": {
                "v1": {"metadata": {"name": "multimodal_demo", "version": "v1", "description": "Demo multimodal model"}},
            },
        }
        self._loaded = {}

    def list_models(self):
        return [ModelSummary(name=k, latest_version=max(v.keys())) for k, v in self._registry.items()]

    def list_versions(self, model_name: str):
        if model_name not in self._registry:
            raise KeyError("model not found")
        return list(self._registry[model_name].keys())

    def get_metadata(self, model_name: str, version: str):
        if model_name not in self._registry or version not in self._registry[model_name]:
            raise KeyError("model or version not found")
        return ModelMetadata(**self._registry[model_name][version]["metadata"])

    def load(self, model_name: str, version: str):
        key = (model_name, version)
        if key in self._loaded:
            return self._loaded[key]
        if model_name not in self._registry or version not in self._registry[model_name]:
            raise KeyError("model or version not found")
        model_obj = {"name": model_name, "version": version, "type": "stub"}
        self._loaded[key] = model_obj
        logger.info("Loaded model %s:%s", model_name, version)
        return model_obj

    def predict(self, model_obj, request: PredictionRequest) -> Dict[str, Any]:
        time.sleep(0.01)
        out = {
            "echo_text": request.text,
            "has_image": bool(request.image_base64 or request.image_url),
            "has_audio": bool(request.audio_base64),
            "labels": [{"label": "demo_label", "score": 0.87}]
        }
        return out


registry = ModelRegistry()


# --- Utility functions --------------------------------------------------------------------

def make_request_id() -> str:
    return str(uuid.uuid4())


def decode_base64_file(b64: str) -> bytes:
    return base64.b64decode(b64)


# --- Health & readiness endpoints ---------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    loaded = len(registry._loaded)
    return {"ready": True, "loaded_models": loaded}


# --- Model registry endpoints ------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    return {"models": [m.dict() for m in registry.list_models()]}


@app.get("/v1/models/{model_name}/versions")
async def list_versions(model_name: str):
    try:
        versions = registry.list_versions(model_name)
        return {"versions": versions}
    except KeyError:
        raise HTTPException(status_code=404, detail="model not found")


@app.get("/v1/models/{model_name}/versions/{version}/metadata")
async def model_metadata(model_name: str, version: str):
    try:
        meta = registry.get_metadata(model_name, version)
        return meta.dict()
    except KeyError:
        raise HTTPException(status_code=404, detail="model or version not found")


# --- Prediction endpoint (JSON body) protected by scopes --------------------------------

# Enforce the 'predict' scope for synchronous prediction
predict_scope_dependency = auth.require_scopes(["predict"])


@app.post("/v1/models/{model_name}/versions/{version}/predict", response_model=PredictionResponse)
async def predict_json(model_name: str, version: str, req: PredictionRequest,
                       current_user = Depends(predict_scope_dependency)):
    request_id = make_request_id()
    start = time.time()
    try:
        model_obj = registry.load(model_name, version)
    except KeyError:
        raise HTTPException(status_code=404, detail="model or version not found")

    if not (req.text or req.image_base64 or req.image_url or req.audio_base64):
        raise HTTPException(status_code=400, detail="No input provided; supply text, image or audio")

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
        metrics={"inference_ms": elapsed_ms}
    )
    logger.info("predict %s %s request_id=%s user=%s elapsed=%.2fms", model_name, version, request_id, current_user.username, elapsed_ms)
    return resp


@app.post("/v1/models/{model_name}/versions/{version}/predict-multipart", response_model=PredictionResponse)
async def predict_multipart(model_name: str, version: str, text: Optional[str] = None,
                            image_file: Optional[UploadFile] = File(None),
                            audio_file: Optional[UploadFile] = File(None),
                            current_user = Depends(predict_scope_dependency)):
    request_id = make_request_id()
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
    logger.info("predict-multipart %s %s request_id=%s user=%s elapsed=%.2fms", model_name, version, request_id, current_user.username, elapsed_ms)
    return PredictionResponse(
        request_id=request_id, model=model_name, version=version, result=result,
        metrics={"inference_ms": elapsed_ms}
    )


# --- Basic error handler -----------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"code": exc.status_code, "message": exc.detail})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.api_server:app", host="0.0.0.0", port=8080, log_level="info")
