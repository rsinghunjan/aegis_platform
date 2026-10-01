import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .multimodal_ai_system import MultimodalAISystem
from .safety_checker import SafetyChecker

app = FastAPI(title="Aegis Multimodal AI System")
ai_system = MultimodalAISystem()
safety_checker = SafetyChecker()
logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    query: str = Field(min_length=1)


class GenerateResponse(BaseModel):
    response: str

@app.get("/")
def root():
    return {"status": "Aegis Multimodal AI System is running."}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be blank")
    if safety_checker.is_unsafe(request.query):
        raise HTTPException(status_code=422, detail="query blocked by safety checker")
    try:
        response = ai_system.generate_safe_response(request.query)
    except Exception as exc:
        logger.exception("Multimodal generation failed")
        raise HTTPException(status_code=500, detail="generation failed") from exc
    response = str(response)
    if safety_checker.is_unsafe(response):
        raise HTTPException(status_code=422, detail="response blocked by safety checker")
    return GenerateResponse(response=response)