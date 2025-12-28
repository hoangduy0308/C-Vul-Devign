"""FastAPI app for vulnerability detection."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from devign_pipeline.api.inference import (
    get_model_wrapper,
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title="Devign SliceAttBiGRU Vulnerability Scanner",
    description="API for detecting vulnerabilities in C code using BiGRU with Slice Attention",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "SliceAttBiGRU"}


@app.get("/info")
def info() -> dict:
    wrapper = get_model_wrapper()
    return {
        "model": "SliceAttBiGRU",
        "threshold": wrapper.threshold,
        "device": str(wrapper.model.embedding.weight.device),
        "metrics": wrapper.config,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")
    
    wrapper = get_model_wrapper()
    return wrapper.predict(req.code)


@app.post("/batch_predict")
def batch_predict(requests: list[PredictionRequest]) -> list[PredictionResponse]:
    if not requests:
        raise HTTPException(status_code=400, detail="Request list cannot be empty")
    
    wrapper = get_model_wrapper()
    return [wrapper.predict(req.code) for req in requests]
