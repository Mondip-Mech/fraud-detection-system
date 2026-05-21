"""
Fraud Detection API — FastAPI service.

Endpoints:
- GET  /health    — readiness check
- GET  /metrics   — service-level metrics
- POST /predict   — score a single transaction
- GET  /docs      — interactive OpenAPI documentation
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.inference import MODEL_VERSION, FraudDetector
from src.api.schemas import (
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    TransactionRequest,
)

# ---- Service state ----

_state = {
    "detector": None,
    "started_at": None,
    "requests_total": 0,
    "predictions_total": 0,
    "flags_total": 0,
    "latencies_ms": [],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup; clean up at shutdown."""
    print("Loading model artifacts...")
    _state["detector"] = FraudDetector()
    _state["started_at"] = time.time()
    print(f"Service ready. Model version: {MODEL_VERSION}")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit card fraud detection powered by LightGBM. "
        "Returns fraud probability, decision (allow/flag), and top SHAP feature contributors per prediction. "
        "Trained on the IEEE-CIS Fraud Detection dataset (590K transactions, PR-AUC 0.473)."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Service health check."""
    detector = _state["detector"]
    if detector is None:
        return HealthResponse(
            status="down",
            model_loaded=False,
            feature_engineer_loaded=False,
            explainer_loaded=False,
            model_version=MODEL_VERSION,
        )

    ready = detector.is_ready()
    all_ok = all(ready.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        model_loaded=ready["model_loaded"],
        feature_engineer_loaded=ready["feature_engineer_loaded"],
        explainer_loaded=ready["explainer_loaded"],
        model_version=MODEL_VERSION,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["meta"])
async def metrics() -> MetricsResponse:
    """Service-level metrics since startup."""
    latencies = _state["latencies_ms"]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    uptime = time.time() - _state["started_at"] if _state["started_at"] else 0.0

    return MetricsResponse(
        requests_total=_state["requests_total"],
        predictions_total=_state["predictions_total"],
        flags_total=_state["flags_total"],
        avg_latency_ms=round(avg_latency, 2),
        uptime_seconds=round(uptime, 1),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(txn: TransactionRequest) -> PredictionResponse:
    """Score one transaction for fraud risk."""
    _state["requests_total"] += 1

    detector = _state["detector"]
    if detector is None:
        raise HTTPException(status_code=503, detail="Service not ready: model not loaded.")

    try:
        result = detector.predict(txn.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    _state["predictions_total"] += 1
    if result.decision == "flag":
        _state["flags_total"] += 1
    _state["latencies_ms"].append(result.latency_ms)

    return result


@app.get("/", tags=["meta"])
async def root() -> dict:
    """Friendly landing endpoint."""
    return {
        "service": "Fraud Detection API",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }
