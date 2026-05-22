"""
Fraud Detection API — FastAPI service.

Endpoints:
- GET  /          — landing page (HTML)
- GET  /health    — readiness check (JSON)
- GET  /metrics   — service-level metrics (JSON)
- POST /predict   — score a single transaction (JSON)
- GET  /docs      — interactive OpenAPI documentation
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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


LANDING_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fraud Detection API</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #fff;
    min-height: 100vh;
    padding: 40px 20px;
    line-height: 1.6;
  }
  .container { max-width: 900px; margin: 0 auto; }
  header { text-align: center; margin-bottom: 40px; }
  .badge {
    display: inline-block;
    background: rgba(46, 204, 113, 0.2);
    color: #2ecc71;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 20px;
    border: 1px solid #2ecc71;
  }
  .badge::before { content: "● "; }
  h1 { font-size: 2.5em; margin-bottom: 10px; font-weight: 700; }
  .subtitle { font-size: 1.15em; color: #b8c5d6; max-width: 700px; margin: 0 auto; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin: 40px 0; }
  .stat-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    transition: transform 0.2s;
  }
  .stat-card:hover { transform: translateY(-4px); border-color: rgba(74, 144, 226, 0.5); }
  .stat-value { font-size: 2em; font-weight: 700; color: #4a90e2; }
  .stat-label { font-size: 0.9em; color: #b8c5d6; margin-top: 6px; }
  .actions { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 40px 0; }
  .btn {
    display: block;
    background: #4a90e2;
    color: white;
    padding: 18px 24px;
    text-decoration: none;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
    font-size: 1em;
    transition: all 0.2s;
    border: 1px solid #4a90e2;
  }
  .btn:hover { background: #357abd; transform: translateY(-2px); box-shadow: 0 8px 20px rgba(74, 144, 226, 0.3); }
  .btn-secondary { background: transparent; }
  .btn-secondary:hover { background: rgba(74, 144, 226, 0.1); }
  .section { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 30px; margin: 30px 0; }
  .section h2 { font-size: 1.3em; margin-bottom: 16px; color: #4a90e2; }
  pre { background: #0a0a14; padding: 18px; border-radius: 8px; overflow-x: auto; font-size: 0.9em; line-height: 1.5; border: 1px solid rgba(255, 255, 255, 0.08); }
  code { color: #87ceeb; }
  .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; }
  .feature { padding: 16px; background: rgba(255, 255, 255, 0.03); border-radius: 8px; border-left: 3px solid #4a90e2; }
  .feature-title { font-weight: 600; margin-bottom: 6px; color: #87ceeb; }
  .feature-desc { font-size: 0.9em; color: #b8c5d6; }
  footer { text-align: center; margin-top: 50px; padding-top: 30px; border-top: 1px solid rgba(255, 255, 255, 0.1); color: #7a8aa0; font-size: 0.9em; }
  footer a { color: #4a90e2; text-decoration: none; }
  footer a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="badge">Service Healthy · MODEL_VERSION_HERE</div>
    <h1>🛡️ Fraud Detection API</h1>
    <p class="subtitle">
      Real-time credit card fraud detection powered by LightGBM.
      Returns fraud probability, decision, and SHAP-explained top feature contributors per prediction.
    </p>
  </header>

  <div class="stats">
    <div class="stat-card">
      <div class="stat-value">0.473</div>
      <div class="stat-label">PR-AUC<br/>(14× random)</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">41.3%</div>
      <div class="stat-label">Recall @ 1% FPR</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">$232K</div>
      <div class="stat-label">Validation savings<br/>(57% cost reduction)</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">&lt;100ms</div>
      <div class="stat-label">Inference latency</div>
    </div>
  </div>

  <div class="actions">
    <a class="btn" href="/docs">📘 Interactive API Docs</a>
    <a class="btn btn-secondary" href="/health">🏥 Health Check</a>
    <a class="btn btn-secondary" href="/metrics">📊 Metrics</a>
  </div>

  <div class="section">
    <h2>Try the API</h2>
    <p style="margin-bottom: 16px; color: #b8c5d6;">Send a POST request to <code>/predict</code>:</p>
    <pre><code>curl -X POST https://mechscientist26-fraud-detection-system.hf.space/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "TransactionID": 12345678,
    "TransactionDT": 7776000,
    "TransactionAmt": 149.50,
    "ProductCD": "W",
    "card1": 13926,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "DeviceType": "desktop"
  }'</code></pre>
  </div>

  <div class="section">
    <h2>Architecture</h2>
    <div class="features">
      <div class="feature">
        <div class="feature-title">Feature Engineering</div>
        <div class="feature-desc">36 engineered features: cyclical temporal encoding, UID-based velocity, Bayesian-smoothed email risk</div>
      </div>
      <div class="feature">
        <div class="feature-title">Model</div>
        <div class="feature-desc">LightGBM (1,250 trees), trained on 472K transactions with scale_pos_weight=28 for imbalance</div>
      </div>
      <div class="feature">
        <div class="feature-title">Explainability</div>
        <div class="feature-desc">SHAP TreeExplainer returns top contributing features for every prediction</div>
      </div>
      <div class="feature">
        <div class="feature-title">Production</div>
        <div class="feature-desc">FastAPI · Pydantic validation · Docker · GitHub Actions CI · Evidently drift monitoring</div>
      </div>
    </div>
  </div>

  <footer>
    <p>
      Built by <a href="https://github.com/Mondip-Mech">Mondip Mech</a> ·
      <a href="https://github.com/Mondip-Mech/fraud-detection-system">GitHub Repository</a> ·
      <a href="https://mechscientist26-fraud-detection-monitoring.hf.space">Monitoring Dashboard</a>
    </p>
  </footer>
</div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, tags=["meta"])
async def root() -> HTMLResponse:
    """Landing page with project info and links to API endpoints."""
    html = LANDING_PAGE_HTML.replace("MODEL_VERSION_HERE", MODEL_VERSION)
    return HTMLResponse(content=html)


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