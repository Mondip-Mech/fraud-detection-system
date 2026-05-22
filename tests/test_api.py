"""Integration tests for the Fraud Detection API.

These tests require the trained production artifacts in models/. In CI environments
where artifacts don't exist (they're gitignored), all tests in this file are skipped.
To enable them locally, run notebook 03_modeling.ipynb to generate the artifacts.
"""

from pathlib import Path

import pytest

# Detect if the production model artifacts are available
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _is_real_artifact(path: Path) -> bool:
    """Check if file exists AND is a real artifact (not a Git LFS pointer stub).

    LFS pointer files start with 'version https://git-lfs.github.com/spec/v1'.
    Real pickle files start with bytes like \\x80\\x04 (pickle protocol header).
    """
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            first_bytes = f.read(20)
        # LFS pointer starts with 'version '
        if first_bytes.startswith(b"version "):
            return False
        return True
    except OSError:
        return False


HAS_ARTIFACTS = all([
    _is_real_artifact(MODELS_DIR / "feature_engineer.pkl"),
    _is_real_artifact(MODELS_DIR / "fraud_model.pkl"),
    _is_real_artifact(MODELS_DIR / "shap_explainer.pkl"),
    (MODELS_DIR / "model_metadata.json").exists(),  # JSON has no LFS issue
])

# Skip all tests in this file when artifacts are missing (e.g., in CI)
pytestmark = pytest.mark.skipif(
    not HAS_ARTIFACTS,
    reason="Production model artifacts not available (run notebook 03_modeling.ipynb to generate)",
)

# Import the app only after the skip check — avoids import errors in CI
if HAS_ARTIFACTS:
    from fastapi.testclient import TestClient

    from src.api.main import app


def _sample_transaction() -> dict:
    return {
        "TransactionID": 12345678,
        "TransactionDT": 7776000,
        "TransactionAmt": 149.50,
        "ProductCD": "W",
        "card1": 13926,
        "card2": 404.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 142.0,
        "card6": "debit",
        "addr1": 315.0,
        "addr2": 87.0,
        "P_emaildomain": "gmail.com",
        "DeviceType": "desktop",
        "C1": 1.0,
        "C2": 1.0,
        "C13": 1.0,
    }


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        with TestClient(app) as client:
            r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["feature_engineer_loaded"] is True
        assert body["explainer_loaded"] is True


class TestPredictEndpoint:
    def test_predict_valid_transaction(self):
        with TestClient(app) as client:
            r = client.post("/predict", json=_sample_transaction())
        assert r.status_code == 200
        body = r.json()
        assert "fraud_probability" in body
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["decision"] in ("allow", "flag")
        assert isinstance(body["top_contributors"], list)
        assert len(body["top_contributors"]) > 0
        assert body["latency_ms"] > 0

    def test_predict_missing_required_field(self):
        bad = _sample_transaction()
        del bad["TransactionAmt"]
        with TestClient(app) as client:
            r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_invalid_product_code(self):
        bad = _sample_transaction()
        bad["ProductCD"] = "Z"
        with TestClient(app) as client:
            r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_negative_amount_rejected(self):
        bad = _sample_transaction()
        bad["TransactionAmt"] = -10.0
        with TestClient(app) as client:
            r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_decision_consistent_with_threshold(self):
        with TestClient(app) as client:
            r = client.post("/predict", json=_sample_transaction())
        body = r.json()
        if body["fraud_probability"] >= body["threshold"]:
            assert body["decision"] == "flag"
        else:
            assert body["decision"] == "allow"


class TestMetricsEndpoint:
    def test_metrics_tracks_requests(self):
        with TestClient(app) as client:
            client.post("/predict", json=_sample_transaction())
            client.post("/predict", json=_sample_transaction())
            r = client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert body["predictions_total"] >= 2
        assert body["avg_latency_ms"] > 0


class TestRootEndpoint:
    def test_root_returns_landing_page(self):
        with TestClient(app) as client:
            r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Fraud Detection API" in r.text
        assert "/docs" in r.text  # Link to API docs present
