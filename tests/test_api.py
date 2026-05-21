"""Integration tests for the Fraud Detection API."""

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
        assert r.status_code == 422  # Pydantic validation error

    def test_predict_invalid_product_code(self):
        bad = _sample_transaction()
        bad["ProductCD"] = "Z"  # Not in allowed set
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
            # Make 2 prediction requests
            client.post("/predict", json=_sample_transaction())
            client.post("/predict", json=_sample_transaction())
            r = client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert body["predictions_total"] >= 2
        assert body["avg_latency_ms"] > 0


class TestRootEndpoint:
    def test_root_returns_service_info(self):
        with TestClient(app) as client:
            r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert "service" in body
        assert body["docs"] == "/docs"