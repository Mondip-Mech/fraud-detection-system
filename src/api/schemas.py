"""
Pydantic schemas for the Fraud Detection API.

These models define the request/response contract and provide automatic
validation. FastAPI uses them to generate OpenAPI documentation at /docs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """A single credit card transaction to score for fraud.

    Only the most commonly-available fields are required. Optional fields
    can be omitted; the feature pipeline handles missing values natively.
    """

    model_config = ConfigDict(json_schema_extra={
        "example": {
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
    })

    # Required core fields
    TransactionID: int = Field(..., description="Unique transaction identifier")
    TransactionDT: int = Field(..., ge=0, description="Seconds since reference start date (2017-12-01)")
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD: Literal["W", "C", "H", "R", "S"] = Field(..., description="Product code")
    card1: int = Field(..., description="Card identifier 1")

    # Commonly-present optional fields
    card2: float | None = None
    card3: float | None = None
    card4: Literal["visa", "mastercard", "discover", "american express"] | None = None
    card5: float | None = None
    card6: Literal["debit", "credit", "debit or credit", "charge card"] | None = None
    addr1: float | None = None
    addr2: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: Literal["desktop", "mobile"] | None = None
    DeviceInfo: str | None = None

    # Aggregation/count features (IEEE-CIS C1-C14, D1-D15) — all optional
    C1: float | None = None
    C2: float | None = None
    C13: float | None = None
    D1: float | None = None
    D2: float | None = None
    D15: float | None = None
    dist1: float | None = None
    id_01: float | None = None
    id_02: float | None = None


class FeatureContribution(BaseModel):
    """A single feature's contribution to the prediction (from SHAP)."""

    feature: str
    value: str | float | None  # Stringified for readability
    shap_value: float = Field(..., description="Contribution to log-odds of fraud. Positive=pushes toward fraud.")


class PredictionResponse(BaseModel):
    """The fraud scoring result for one transaction."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "transaction_id": 12345678,
                "fraud_probability": 0.0582,
                "decision": "allow",
                "threshold": 0.376,
                "top_contributors": [
                    {"feature": "C13", "value": 1.0, "shap_value": 0.558},
                    {"feature": "C1", "value": 1.0, "shap_value": -0.314}
                ],
                "model_version": "lgbm-v1",
                "latency_ms": 8.4
            }
        }
    )

    transaction_id: int
    fraud_probability: float = Field(..., ge=0, le=1, description="Model-predicted P(fraud)")
    decision: Literal["allow", "flag"] = Field(..., description="Action based on threshold")
    threshold: float = Field(..., description="Decision threshold used")
    top_contributors: list[FeatureContribution] = Field(
        ..., description="Top SHAP contributors for this prediction"
    )
    model_version: str
    latency_ms: float = Field(..., description="Inference time in milliseconds")


class HealthResponse(BaseModel):
    """Service health check."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded", "down"]
    model_loaded: bool
    feature_engineer_loaded: bool
    explainer_loaded: bool
    model_version: str


class MetricsResponse(BaseModel):
    """Service-level metrics."""

    requests_total: int
    predictions_total: int
    flags_total: int
    avg_latency_ms: float
    uptime_seconds: float