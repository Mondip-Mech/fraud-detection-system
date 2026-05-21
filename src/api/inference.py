"""
Model loading and inference logic for the Fraud Detection API.

Separated from main.py so prediction logic can be unit-tested without
spinning up an HTTP server. The FastAPI app simply wires HTTP to these functions.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.api.schemas import FeatureContribution, PredictionResponse

# Models directory — resolved relative to project root
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

MODEL_VERSION = "lgbm-v1"


class FraudDetector:
    """Loads all production artifacts and runs predictions."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.feature_engineer = None
        self.model = None
        self.explainer = None
        self.metadata: dict[str, Any] = {}
        self.threshold: float = 0.5
        self.feature_names: list[str] = []
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all four artifacts from disk. Called once at startup."""
        with open(self.models_dir / "feature_engineer.pkl", "rb") as f:
            self.feature_engineer = pickle.load(f)

        self.model = joblib.load(self.models_dir / "fraud_model.pkl")
        self.explainer = joblib.load(self.models_dir / "shap_explainer.pkl")

        with open(self.models_dir / "model_metadata.json") as f:
            self.metadata = json.load(f)

        self.threshold = float(self.metadata["optimal_threshold"])
        self.feature_names = list(self.metadata["feature_names"])

    def is_ready(self) -> dict[str, bool]:
        """Status check for /health endpoint."""
        return {
            "model_loaded": self.model is not None,
            "feature_engineer_loaded": self.feature_engineer is not None,
            "explainer_loaded": self.explainer is not None,
        }

    def predict(self, txn: dict, top_k_contributors: int = 5) -> PredictionResponse:
        """Score one transaction for fraud."""
        t0 = time.perf_counter()

        # Build single-row DataFrame from the request dict
        raw_df = pd.DataFrame([txn])

        # Apply Phase 3 feature engineering pipeline
        X = self.feature_engineer.transform(raw_df)

        # Predict fraud probability
        proba_raw = self.model.predict(X, num_iteration=self.model.best_iteration)[0]
        proba = self._safe_float(proba_raw)

        # Decision
        decision = "flag" if proba >= self.threshold else "allow"

        # SHAP explanation
        shap_vals = self.explainer.shap_values(X)
        # Normalize SHAP output: TreeExplainer can return ndarray OR list-of-ndarrays
        if isinstance(shap_vals, list):
            # Binary classifier: take fraud-class SHAP values
            shap_arr = shap_vals[1][0]
        else:
            shap_arr = shap_vals[0]

        # Top K contributors by absolute SHAP value
        contrib_df = pd.DataFrame({
            "feature": X.columns,
            "value": X.iloc[0].values,
            "shap_value": shap_arr,
        })
        contrib_df["abs_shap"] = contrib_df["shap_value"].abs()
        top_k = contrib_df.nlargest(top_k_contributors, "abs_shap")

        contributors = [
            FeatureContribution(
                feature=str(row["feature"]),
                value=self._stringify_value(row["value"]),
                shap_value=self._safe_float(row["shap_value"]),
            )
            for _, row in top_k.iterrows()
        ]

        latency_ms = (time.perf_counter() - t0) * 1000

        return PredictionResponse(
            transaction_id=int(txn["TransactionID"]),
            fraud_probability=proba,
            decision=decision,
            threshold=self.threshold,
            top_contributors=contributors,
            model_version=MODEL_VERSION,
            latency_ms=round(latency_ms, 2),
        )

    @staticmethod
    def _stringify_value(value: Any) -> str | float | None:
        """Convert feature value to JSON-serializable form."""
        if value is None:
            return None
        if isinstance(value, float | np.floating):
            f = float(value)
            if np.isnan(f) or np.isinf(f):
                return None
            return f
        if isinstance(value, int | np.integer):
            return int(value)
        return str(value)

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Convert to float, replacing NaN/Inf with 0.0 (SHAP defaults)."""
        if value is None:
            return 0.0
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return 0.0
        return f
