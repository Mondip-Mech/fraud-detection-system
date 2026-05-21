"""
Generate Evidently AI drift reports comparing reference vs current data.

Reference data: a representative sample from validation set (Phase 3 output).
Current data: simulated "production" traffic, optionally with injected drift
              for demonstration purposes.

In a real deployment, current data would come from logging actual API traffic
to a database. For this portfolio project we simulate by sampling val + adding
controlled drift to specific features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Evidently 0.4.x API
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)

# Project paths
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def load_reference_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load engineered validation set as the reference distribution."""
    X = pd.read_parquet(PROCESSED / "X_val.parquet")
    y = pd.read_parquet(PROCESSED / "y_val.parquet")["isFraud"]
    return X, y


def simulate_current_data(
    reference: pd.DataFrame,
    n_samples: int = 5000,
    drift_seed: int = 42,
    drift_strength: float = 0.3,
) -> pd.DataFrame:
    """Sample reference, then inject controlled drift on a few features.

    This simulates what a slightly-shifted production distribution might look
    like — useful for demonstrating that the monitoring catches real drift.

    Drift injected:
    - 'amt_log': shifted upward by drift_strength * std (transaction amounts trending higher)
    - 'hour': mass shifted toward early-morning hours (8% more 5-9 AM traffic)
    - 'email_risk': slight uniform increase (riskier email mix)

    Other features are sampled without modification so non-drifted features
    show as stable in the report.
    """
    rng = np.random.RandomState(drift_seed)

    # Random subsample
    idx = rng.choice(len(reference), size=n_samples, replace=False)
    current = reference.iloc[idx].copy().reset_index(drop=True)

    # Drift 1: shift amount_log upward
    amt_std = current["amt_log"].std()
    current["amt_log"] = current["amt_log"] + drift_strength * amt_std
    # Keep raw amt consistent
    current["amt"] = np.expm1(current["amt_log"]).astype(np.float32)

    # Drift 2: shift hour distribution toward early morning (5-9 AM)
    # Recall EDA: hour 7 had 10.6% fraud rate vs 2.4% at hour 13
    n_shift = int(0.08 * len(current))  # 8% of rows
    shift_indices = rng.choice(len(current), size=n_shift, replace=False)
    current.loc[shift_indices, "hour"] = rng.randint(5, 10, size=n_shift).astype(np.int8)
    # Recompute hour_sin/hour_cos for consistency
    current["hour_sin"] = np.sin(2 * np.pi * current["hour"] / 24).astype(np.float32)
    current["hour_cos"] = np.cos(2 * np.pi * current["hour"] / 24).astype(np.float32)

    # Drift 3: email risk shifted up slightly
    current["email_risk"] = (current["email_risk"] + 0.005).clip(0, 0.5).astype(np.float32)

    return current


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
) -> dict:
    """Generate an Evidently HTML drift report. Returns a summary dict."""
    # Drop categorical columns — Evidently 0.4.x has quirks with categoricals
    # built via pd.Categorical with custom levels. We focus drift on numerics.
    cat_cols = reference.select_dtypes(include="category").columns.tolist()
    ref_numeric = reference.drop(columns=cat_cols)
    cur_numeric = current.drop(columns=cat_cols)

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=ref_numeric, current_data=cur_numeric)
    report.save_html(str(output_path))

    # Extract summary stats from the report dict
    result_dict = report.as_dict()
    drift_metrics = result_dict["metrics"][0]["result"]

    return {
        "n_features_total": drift_metrics["number_of_columns"],
        "n_features_drifted": drift_metrics["number_of_drifted_columns"],
        "share_drifted": drift_metrics["share_of_drifted_columns"],
        "dataset_drift": drift_metrics["dataset_drift"],
        "report_path": str(output_path),
    }


def generate_predictions_for_monitoring(
    current: pd.DataFrame,
    model,
    explainer=None,
) -> pd.DataFrame:
    """Score the current dataset, return df with predictions and probabilities."""
    proba = model.predict(current, num_iteration=model.best_iteration)
    return pd.DataFrame({
        "fraud_probability": proba,
        "predicted_fraud": (proba >= 0.376).astype(int),  # Phase 4 optimal threshold
    })


if __name__ == "__main__":
    print("Loading reference data...")
    X_ref, y_ref = load_reference_data()
    print(f"  Reference shape: {X_ref.shape}")

    print("\nSimulating current data with injected drift...")
    X_cur = simulate_current_data(X_ref, n_samples=5000)
    print(f"  Current shape:   {X_cur.shape}")

    print("\nGenerating Evidently drift report...")
    summary = generate_drift_report(
        reference=X_ref,
        current=X_cur,
        output_path=REPORTS / "drift_report.html",
    )

    print("\n=== Drift report summary ===")
    print(f"  Total features:    {summary['n_features_total']}")
    print(f"  Features drifted:  {summary['n_features_drifted']} ({summary['share_drifted']*100:.1f}%)")
    print(f"  Dataset drift:     {summary['dataset_drift']}")
    print(f"  Report saved to:   {summary['report_path']}")