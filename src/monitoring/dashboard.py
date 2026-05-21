"""
Streamlit dashboard for fraud detection monitoring.

Sections:
1. Live API health and metrics (calls the FastAPI service)
2. Drift report (embedded Evidently HTML)
3. Recent predictions distribution
4. Model performance summary

Deployable to Streamlit Cloud or HuggingFace Spaces alongside the API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

# Optional: use requests to call the live API if URL is provided
try:
    import requests
except ImportError:
    requests = None

# Configuration
DEFAULT_API_URL = "https://mechscientist26-fraud-detection-system.hf.space"
API_URL = os.environ.get("FRAUD_API_URL", DEFAULT_API_URL)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MODELS_DIR = ROOT / "models"

# --- Page setup ---
st.set_page_config(
    page_title="Fraud Detection Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Fraud Detection — Monitoring Dashboard")
st.caption(
    "Live operational view of the deployed fraud detection service. "
    "Shows API health, data drift, and model performance over time."
)

# --- Sidebar ---
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API URL", value=API_URL, help="Public URL of the deployed FastAPI service")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "### About\n"
    "Monitoring dashboard for the [Fraud Detection API](https://mechscientist26-fraud-detection-system.hf.space/docs). "
    "Built with Streamlit + Evidently AI.\n\n"
    "Reference dataset: Apr-Jun 2018 validation split (118K transactions).\n\n"
    "[GitHub](https://github.com/Mondip-Mech/fraud-detection-system)"
)

# --- Section 1: Live API health ---
st.header("1. Service Health")

col_health, col_meta = st.columns([1, 2])

with col_health:
    if requests is not None:
        try:
            r = requests.get(f"{api_url}/health", timeout=10)
            if r.status_code == 200:
                h = r.json()
                if h.get("status") == "ok":
                    st.success("🟢 Service: **Healthy**")
                else:
                    st.warning(f"🟡 Service: **{h.get('status', 'unknown')}**")
                st.json(h)
            else:
                st.error(f"🔴 Service returned HTTP {r.status_code}")
        except Exception as e:
            st.error(f"🔴 Could not reach API: {e}")
    else:
        st.warning("`requests` library unavailable; install it to enable live health checks.")

with col_meta:
    if requests is not None:
        try:
            r = requests.get(f"{api_url}/metrics", timeout=10)
            if r.status_code == 200:
                m = r.json()
                st.metric("Requests served", f"{m.get('requests_total', 0):,}")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Predictions", f"{m.get('predictions_total', 0):,}")
                col_b.metric("Flagged as fraud", f"{m.get('flags_total', 0):,}")
                col_c.metric("Avg latency", f"{m.get('avg_latency_ms', 0):.1f} ms")
                st.caption(f"Uptime: {m.get('uptime_seconds', 0):.0f} seconds since service start")
        except Exception:
            pass

# --- Section 2: Model metadata ---
st.header("2. Model Information")

metadata_path = MODELS_DIR / "model_metadata.json"
if metadata_path.exists():
    with open(metadata_path) as f:
        meta = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model type", meta.get("model_type", "—"))
    col2.metric("Features", meta.get("n_features", 0))
    col3.metric("Threshold", f"{meta.get('optimal_threshold', 0):.3f}")
    col4.metric("Trees", meta.get("best_iteration", 0))

    vmetrics = meta.get("validation_metrics", {})
    if vmetrics:
        st.subheader("Validation performance (held-out set, 118K transactions)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PR-AUC", f"{vmetrics.get('pr_auc', 0):.4f}")
        m2.metric("ROC-AUC", f"{vmetrics.get('roc_auc', 0):.4f}")
        m3.metric("Recall @ 1% FPR", f"{vmetrics.get('recall_at_1pct_fpr', 0):.4f}")
        m4.metric("F1 @ default", f"{vmetrics.get('f1', 0):.4f}")
else:
    st.info("Model metadata not found. Has the model been trained yet?")

# --- Section 3: Drift report ---
st.header("3. Data Drift Report")

st.markdown(
    "Comparison of recent inference data against the training-time validation distribution. "
    "Built with **Evidently AI**. Drift is computed per-feature using Wasserstein distance "
    "(numerics) and chi-square (categoricals)."
)

drift_path = REPORTS / "drift_report.html"
if drift_path.exists():
    with open(drift_path, encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=800, scrolling=True)
else:
    st.warning(
        "No drift report found. Generate one by running:\n\n"
        "```bash\npython -m src.monitoring.drift_report\n```"
    )

# --- Footer ---
st.markdown("---")
st.markdown(
    "Built by [Mondip Mech](https://github.com/Mondip-Mech) · "
    "Part of the [Fraud Detection System](https://github.com/Mondip-Mech/fraud-detection-system) project."
)
