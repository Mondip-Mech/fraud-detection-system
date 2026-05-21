---
title: Fraud Detection API
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: LightGBM fraud detection — FastAPI, SHAP, Docker
---

# Real-Time Credit Card Fraud Detection System

![CI](https://github.com/Mondip-Mech/fraud-detection-system/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg?logo=docker)
![HuggingFace](https://img.shields.io/badge/🤗-deployed-yellow.svg)

End-to-end production ML system for detecting fraudulent credit card transactions. Trained on **590K transactions** from the IEEE-CIS Fraud Detection dataset (3.5% fraud rate); served via FastAPI in Docker on HuggingFace Spaces with live monitoring.

## 🚀 Live Demo

| Service | URL | Description |
|---|---|---|
| **🔌 API (Swagger UI)** | [fraud-detection-system.hf.space/docs](https://mechscientist26-fraud-detection-system.hf.space/docs) | Interactive API documentation — try `/predict` directly |
| **📊 Monitoring Dashboard** | [fraud-detection-monitoring.hf.space](https://mechscientist26-fraud-detection-monitoring.hf.space) | Live service health, model metrics, and Evidently drift report |
| **🏥 Health endpoint** | [/health](https://mechscientist26-fraud-detection-system.hf.space/health) | JSON health check |

> 💡 Try POST `/predict` in the [Swagger UI](https://mechscientist26-fraud-detection-system.hf.space/docs) with the pre-filled example payload — get back a fraud probability and SHAP-explained top contributing features in under 100ms.

## 📊 Results

| Metric | Value | Context |
|---|---|---|
| **PR-AUC** | **0.473** | 14× the random baseline (0.034) |
| **ROC-AUC** | 0.898 | — |
| **Recall @ 1% FPR** | **41.3%** | Catches 41% of fraud while flagging just 1% of legitimate transactions |
| **Optimal threshold** | 0.376 | Optimized against business cost matrix ($100/missed fraud, $5/false alarm) |
| **Business impact** | **$232K savings** | Validation set: $406K do-nothing cost → $175K with model (57% reduction on 118K transactions) |
| **Inference latency** | <100ms | Per-transaction prediction including SHAP explanation |

## 🏗️ Architecture

```
Raw Transaction (JSON)
      │
      ▼
┌──────────────────────────┐
│   FastAPI /predict       │  Pydantic validation
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ FraudFeatureEngineer     │  36 engineered features
│ (sklearn-compatible)     │  - Cyclical temporal (sin/cos)
│                          │  - UID-based velocity
│                          │  - Bayesian-smoothed email risk
│                          │  - Missing-data indicators
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ LightGBM Classifier      │  scale_pos_weight=28
│ (1250 trees, 36 feats)   │  Early-stopped on PR-AUC
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ SHAP TreeExplainer       │  Per-prediction feature contributions
└──────────────────────────┘
      │
      ▼
{
  "fraud_probability": 0.058,
  "decision": "allow",
  "top_contributors": [...]
}
```

## 🧰 Tech Stack

| Layer | Tools | Why |
|---|---|---|
| **Modeling** | LightGBM, scikit-learn, SHAP | Tree-based: handles NaN natively, fast on 472K rows, supports feature interactions |
| **Experiment tracking** | MLflow | Reproducible runs with metrics, params, and model versioning |
| **Feature engineering** | pandas, NumPy | Custom sklearn-compatible class prevents train/serve skew |
| **API** | FastAPI, Pydantic | Async, auto-generated OpenAPI docs, runtime input validation |
| **Containerization** | Docker (multi-stage), python:3.11-slim | Small image (~1GB), non-root user, baked-in healthcheck |
| **Deployment** | HuggingFace Spaces (Docker SDK + Streamlit SDK) | Free hosting, public URLs, automatic builds on push |
| **Monitoring** | Evidently AI, Streamlit | Data drift detection (Wasserstein for numerics, chi-square for categoricals) |
| **Testing** | pytest, fastapi.TestClient | 17 tests covering feature engineering and API endpoints |
| **CI/CD** | GitHub Actions, ruff, mypy | Auto-test + lint + type-check on every push |

## 📂 Project Structure

```
fraud-detection-system/
├── .github/workflows/
│   └── ci.yml                          # GitHub Actions CI pipeline
├── src/
│   ├── api/
│   │   ├── main.py                     # FastAPI app, endpoints, state
│   │   ├── inference.py                # Model loading, prediction logic
│   │   └── schemas.py                  # Pydantic request/response models
│   ├── features/
│   │   └── build_features.py           # FraudFeatureEngineer (sklearn-compatible)
│   ├── models/
│   │   └── train.py                    # LightGBM training + MLflow tracking
│   └── monitoring/
│       ├── drift_report.py             # Evidently drift report generator
│       └── dashboard.py                # Streamlit monitoring UI
├── tests/
│   ├── test_build_features.py          # 9 unit tests (feature engineering)
│   └── test_api.py                     # 8 integration tests (API)
├── notebooks/
│   ├── 00_data_load_check.ipynb        # Phase 1: data sanity
│   ├── 01_eda.ipynb                    # Phase 2: 7 EDA plots
│   ├── 02_feature_engineering.ipynb    # Phase 3: pipeline application
│   └── 03_modeling.ipynb               # Phase 4: training, threshold opt, SHAP
├── models/                             # (gitignored) Trained artifacts
│   ├── feature_engineer.pkl
│   ├── fraud_model.pkl
│   ├── shap_explainer.pkl
│   └── model_metadata.json
├── reports/
│   ├── drift_report.html               # Sample Evidently report
│   └── figures/                        # EDA + evaluation plots (LFS)
├── Dockerfile                          # Multi-stage build for production
├── .dockerignore
├── requirements.txt                    # Pinned dependencies
└── README.md                           # This file
```

## 🔑 Key ML Engineering Decisions

### 1. Tree-based model over neural network or logistic regression
The data has **214 of 434 columns >50% missing** (bimodal pattern: most columns either fully populated or heavily sparse). LightGBM handles NaN natively without imputation, captures non-linear feature interactions, and trains in under 90 seconds on 472K rows.

### 2. PR-AUC over ROC-AUC for evaluation
With 3.5% positive class, ROC-AUC overstates performance. A trivial "always predict not fraud" achieves 96.5% accuracy. Trained models use **PR-AUC for early stopping** and report **Recall @ 1% FPR** as the operational metric — what fraction of fraud do we catch when we tolerate flagging 1% of legitimate transactions?

### 3. Time-based train/val split (NOT random)
Fraud patterns drift. Random shuffles leak future patterns into training. The pipeline sorts on `TransactionDT` and uses the first 80% (Dec 2017–Apr 2018) for training and the last 20% (Apr–Jun 2018) for validation — realistic temporal holdout.

### 4. Composite UID for velocity features
EDA discovery: `card1` is NOT a unique card identifier — high-frequency `card1` values have *lower* fraud rates than low-frequency ones (0.91× lift). Used `card1 + card2 + addr1 + P_emaildomain` as a composite UID for velocity/aggregation features instead.

### 5. Threshold optimization against a business cost matrix
Default threshold 0.5 is rarely optimal for imbalanced classification. Scanned thresholds from 0.01 to 0.99, minimized total cost = `$100 × false_negatives + $5 × false_positives`. **Optimal threshold = 0.376** with 80.8% recall and $174,850 total cost (vs $406,400 do-nothing baseline = 57% reduction).

## 🐛 Bug-Find Stories

These bugs surfaced during development and made the system more robust:

1. **Train/serve schema drift via categorical levels** — `card6` had different category lists between train and val splits (a rare value appeared in only one). Pandas reported dtype as `category` in both, but the underlying categories differed, causing silent encoding mismatch. *Fix:* Made `FraudFeatureEngineer` stateful for categoricals — `fit()` records training category lists, `transform()` enforces them. Unseen categories at inference become NaN (LightGBM handles natively).

2. **NaN serialization crash in API responses** — Any optional transaction field that arrived as `null` became `NaN` after feature engineering. When SHAP picked one as a top contributor, the response failed JSON serialization with `ValueError: Out of range float values are not JSON compliant`. *Fix:* Sanitization layer that converts NaN/Inf to `None` (feature values) and `0.0` (SHAP values) before the response leaves inference.

3. **CI pickle errors with LFS-stored artifacts** — Initial CI run failed with `_pickle.UnpicklingError: invalid load key, 'v'`. The `v` was the first byte of "version", the LFS pointer file header. GitHub Actions was checking out the LFS pointer text stubs instead of the actual binaries. *Fix:* `_is_real_artifact()` helper checks for the LFS pointer header before treating a file as a real pickle.

## 🛠️ Local Setup

```bash
# 1. Clone and set up environment
git clone https://github.com/Mondip-Mech/fraud-detection-system.git
cd fraud-detection-system
python -m venv .venv
.venv\Scripts\activate  # Windows; or `source .venv/bin/activate` on Mac/Linux
pip install -r requirements.txt

# 2. Download IEEE-CIS dataset
# Place train_transaction.csv and train_identity.csv in data/raw/
# From: https://www.kaggle.com/c/ieee-fraud-detection/data

# 3. Run the pipeline notebooks in order
jupyter notebook
#   notebooks/01_eda.ipynb
#   notebooks/02_feature_engineering.ipynb
#   notebooks/03_modeling.ipynb

# 4. Run the API locally
uvicorn src.api.main:app --port 8000
# Visit http://localhost:8000/docs

# 5. Run the monitoring dashboard
streamlit run src/monitoring/dashboard.py
# Visit http://localhost:8501

# 6. Run tests
pytest tests/ -v
```

## 📦 Docker

```bash
docker build -t fraud-api:latest .
docker run --rm -d -p 8000:7860 --name fraud-api fraud-api:latest
curl http://localhost:8000/health
```

## 👤 Author

**Mondip Mech** — Data Analyst with 2 YOE transitioning into Data Science / ML Engineering.

- 🐙 [GitHub](https://github.com/Mondip-Mech)
- 💼 [LinkedIn](https://www.linkedin.com/in/mondip-mech)
- 📧 mechmondip@gmail.com

## 📄 License

MIT