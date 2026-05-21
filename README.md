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

End-to-end ML system for detecting fraudulent credit card transactions, built on the IEEE-CIS Fraud Detection dataset (590K transactions, 3.5% fraud rate).

## Live Demo

- **API Docs (Swagger UI):** [/docs](docs)
- **Health Check:** [/health](health)
- **Metrics:** [/metrics](metrics)

Try POST `/predict` in the Swagger UI with the pre-filled example payload.

## Results

| Metric | Value |
|---|---|
| PR-AUC | **0.473** (14× random baseline) |
| ROC-AUC | 0.898 |
| Recall @ 1% FPR | **41.3%** |
| Optimal threshold | 0.376 |
| Business savings | **$231K** on 118K validation transactions |

## Tech Stack

- **Modeling:** LightGBM, scikit-learn, SHAP
- **Experiment Tracking:** MLflow
- **Serving:** FastAPI, Pydantic, Docker
- **Testing:** pytest (17 tests)
- **CI/CD:** GitHub Actions

## Project Structure

fraud-detection-system/
├── src/
│   ├── api/          # FastAPI service + Pydantic schemas
│   ├── features/     # Feature engineering pipeline
│   └── models/       # LightGBM training + MLflow tracking
├── notebooks/        # EDA, modeling, evaluation
├── tests/            # pytest suite (17 tests)
├── models/           # Production artifacts (LFS)
└── reports/figures/  # EDA + evaluation plots

## Author

**Mondip Mech** — [GitHub](https://github.com/Mondip-Mech) · [LinkedIn](https://www.linkedin.com/in/mondip-mech)