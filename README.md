---
title: Fraud Detection API
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Real-time credit card fraud detection — LightGBM, SHAP, FastAPI
---



# Real-Time Credit Card Fraud Detection System

End-to-end ML system for detecting fraudulent credit card transactions in real time, built on the IEEE-CIS Fraud Detection dataset (590K transactions, 400+ features).

🚧 **Under active development.** Live demo and full documentation coming soon.

## Planned Tech Stack

- **Modeling:** LightGBM, XGBoost, scikit-learn
- **Explainability:** SHAP
- **Experiment Tracking:** MLflow
- **Serving:** FastAPI, Docker
- **Monitoring:** Evidently AI, Streamlit
- **CI/CD:** GitHub Actions, pytest, ruff, mypy

## Project Structure
fraud-detection-system/
├── src/
│   ├── data/         # Data loading and preprocessing
│   ├── features/     # Feature engineering pipeline
│   ├── models/       # Training and evaluation
│   ├── monitoring/   # Drift detection
│   └── api/          # FastAPI service
├── notebooks/        # EDA and experimentation
├── tests/            # pytest test suite
├── configs/          # YAML configuration files
└── .github/workflows/ # CI/CD pipelines


## Author

**Mondip Mech** — [GitHub](https://github.com/Mondip-Mech) · [LinkedIn](https://www.linkedin.com/in/mondip-mech)