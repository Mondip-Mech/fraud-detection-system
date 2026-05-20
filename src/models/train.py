"""
LightGBM fraud detection model training.

This module provides reusable training functions that can be called from:
- Notebooks (for experimentation and analysis)
- CI/CD pipelines (for reproducible training runs)
- Scheduled retraining jobs

Design:
- All hyperparameters explicit (no defaults hidden in function bodies)
- Every run logged to MLflow with params, metrics, and model artifact
- Deterministic via fixed random_state
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---- Default hyperparameters ----

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",  # PR-AUC, the right metric for imbalanced classification
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 100,
    "scale_pos_weight": 28,  # ~ (1 - fraud_rate) / fraud_rate, from EDA
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics suitable for imbalanced fraud detection.

    Returns
    -------
    dict with PR-AUC, ROC-AUC, precision, recall, F1 at the given threshold,
    plus recall@1%FPR which is the most business-relevant metric for fraud.
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_predicted_positive": int(y_pred.sum()),
        "threshold": threshold,
    }

    # Recall @ 1% FPR — "if we tolerate flagging 1% of legitimate transactions,
    # what fraction of frauds do we catch?" This is the operational metric.
    metrics["recall_at_1pct_fpr"] = recall_at_fpr(y_true, y_proba, target_fpr=0.01)

    return metrics


def recall_at_fpr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """Recall at a fixed false-positive rate.

    Standard fraud-detection operational metric. At a fixed acceptable FPR
    (e.g., 1% of legit transactions flagged for review), what fraction of
    actual frauds do we catch?
    """
    # Sort by predicted probability descending
    order = np.argsort(-y_proba)
    y_true_sorted = y_true[order]

    n_neg = (y_true == 0).sum()
    n_pos = (y_true == 1).sum()

    max_fp_allowed = int(target_fpr * n_neg)
    fp_count = 0
    tp_count = 0

    for label in y_true_sorted:
        if label == 0:
            fp_count += 1
            if fp_count > max_fp_allowed:
                break
        else:
            tp_count += 1

    return tp_count / n_pos if n_pos > 0 else 0.0


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    experiment_name: str = "fraud-detection",
    run_name: str | None = None,
) -> tuple[lgb.Booster, dict[str, float]]:
    """Train a LightGBM model with MLflow tracking.

    Returns
    -------
    model : lgb.Booster
    metrics : dict of validation metrics
    """
    params = {**DEFAULT_PARAMS, **(params or {})}

    # Identify categorical columns by dtype
    cat_cols = X_train.select_dtypes(include="category").columns.tolist()

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("num_boost_round", num_boost_round)
        mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_val_samples", len(X_val))
        mlflow.log_param("train_fraud_rate", float(y_train.mean()))
        mlflow.log_param("val_fraud_rate", float(y_val.mean()))
        mlflow.log_param("categorical_features", ",".join(cat_cols))

        train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[train_set, val_set],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        y_val_proba = model.predict(X_val, num_iteration=model.best_iteration)
        metrics = compute_metrics(y_val.values, y_val_proba)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("best_iteration", model.best_iteration)

        # Save model artifact
        mlflow.lightgbm.log_model(model, artifact_path="model")

        return model, metrics