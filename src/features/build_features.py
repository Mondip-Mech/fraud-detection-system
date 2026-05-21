"""
Feature engineering pipeline for fraud detection.

This module provides FraudFeatureEngineer, a sklearn-compatible transformer
that converts raw IEEE-CIS transaction+identity data into a feature matrix
ready for tree-based models (LightGBM/XGBoost).

Design principles:
- Stateful: learn aggregations and encodings from training data via fit(),
  apply deterministically via transform(). No leakage at inference time.
- Tree-friendly: do NOT impute NaN — LightGBM/XGBoost handle missing values
  natively. Imputation only for engineered ratios where NaN would break math.
- Production-ready: same code path runs in training and FastAPI inference.
  No notebook-only helpers, no globals, no hidden state.

Usage:
    >>> fe = FraudFeatureEngineer()
    >>> X_train = fe.fit_transform(train_df)
    >>> X_val = fe.transform(val_df)
    >>> # In production:
    >>> X_request = fe.transform(single_transaction_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Reference start date for IEEE-CIS TransactionDT (seconds since this date)
START_DATE = pd.Timestamp("2017-12-01")

# Email domain risk smoothing parameter — Bayesian shrinkage toward global rate
# Higher = more shrinkage = safer for rare domains. 100 is a sensible default.
EMAIL_SMOOTHING_M = 100

# Composite entity key — our best approximation of a unique card/user
UID_COLS = ["card1", "card2", "addr1", "P_emaildomain"]
# Categorical columns whose category lists are learned during fit() and enforced at transform()
CATEGORICAL_COLS = ["ProductCD", "card4", "card6", "DeviceType"]



class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible feature engineering for IEEE-CIS fraud detection.

    Parameters
    ----------
    email_smoothing : int, default=100
        Smoothing constant for target-encoded email domain risk.
        Higher values shrink rare-domain estimates toward the global fraud rate.

    Attributes (learned during fit)
    -------------------------------
    global_fraud_rate_ : float
        Overall fraud rate in the training data.
    email_risk_map_ : dict[str, float]
        Smoothed fraud rate per P_emaildomain learned from training.
    uid_amt_stats_ : pd.DataFrame
        Per-UID mean and std of TransactionAmt from training.
    uid_count_map_ : dict
        Transaction count per UID from training.
    feature_names_ : list[str]
        Names of output features after transform.
    """

    def __init__(self, email_smoothing: int = EMAIL_SMOOTHING_M) -> None:
        self.email_smoothing = email_smoothing

    # ---- Public sklearn API ----

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FraudFeatureEngineer":
        """Learn aggregations and target encodings from training data."""
        if y is None:
            raise ValueError("FraudFeatureEngineer.fit() requires y (the isFraud target).")

        X = X.copy()
        X["isFraud"] = y.values

        # Compute UID column for groupby aggregations
        X["UID"] = self._compute_uid(X)

        self.global_fraud_rate_ = float(y.mean())
        self.email_risk_map_ = self._fit_email_risk(X)
        self.uid_amt_stats_ = self._fit_uid_amount_stats(X)
        self.uid_count_map_ = X["UID"].value_counts().to_dict()
        self.cat_categories_ = self._fit_categorical_categories(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned transformations to produce the model-ready feature matrix."""
        self._check_is_fitted()

        X = X.copy()
        X["UID"] = self._compute_uid(X)

        features = {}

        # Group 1: Temporal features
        features.update(self._make_temporal_features(X))

        # Group 2: Amount features
        features.update(self._make_amount_features(X))

        # Group 3: UID-based aggregations (velocity, amount stats)
        features.update(self._make_uid_features(X))

        # Group 4: Email domain risk (target-encoded)
        features.update(self._make_email_features(X))

        # Group 5: Missing-data indicators for high-signal sparse columns
        features.update(self._make_missing_indicators(X))

        # Group 6: Pass-through raw features that tree models handle directly
        features.update(self._pass_through_features(X))

        out = pd.DataFrame(features, index=X.index)
        self.feature_names_ = out.columns.tolist()
        return out

    # ---- Internal helpers (defined in next steps) ----

    @staticmethod
    def _compute_uid(X: pd.DataFrame) -> pd.Series:
        """Compose a UID from card + address + email columns."""
        uid_parts = [X[col].astype(str).fillna("missing") for col in UID_COLS]
        return uid_parts[0].str.cat(uid_parts[1:], sep="_")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "global_fraud_rate_"):
            raise RuntimeError("FraudFeatureEngineer must be fit() before transform().")
        

        # ---- Group 1: Temporal features ----

    @staticmethod
    def _make_temporal_features(X: pd.DataFrame) -> dict:
        """Hour, day-of-week, and cyclical encodings.

        EDA showed fraud rate peaks at hour 7 (10.6%) and troughs at hour 13 (2.4%).
        Cyclical encoding (sin/cos) lets the model treat hour 23 and hour 0 as adjacent.
        """
        dt = START_DATE + pd.to_timedelta(X["TransactionDT"], unit="s")
        hour = dt.dt.hour
        dow = dt.dt.dayofweek

        return {
            "hour": hour.astype(np.int8),
            "day_of_week": dow.astype(np.int8),
            "hour_sin": np.sin(2 * np.pi * hour / 24).astype(np.float32),
            "hour_cos": np.cos(2 * np.pi * hour / 24).astype(np.float32),
            "dow_sin": np.sin(2 * np.pi * dow / 7).astype(np.float32),
            "dow_cos": np.cos(2 * np.pi * dow / 7).astype(np.float32),
            "is_weekend": (dow >= 5).astype(np.int8),
        }

    # ---- Group 2: Amount features ----

    @staticmethod
    def _make_amount_features(X: pd.DataFrame) -> dict:
        """Transaction amount transformations.

        Log-transform handles the heavy right skew seen in EDA.
        Decimal part captures patterns like '.99' pricing common in legitimate retail.
        """
        amt = X["TransactionAmt"].astype(np.float32)
        return {
            "amt": amt,
            "amt_log": np.log1p(amt).astype(np.float32),
            "amt_decimal": (amt - amt.astype(int)).astype(np.float32),
            "amt_is_round": (amt == amt.astype(int)).astype(np.int8),
        }
    

    # ---- Group 3: UID-based velocity and aggregation features ----

    def _fit_uid_amount_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """Learn mean and std of TransactionAmt per UID from training."""
        stats = (
            X.groupby("UID")["TransactionAmt"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "uid_amt_mean", "std": "uid_amt_std"})
        )
        return stats
    
    @staticmethod
    def _fit_categorical_categories(X: pd.DataFrame) -> dict:
        """Learn the sorted category list per categorical column from training.

        At transform time, these exact category lists are applied so that train,
        validation, and production inference all share identical encodings.
        Unseen categories at inference become NaN, which LightGBM handles natively.
        """
        out = {}
        for col in CATEGORICAL_COLS:
            if col in X.columns:
                # Include "missing" so NaN fills always map cleanly
                cats = sorted(set(X[col].astype("object").fillna("missing").unique()))
                out[col] = cats
        return out

    def _make_uid_features(self, X: pd.DataFrame) -> dict:
        """Per-UID transaction count and amount deviation features.

        Insight from EDA: card1 alone is not a unique identifier. The composite UID
        (card1+card2+addr1+P_emaildomain) is our best proxy for unique cards/users.
        Deviation from the user's own historical mean amount is a strong fraud signal.
        """
        # Transaction count per UID (looked up from training; unknown UIDs get 0)
        uid_count = X["UID"].map(self.uid_count_map_).fillna(0).astype(np.int32)

        # Amount stats per UID
        stats = X[["UID"]].merge(
            self.uid_amt_stats_, left_on="UID", right_index=True, how="left"
        )
        uid_amt_mean = stats["uid_amt_mean"].astype(np.float32)
        uid_amt_std = stats["uid_amt_std"].astype(np.float32)

        # Deviation from user's mean amount, scaled by user's std
        # NaN-safe: if std is 0 or NaN, deviation is 0 (no signal)
        std_safe = uid_amt_std.replace(0, np.nan)
        amt_deviation = ((X["TransactionAmt"] - uid_amt_mean) / std_safe).fillna(0).astype(np.float32)

        return {
            "uid_txn_count": uid_count,
            "uid_amt_mean": uid_amt_mean,
            "uid_amt_std": uid_amt_std.fillna(0).astype(np.float32),
            "uid_amt_deviation": amt_deviation,
            "uid_is_new": (uid_count == 0).astype(np.int8),
        }

    # ---- Group 4: Email domain risk (target-encoded with smoothing) ----

    def _fit_email_risk(self, X: pd.DataFrame) -> dict:
        """Bayesian-smoothed fraud rate per email domain.

        Smoothed estimate: (n * domain_rate + m * global_rate) / (n + m)
        For domains with few transactions, this shrinks toward the global rate,
        preventing overfitting to rare domains.
        """
        grouped = X.groupby("P_emaildomain")["isFraud"].agg(["sum", "count"])
        m = self.email_smoothing
        smoothed = (grouped["sum"] + m * self.global_fraud_rate_) / (grouped["count"] + m)
        return smoothed.to_dict()

    def _make_email_features(self, X: pd.DataFrame) -> dict:
        """Email domain risk score and free/corporate flags.

        Insight from EDA: mail.com has 19% fraud rate, earthlink.net has 2%.
        Target-encoded smoothed risk captures this directly.
        """
        free_domains = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
                        "aol.com", "mail.com", "live.com", "msn.com", "ymail.com"}

        domain = X["P_emaildomain"].fillna("missing")
        risk = domain.map(self.email_risk_map_).fillna(self.global_fraud_rate_).astype(np.float32)

        return {
            "email_risk": risk,
            "email_is_free": domain.isin(free_domains).astype(np.int8),
            "email_is_missing": (X["P_emaildomain"].isna()).astype(np.int8),
        }

    # ---- Group 5: Missing-data indicators ----

    @staticmethod
    def _make_missing_indicators(X: pd.DataFrame) -> dict:
        """Binary flags for high-signal sparse columns.

        EDA showed 214 columns are >50% missing. For identity columns specifically,
        absence-of-data is itself a signal (most transactions have no identity record).
        """
        # Just a few high-signal ones; full list goes too wide
        sparse_signal_cols = ["id_01", "id_02", "DeviceType", "DeviceInfo", "dist1"]
        return {
            f"missing_{col}": X[col].isna().astype(np.int8)
            for col in sparse_signal_cols
            if col in X.columns
        }

    # ---- Group 6: Pass-through features (LightGBM handles NaN natively) ----

    def _pass_through_features(self, X: pd.DataFrame) -> dict:
        """Raw numeric and categorical features that go straight to the model.

        Categoricals use the training category lists learned in fit(), so train/val/prod
        all produce identical encodings.
        """
        num_cols = ["C1", "C2", "C13", "D1", "D2", "D15", "addr1", "addr2"]

        out = {}
        # Categoricals — enforce training-time category lists
        for col, cats in self.cat_categories_.items():
            if col in X.columns:
                series = X[col].astype("object").fillna("missing")
                # Categories not seen in training become NaN
                out[col] = pd.Categorical(series, categories=cats)
        # Numerics
        for col in num_cols:
            if col in X.columns:
                out[col] = X[col].astype(np.float32)
        return out