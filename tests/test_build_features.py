"""Unit tests for FraudFeatureEngineer."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FraudFeatureEngineer


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small hand-crafted dataset that exercises every feature group."""
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6],
        "TransactionDT": [
            86400,        # 2017-12-02 00:00 — midnight
            86400 + 7*3600,  # 2017-12-02 07:00 — peak fraud hour
            86400 + 13*3600, # 2017-12-02 13:00 — safest hour
            86400 + 86400,   # 2017-12-03 00:00
            86400 + 86400 + 86400,  # 2017-12-04 00:00
            86400 + 7*3600,  # repeat of UID #1's profile
        ],
        "TransactionAmt": [50.0, 100.0, 75.50, 1000.0, 25.99, 60.0],
        "ProductCD": ["W", "C", "W", "C", "H", "W"],
        "card1": [1000, 1000, 2000, 1000, 3000, 1000],
        "card2": [200, 200, 300, 200, 400, 200],
        "card4": ["visa", "discover", "visa", "discover", "mastercard", "visa"],
        "card6": ["debit", "credit", "debit", "credit", "credit", "debit"],
        "addr1": [100, 100, 200, 100, 300, 100],
        "addr2": [87, 87, 87, 87, 87, 87],
        "P_emaildomain": ["gmail.com", "mail.com", "gmail.com", "mail.com", None, "gmail.com"],
        "DeviceType": ["desktop", "mobile", "desktop", "mobile", None, "desktop"],
        "DeviceInfo": ["Windows", "iOS", "Windows", None, None, "Windows"],
        "C1": [1.0, 2.0, 1.0, 5.0, 1.0, 1.0],
        "C2": [1.0, 2.0, 1.0, 5.0, 1.0, 1.0],
        "C13": [1.0, 2.0, 1.0, 5.0, 1.0, 1.0],
        "D1": [0.0, 0.0, 5.0, 0.0, np.nan, 0.0],
        "D2": [0.0, 0.0, 5.0, 0.0, np.nan, 0.0],
        "D15": [0.0, 0.0, 5.0, 0.0, np.nan, 0.0],
        "id_01": [np.nan, -5.0, np.nan, -10.0, np.nan, np.nan],
        "id_02": [np.nan, 70000.0, np.nan, 80000.0, np.nan, np.nan],
        "dist1": [10.0, np.nan, 20.0, np.nan, np.nan, 15.0],
    })
    y = pd.Series([0, 1, 0, 1, 0, 0])  # rows 2 and 4 are fraud
    return df, y


class TestFraudFeatureEngineer:
    def test_fit_returns_self(self, toy_data):
        X, y = toy_data
        fe = FraudFeatureEngineer()
        result = fe.fit(X, y)
        assert result is fe

    def test_fit_requires_y(self, toy_data):
        X, _ = toy_data
        fe = FraudFeatureEngineer()
        with pytest.raises(ValueError):
            fe.fit(X)

    def test_transform_before_fit_raises(self, toy_data):
        X, _ = toy_data
        fe = FraudFeatureEngineer()
        with pytest.raises(RuntimeError):
            fe.transform(X)

    def test_fit_transform_produces_dataframe(self, toy_data):
        X, y = toy_data
        fe = FraudFeatureEngineer()
        Xt = fe.fit(X, y).transform(X)
        assert isinstance(Xt, pd.DataFrame)
        assert len(Xt) == len(X)
        assert Xt.index.equals(X.index)

    def test_temporal_features_correct_hour(self, toy_data):
        X, y = toy_data
        fe = FraudFeatureEngineer().fit(X, y)
        Xt = fe.transform(X)
        # Row 1 was 7 AM, row 2 was 1 PM
        assert Xt.iloc[1]["hour"] == 7
        assert Xt.iloc[2]["hour"] == 13

    def test_email_risk_uses_global_rate_for_unseen_domain(self, toy_data):
        X, y = toy_data
        fe = FraudFeatureEngineer().fit(X, y)
        # New transaction with a never-seen email domain
        new_row = X.iloc[:1].copy()
        new_row["P_emaildomain"] = "nevr-seen-domain.com"
        Xt = fe.transform(new_row)
        # Should fall back to global fraud rate (1/3 in toy data — wait, 2/6 = 0.333)
        assert Xt["email_risk"].iloc[0] == pytest.approx(fe.global_fraud_rate_, rel=0.01)

    def test_uid_features_capture_repeats(self, toy_data):
        X, y = toy_data
        fe = FraudFeatureEngineer().fit(X, y)
        Xt = fe.transform(X)
        # Rows 0, 1, 3, 5 share UID (same card1, card2, addr1)
        # Wait — they have different P_emaildomains. Let's check actual repeats.
        # Rows 0 and 5 share UID exactly (gmail.com twice for same card/addr)
        uid_counts = Xt["uid_txn_count"]
        # The UID for row 0 should appear >= 2 times in training (rows 0 and 5)
        assert uid_counts.iloc[0] >= 2

    def test_no_nan_in_engineered_features(self, toy_data):
        """Engineered ratio features should not produce NaN (model expects clean numerics)."""
        X, y = toy_data
        fe = FraudFeatureEngineer().fit(X, y)
        Xt = fe.transform(X)
        for col in ["email_risk", "amt_log", "uid_amt_deviation", "hour_sin"]:
            assert Xt[col].isna().sum() == 0, f"{col} has NaN values"

    def test_output_columns_stable_across_calls(self, toy_data):
        """transform() should produce identical column set every time."""
        X, y = toy_data
        fe = FraudFeatureEngineer().fit(X, y)
        cols1 = fe.transform(X).columns.tolist()
        cols2 = fe.transform(X.iloc[:3]).columns.tolist()
        assert cols1 == cols2