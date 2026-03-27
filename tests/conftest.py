"""
conftest.py
-----------
Shared pytest fixtures — synthetic DataFrames and arrays that stand in
for the real (gitignored) datasets. All tests run without any downloaded data.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ibm_df():
    """200-row synthetic IBM-like transactions DataFrame."""
    n = 200
    rng = np.random.default_rng(42)
    accounts = [f"ACC{i:04d}" for i in range(50)]
    return pd.DataFrame({
        "Timestamp":          pd.date_range("2022-01-01", periods=n, freq="h"),
        "From Bank":          rng.integers(1, 10, n).astype(str),
        "Account":            rng.choice(accounts, n),
        "To Bank":            rng.integers(1, 10, n).astype(str),
        "Account.1":          rng.choice(accounts, n),
        "Amount Paid":        rng.exponential(1000, n),
        "Amount Received":    rng.exponential(1000, n),
        "Payment Currency":   rng.choice(["US Dollar", "Euro", "Yuan"], n),
        "Receiving Currency": rng.choice(["US Dollar", "Euro", "Yuan"], n),
        "Payment Format":     rng.choice(["Cheque", "Wire", "Credit Card", "Reinvestment"], n),
        "Is Laundering":      rng.choice([0, 1], n, p=[0.92, 0.08]),
    })


@pytest.fixture
def czech_tables():
    """Minimal Czech trans + loan tables mimicking the real schema."""
    n = 100
    rng = np.random.default_rng(0)
    trans = pd.DataFrame({
        "account_id": np.arange(n),
        "date":       rng.integers(930101, 981231, n),   # YYMMDD int format
        "amount":     rng.exponential(5000, n),
        "type":       rng.choice(["PRIJEM", "VYDAJ"], n),
        "k_symbol":   [None if i % 5 == 0 else "SIPO" for i in range(n)],
        "partner":    [f"PARTNER{i % 20}" for i in range(n)],
    })
    loan = pd.DataFrame({
        "account_id": np.arange(20),
        "status":     rng.choice(["A", "B", "C", "D"], 20),
        "amount":     rng.exponential(10_000, 20),
    })
    return {"trans": trans, "loan": loan}


@pytest.fixture
def X_y():
    """300-sample feature matrix with imbalanced binary labels (5% fraud)."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((300, 8)).astype(np.float32)
    y = np.zeros(300, dtype=int)
    y[:15] = 1   # 5% fraud rate
    return X, y
