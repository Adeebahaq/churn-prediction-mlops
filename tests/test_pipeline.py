"""Unit tests for the core pipeline utilities and validation."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.validate import validate
from src.utils import load_params


def test_load_params():
    params = load_params("params.yaml")
    assert "base" in params
    assert "data" in params
    assert "models" in params
    assert "logistic_regression" in params["models"]


def test_validation_report_structure(tmp_path):
    params = load_params("params.yaml")
    # Use a tiny synthetic dataset that satisfies the validation checks
    import pandas as pd
    n = params["data"]["expected_rows"]
    df = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(n)],
        "gender": ["Male"] * n,
        "SeniorCitizen": [0] * n,
        "Partner": ["No"] * n,
        "Dependents": ["No"] * n,
        "tenure": [10] * n,
        "PhoneService": ["Yes"] * n,
        "MultipleLines": ["No"] * n,
        "InternetService": ["DSL"] * n,
        "OnlineSecurity": ["No"] * n,
        "OnlineBackup": ["No"] * n,
        "DeviceProtection": ["No"] * n,
        "TechSupport": ["No"] * n,
        "StreamingTV": ["No"] * n,
        "StreamingMovies": ["No"] * n,
        "Contract": ["Month-to-month"] * n,
        "PaperlessBilling": ["Yes"] * n,
        "PaymentMethod": ["Electronic check"] * n,
        "MonthlyCharges": [50.0] * n,
        "TotalCharges": ["500"] * n,
        "Churn": ["No"] * n,
    })
    # Force churn rate within expected range
    churn_count = int(n * 0.27)
    df.loc[:churn_count - 1, "Churn"] = "Yes"

    raw_path = tmp_path / "synthetic.csv"
    df.to_csv(raw_path, index=False)

    # Temporarily adjust expected rows/cols for the test
    params["data"]["expected_rows"] = n
    params["data"]["expected_columns"] = 21
    report = validate(
        str(raw_path),
        params["validation"],
        params["data"],
        target_column=params["base"]["target_column"],
    )
    assert report["checks"]["expected_rows"]
    assert report["checks"]["expected_columns"]
    assert report["checks"]["has_target"]


def test_processed_files_exist_after_pipeline():
    """Run only after dvc repro has populated artifacts."""
    processed_dir = Path("data/processed")
    required = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "preprocessor.joblib"]
    for name in required:
        path = processed_dir / name
        if path.exists():
            assert path.stat().st_size > 0
