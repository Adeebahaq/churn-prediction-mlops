"""Preprocess raw Telco churn data into train/test splits and artifacts."""
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import load_params, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def preprocess(raw_path: str, output_dir: str, cfg: dict) -> dict:
    df = pd.read_csv(raw_path)
    target_col = cfg["target_column"]
    random_state = cfg["random_state"]
    test_size = cfg["test_size"]
    prep_cfg = cfg["preprocessing"]

    # Clean target
    df[target_col] = df[target_col].str.lower().map({"yes": 1, "no": 0})

    # Drop customer ID
    if prep_cfg["drop_customer_id"]:
        df = df.drop(columns=["customerID"])

    # Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # SeniorCitizen is already 0/1 numeric; ensure type
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Separate features / target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train.npy", X_train_processed)
    np.save(output_dir / "X_test.npy", X_test_processed)
    np.save(output_dir / "y_train.npy", y_train.values)
    np.save(output_dir / "y_test.npy", y_test.values)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    # Save feature names for interpretability / serving
    feature_names = numeric_features + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features))
    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(list(feature_names), f)

    logger.info("Processed data saved to %s", output_dir)
    logger.info("Train shape: %s | Test shape: %s", X_train_processed.shape, X_test_processed.shape)

    return {
        "train_shape": X_train_processed.shape,
        "test_shape": X_test_processed.shape,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "n_features": len(feature_names),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess Telco churn data")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--output", default="data/processed/preprocess_metadata.json")
    args = parser.parse_args()

    params = load_params(args.params)
    metadata = preprocess(
        params["data"]["raw_path"],
        params["data"]["processed_dir"],
        {**params["base"], "preprocessing": params["preprocessing"]},
    )

    ensure_dir(args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
