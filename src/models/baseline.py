"""Train and evaluate the baseline Logistic Regression model."""
import argparse
import json
import logging
import os
from pathlib import Path

import dagshub
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import load_params, ensure_dir, load_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def setup_mlflow(params: dict):
    load_env()
    mlflow_cfg = params.get("mlflow", {})
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        try:
            dagshub.init(
                repo_name=os.getenv("DAGSHUB_REPO_NAME"),
                repo_owner=os.getenv("DAGSHUB_REPO_OWNER"),
                mlflow=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialize DagsHub MLflow automatically: %s", exc)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "telco_churn_experiment"))


def train_and_evaluate(processed_dir: str, output_dir: str, cfg: dict) -> dict:
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(processed_dir / "X_train.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    model_cfg = cfg["models"]["logistic_regression"]
    base_cfg = cfg["base"]

    with mlflow.start_run(run_name=model_cfg["name"]):
        mlflow.log_params({
            "model": "LogisticRegression",
            "max_iter": model_cfg["max_iter"],
            "class_weight": model_cfg["class_weight"],
            "solver": model_cfg["solver"],
            "random_state": base_cfg["random_state"],
            "test_size": base_cfg["test_size"],
        })

        model = LogisticRegression(
            max_iter=model_cfg["max_iter"],
            class_weight=model_cfg["class_weight"],
            solver=model_cfg["solver"],
            random_state=base_cfg["random_state"],
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        }

        mlflow.log_metrics(metrics)
        model_path = output_dir / "baseline_model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.log_artifact(str(processed_dir / "feature_names.json"), artifact_path="features")

        metrics_path = output_dir / "baseline_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Baseline metrics: %s", metrics)
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Train baseline Logistic Regression")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    setup_mlflow(params)
    train_and_evaluate(
        params["data"]["processed_dir"],
        "models",
        params,
    )


if __name__ == "__main__":
    main()
