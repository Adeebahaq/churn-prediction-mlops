"""Compare baseline and main model, register the best model in MLflow."""
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

from src.utils import load_params, ensure_dir, load_env, fix_stdout_encoding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def setup_mlflow(params: dict):
    fix_stdout_encoding()
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


def compare_and_register(models_dir: str, processed_dir: str, cfg: dict) -> dict:
    models_dir = Path(models_dir)
    processed_dir = Path(processed_dir)

    with open(models_dir / "baseline_metrics.json", "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)
    with open(models_dir / "main_metrics.json", "r", encoding="utf-8") as f:
        main_metrics = json.load(f)

    metric = cfg["mlflow"]["metric_for_selection"]
    baseline_score = baseline_metrics[metric]
    main_score = main_metrics[metric]

    if main_score >= baseline_score:
        best_name = "main_random_forest"
        best_metrics = main_metrics
        best_model_path = models_dir / "main_model.joblib"
        best_stage = "Staging"
    else:
        best_name = "baseline_logistic_regression"
        best_metrics = baseline_metrics
        best_model_path = models_dir / "baseline_model.joblib"
        best_stage = "Staging"

    logger.info("Best model by %s: %s (%.4f)", metric, best_name, best_metrics[metric])

    with mlflow.start_run(run_name="model_selection"):
        mlflow.log_metrics({
            "baseline_roc_auc": baseline_metrics["roc_auc"],
            "main_roc_auc": main_metrics["roc_auc"],
            "best_roc_auc": best_metrics["roc_auc"],
        })
        mlflow.log_param("best_model", best_name)
        mlflow.log_param("selection_metric", metric)

        # Load and log the chosen model as an MLflow model
        model = joblib.load(best_model_path)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=cfg["mlflow"]["registered_model_name"],
        )

    # Transition latest version to Staging/Production
    client = mlflow.tracking.MlflowClient()
    try:
        latest_versions = client.get_latest_versions(
            cfg["mlflow"]["registered_model_name"], stages=["None"]
        )
        if latest_versions:
            client.transition_model_version_stage(
                name=cfg["mlflow"]["registered_model_name"],
                version=latest_versions[0].version,
                stage=best_stage,
            )
            logger.info("Registered model transitioned to %s", best_stage)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not transition model stage: %s", exc)

    comparison = {
        "baseline": baseline_metrics,
        "main": main_metrics,
        "best_model": best_name,
        "selection_metric": metric,
        "best_score": best_metrics[metric],
        "registered_stage": best_stage,
    }

    output_path = Path("metrics") / "model_comparison.json"
    ensure_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate and register best model")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    setup_mlflow(params)
    compare_and_register(
        "models",
        params["data"]["processed_dir"],
        params,
    )


if __name__ == "__main__":
    main()
