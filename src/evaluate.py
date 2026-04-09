"""
evaluate.py - Evaluates the trained model on the held-out test set,
logs metrics to MLflow, writes DVC metrics JSON, and enforces quality gate.
"""

import os
import json
import logging
import yaml
import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def evaluate(params: dict) -> None:
    processed_dir = params["data"]["processed_path"]
    model_path = params["training"]["model_output_path"]
    mlflow_cfg = params["mlflow"]
    threshold = params["evaluate"]["threshold"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    logger.info("Loading test data and model...")
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    model = joblib.load(model_path)

    run_id_path = "models/run_id.txt"
    run_id = None
    if os.path.exists(run_id_path):
        with open(run_id_path) as f:
            run_id = f.read().strip()

    y_pred = model.predict(X_test)
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "test_precision_weighted": float(precision_score(y_test, y_pred, average="weighted")),
        "test_recall_weighted": float(recall_score(y_test, y_pred, average="weighted")),
    }

    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    # --- DVC metrics file ---
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Log to MLflow ---
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_dict(report, "classification_report_test.json")
        cm = confusion_matrix(y_test, y_pred).tolist()
        mlflow.log_dict({"confusion_matrix": cm}, "confusion_matrix.json")

    # --- Quality gate ---
    for metric_name, min_value in threshold.items():
        key = f"test_{metric_name}"
        if key in metrics and metrics[key] < min_value:
            raise ValueError(
                f"Quality gate FAILED: {key}={metrics[key]:.4f} < threshold={min_value}. "
                "Model will NOT be deployed."
            )
    logger.info("Quality gate PASSED. Model is ready for deployment.")


if __name__ == "__main__":
    params = load_params()
    evaluate(params)
