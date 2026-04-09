"""
train.py - Trains the ML model, logs all params/metrics/artifacts to MLflow,
and registers the best model in the MLflow Model Registry.
"""

import os
import logging
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
from mlflow.models.signature import infer_signature

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def load_data(processed_dir: str):
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    return X_train, y_train


def build_model(model_params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**model_params)


def train(params: dict) -> None:
    processed_dir = params["data"]["processed_path"]
    model_output = params["training"]["model_output_path"]
    cv_folds = params["training"]["cv_folds"]
    scoring = params["training"]["scoring"]
    mlflow_cfg = params["mlflow"]
    model_params = params["model"]["params"]

    os.makedirs(os.path.dirname(model_output), exist_ok=True)

    # --- Configure MLflow ---
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    logger.info("Loading processed training data...")
    X_train, y_train = load_data(processed_dir)
    logger.info(f"Training data shape: {X_train.shape}")

    with mlflow.start_run(run_name="rf-training") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # --- Log all parameters ---
        mlflow.log_params(model_params)
        mlflow.log_params({
            "cv_folds": cv_folds,
            "scoring": scoring,
            "model_type": params["model"]["type"],
            "train_samples": X_train.shape[0],
            "n_features": X_train.shape[1],
        })

        # --- Cross-validation ---
        logger.info(f"Running {cv_folds}-fold cross-validation...")
        model = build_model(model_params)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)

        mlflow.log_metric("cv_mean_f1", float(np.mean(cv_scores)))
        mlflow.log_metric("cv_std_f1", float(np.std(cv_scores)))
        logger.info(f"CV {scoring}: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

        # --- Final model fit on full training set ---
        logger.info("Fitting final model on full training set...")
        model.fit(X_train, y_train)

        # --- Training-set metrics ---
        y_pred_train = model.predict(X_train)
        train_metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "train_f1_weighted": f1_score(y_train, y_pred_train, average="weighted"),
            "train_precision_weighted": precision_score(y_train, y_pred_train, average="weighted"),
            "train_recall_weighted": recall_score(y_train, y_pred_train, average="weighted"),
        }
        mlflow.log_metrics(train_metrics)

        # --- Log feature importances ---
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        sorted_fi = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        mlflow.log_dict(sorted_fi, "feature_importances.json")

        # --- Log model with signature ---
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=mlflow_cfg["registered_model_name"],
            input_example=X_train.head(5),
        )

        # --- Save model locally for DVC tracking ---
        joblib.dump(model, model_output)
        mlflow.log_artifact(model_output)

        # --- Log classification report ---
        report = classification_report(y_train, y_pred_train, output_dict=True)
        mlflow.log_dict(report, "classification_report_train.json")

        # --- Save run ID for downstream stages ---
        os.makedirs("models", exist_ok=True)
        with open("models/run_id.txt", "w") as f:
            f.write(run_id)
        mlflow.log_artifact("models/run_id.txt")

        logger.info(f"Model saved -> {model_output}")


if __name__ == "__main__":
    params = load_params()
    train(params)
