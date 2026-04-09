"""
predict.py - Standalone CLI inference script.

Loads the trained model + preprocessor and runs predictions on:
  - A single JSON input via --input flag
  - A CSV batch file via --batch flag
  - An interactive prompt via --interactive flag

Usage:
    python src/predict.py --input '{"feature_1": 1.2, ..., "category_1": "A"}'
    python src/predict.py --batch data/new_samples.csv --output predictions.csv
    python src/predict.py --interactive
    python src/predict.py --mlflow --model-name production-classifier --stage Production
"""

import os
import sys
import json
import argparse
import logging
import yaml
import joblib
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def load_local_artifacts(params: dict):
    """Load model and preprocessor from local disk (DVC-tracked artifacts)."""
    model_path = params["training"]["model_output_path"]
    preprocessor_path = os.path.join(
        params["data"]["processed_path"], "preprocessor.joblib"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Run 'dvc repro' first."
        )
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"Preprocessor not found at '{preprocessor_path}'. Run 'dvc repro' first."
        )

    model = joblib.load(model_path)
    artifact = joblib.load(preprocessor_path)
    preprocessor = artifact["preprocessor"]
    label_encoder = artifact["label_encoder"]

    logger.info(f"Loaded model from       : {model_path}")
    logger.info(f"Loaded preprocessor from: {preprocessor_path}")
    return model, preprocessor, label_encoder


def load_mlflow_artifacts(model_name: str, stage: str, tracking_uri: str):
    """Load model from MLflow Model Registry."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        raise ImportError("mlflow is required for --mlflow mode. pip install mlflow")

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/{stage}"
    logger.info(f"Loading model from MLflow registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    return model


# ─── Core Prediction Logic ────────────────────────────────────────────────────

def build_input_df(raw_input: dict, params: dict) -> pd.DataFrame:
    """Validates and builds a single-row DataFrame from a dict of raw features."""
    numerical = params["features"]["numerical"]
    categorical = params["features"]["categorical"]
    expected_cols = numerical + categorical

    missing = [c for c in expected_cols if c not in raw_input]
    if missing:
        raise ValueError(
            f"Missing required features: {missing}\n"
            f"Expected: {expected_cols}"
        )

    return pd.DataFrame([{col: raw_input[col] for col in expected_cols}])


def predict_single(
    raw_input: dict,
    model,
    preprocessor,
    label_encoder,
    params: dict,
) -> dict:
    """Run inference on a single sample dict. Returns a result dict."""
    df = build_input_df(raw_input, params)

    X = preprocessor.transform(df)
    try:
        feature_names = preprocessor.get_feature_names_out()
        X = pd.DataFrame(X, columns=feature_names)
    except AttributeError:
        pass

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0].tolist()

    label = None
    if label_encoder is not None:
        label = str(label_encoder.inverse_transform([prediction])[0])

    return {
        "prediction": prediction,
        "prediction_label": label,
        "probabilities": {f"class_{i}": round(p, 4) for i, p in enumerate(probabilities)},
        "confidence": round(max(probabilities), 4),
    }


def predict_batch(
    csv_path: str,
    model,
    preprocessor,
    label_encoder,
    params: dict,
    output_path: str = None,
) -> pd.DataFrame:
    """Run inference on a CSV batch file. Returns DataFrame with predictions appended."""
    logger.info(f"Loading batch file: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Batch size: {len(df)} rows")

    numerical = params["features"]["numerical"]
    categorical = params["features"]["categorical"]
    feature_cols = numerical + categorical

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Batch CSV is missing columns: {missing}")

    X_raw = df[feature_cols]
    X = preprocessor.transform(X_raw)
    try:
        feature_names = preprocessor.get_feature_names_out()
        X = pd.DataFrame(X, columns=feature_names)
    except AttributeError:
        pass

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    df["predicted_class"] = predictions
    df["confidence"] = np.max(probabilities, axis=1).round(4)

    if label_encoder is not None:
        df["predicted_label"] = label_encoder.inverse_transform(predictions)

    for i in range(probabilities.shape[1]):
        df[f"prob_class_{i}"] = probabilities[:, i].round(4)

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to: {output_path}")

    return df


# ─── Interactive Mode ─────────────────────────────────────────────────────────

def interactive_mode(model, preprocessor, label_encoder, params: dict):
    """REPL-style interactive prediction prompt."""
    numerical = params["features"]["numerical"]
    categorical = params["features"]["categorical"]

    print("\n=== MLOps Pipeline — Interactive Predictor ===")
    print("Type 'quit' at any prompt to exit.\n")

    while True:
        raw_input = {}
        print("Enter feature values:")

        try:
            for col in numerical:
                val = input(f"  {col} (float): ").strip()
                if val.lower() == "quit":
                    print("Exiting.")
                    return
                raw_input[col] = float(val)

            for col in categorical:
                val = input(f"  {col} (string): ").strip()
                if val.lower() == "quit":
                    print("Exiting.")
                    return
                raw_input[col] = val

            result = predict_single(raw_input, model, preprocessor, label_encoder, params)
            print("\n--- Prediction Result ---")
            print(json.dumps(result, indent=2))
            print()

        except ValueError as e:
            print(f"[ERROR] Invalid input: {e}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            return


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="MLOps Pipeline — Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction from JSON string
  python src/predict.py --input '{"feature_1":1.2,"feature_2":-0.4,"feature_3":0.8,"feature_4":2.1,"feature_5":-1.0,"category_1":"A","category_2":"B"}'

  # Batch prediction from CSV
  python src/predict.py --batch data/new_samples.csv --output predictions.csv

  # Interactive REPL
  python src/predict.py --interactive

  # Load model from MLflow registry instead of local disk
  python src/predict.py --interactive --mlflow --model-name production-classifier --stage Production
        """,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=str, help="JSON string of feature values for single prediction")
    source.add_argument("--batch", type=str, help="Path to CSV file for batch prediction")
    source.add_argument("--interactive", action="store_true", help="Launch interactive REPL")

    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path for batch predictions (default: print to stdout)")
    parser.add_argument("--params", type=str, default="params.yaml",
                        help="Path to params.yaml (default: params.yaml)")

    # MLflow model source
    mlflow_group = parser.add_argument_group("MLflow options")
    mlflow_group.add_argument("--mlflow", action="store_true",
                              help="Load model from MLflow registry instead of local disk")
    mlflow_group.add_argument("--model-name", type=str, default="production-classifier",
                              help="Registered model name in MLflow (default: production-classifier)")
    mlflow_group.add_argument("--stage", type=str, default="Production",
                              help="Model stage in MLflow registry (default: Production)")
    mlflow_group.add_argument("--tracking-uri", type=str,
                              default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                              help="MLflow tracking URI")

    return parser.parse_args()


def main():
    args = parse_args()
    params = load_params(args.params)

    # ── Load artifacts ────────────────────────────────────────────────────────
    label_encoder = None
    preprocessor = None

    if args.mlflow:
        model = load_mlflow_artifacts(args.model_name, args.stage, args.tracking_uri)
        # Still load preprocessor from local disk
        artifact = joblib.load(
            os.path.join(params["data"]["processed_path"], "preprocessor.joblib")
        )
        preprocessor = artifact["preprocessor"]
        label_encoder = artifact["label_encoder"]
    else:
        model, preprocessor, label_encoder = load_local_artifacts(params)

    # ── Run inference ─────────────────────────────────────────────────────────
    if args.input:
        try:
            raw_input = json.loads(args.input)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            sys.exit(1)

        result = predict_single(raw_input, model, preprocessor, label_encoder, params)
        print(json.dumps(result, indent=2))

    elif args.batch:
        result_df = predict_batch(
            args.batch, model, preprocessor, label_encoder, params, args.output
        )
        if not args.output:
            print(result_df[["predicted_class", "predicted_label", "confidence"]].to_string())

    elif args.interactive:
        interactive_mode(model, preprocessor, label_encoder, params)


if __name__ == "__main__":
    main()
