"""
app.py - FastAPI model inference service.
Loads the registered MLflow model and serves predictions via REST API.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "production-classifier")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")          # fallback local path
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "data/processed/preprocessor.joblib")

# ─── Globals ─────────────────────────────────────────────────────────────────
model = None
preprocessor = None
label_encoder = None
model_metadata: dict = {}


# ─── Schemas ─────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    feature_1: float = Field(..., example=1.23)
    feature_2: float = Field(..., example=-0.45)
    feature_3: float = Field(..., example=0.78)
    feature_4: float = Field(..., example=2.10)
    feature_5: float = Field(..., example=-1.05)
    category_1: str = Field(..., example="A")
    category_2: str = Field(..., example="B")


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: Optional[str]
    probabilities: List[float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_stage: str


# ─── Lifecycle ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor, label_encoder, model_metadata
    logger.info("Starting up: loading model and preprocessor...")

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri)
        model_metadata = {
            "name": REGISTERED_MODEL_NAME,
            "stage": MODEL_STAGE,
            "source": "mlflow_registry",
        }
        logger.info(f"Loaded model from MLflow registry: {model_uri}")
    except Exception as e:
        logger.warning(f"MLflow registry load failed ({e}), falling back to local model.")
        model = joblib.load(MODEL_PATH)
        model_metadata = {"name": "local", "stage": "local", "source": MODEL_PATH}

    try:
        artifact = joblib.load(PREPROCESSOR_PATH)
        preprocessor = artifact["preprocessor"]
        label_encoder = artifact["label_encoder"]
        logger.info("Preprocessor loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load preprocessor: {e}")

    yield

    logger.info("Shutting down inference service.")


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MLOps Model Inference API",
    description="End-to-end MLOps pipeline inference endpoint backed by MLflow.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_name=model_metadata.get("name", "unknown"),
        model_stage=model_metadata.get("stage", "unknown"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    try:
        # Build raw input DataFrame (matches original feature schema)
        raw_input = pd.DataFrame([{
            "feature_1": request.feature_1,
            "feature_2": request.feature_2,
            "feature_3": request.feature_3,
            "feature_4": request.feature_4,
            "feature_5": request.feature_5,
            "category_1": request.category_1,
            "category_2": request.category_2,
        }])

        if preprocessor is not None:
            X = preprocessor.transform(raw_input)
            X = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())
        else:
            X = raw_input  # skip preprocessing if unavailable

        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0].tolist()

        # Decode label if encoder is available
        label = None
        if label_encoder is not None:
            label = str(label_encoder.inverse_transform([prediction])[0])

        return PredictionResponse(
            prediction=prediction,
            prediction_label=label,
            probabilities=probabilities,
            model_version=model_metadata.get("stage", "unknown"),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction endpoint — accepts a list of feature sets."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for req in requests:
        result = await predict(req)
        results.append(result)
    return {"predictions": results, "count": len(results)}


@app.get("/model/info", tags=["System"])
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    info = {
        **model_metadata,
        "model_type": type(model).__name__,
    }
    if hasattr(model, "get_params"):
        info["model_params"] = model.get_params()
    if hasattr(model, "feature_importances_"):
        info["n_features"] = len(model.feature_importances_)
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
