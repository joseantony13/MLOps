"""
data_ingestion.py - Fetches or generates raw dataset for the pipeline.
In production, replace generate_synthetic_data() with your actual data source
(e.g., S3, database, API).
"""

import os
import logging
import yaml
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def generate_synthetic_data(params: dict) -> pd.DataFrame:
    """
    Generates a synthetic classification dataset.
    Replace this with real data ingestion logic (S3, DB, REST API, etc.)
    """
    logger.info("Generating synthetic classification dataset...")
    random_state = params["data"]["random_state"]

    X, y = make_classification(
        n_samples=5000,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_classes=3,
        random_state=random_state,
        weights=[0.5, 0.3, 0.2],
    )

    numerical_cols = params["features"]["numerical"]
    categorical_rng = np.random.RandomState(random_state)

    df = pd.DataFrame(X, columns=numerical_cols)
    for cat_col in params["features"]["categorical"]:
        df[cat_col] = categorical_rng.choice(["A", "B", "C"], size=len(df))

    df[params["data"]["target_column"]] = y
    return df


def ingest(params: dict) -> None:
    raw_path = params["data"]["raw_path"]
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    df = generate_synthetic_data(params)
    df.to_csv(raw_path, index=False)

    logger.info(f"Dataset saved → {raw_path}  |  Shape: {df.shape}")
    logger.info(f"Class distribution:\n{df[params['data']['target_column']].value_counts()}")


if __name__ == "__main__":
    params = load_params()
    ingest(params)
