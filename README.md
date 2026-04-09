# MLOps Pipeline — End-to-End ML Model Training & Deployment

> **Resume Line:** Built an end-to-end ML pipeline with MLflow for experiment tracking, DVC for data versioning, Docker for containerization, and Jenkins + Kubernetes for automated CI/CD deployment.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                               │
│                                                                     │
│  Git Push / Cron                                                    │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐    │
│  │   Jenkins   │───▶│  DVC Repro   │───▶│   MLflow Tracking    │   │
│  │  CI/CD      │    │  Pipeline    │    │   Server             │    │
│  └─────────────┘    └──────┬───────┘    └──────────────────────┘    │ 
│                            │                                        │
│              ┌─────────────┼─────────────┐                          │
│              ▼             ▼             ▼                          │ 
│       data_ingestion  preprocess      train ──▶ evaluate           │
│              │             │             │          │               │
│              ▼             ▼             ▼          ▼               │
│          raw CSV    processed CSVs  model.joblib  metrics.json      │
│              │             │             │          │               │
│              └─────────────┴──── DVC ────┘          │               │
│                                                     │               │
│                              Quality Gate ◀─────────┘               │
│                                   │                                 │
│                                   ▼                                 │
│                        ┌──────────────────┐                         │
│                        │  Docker Build    │                         │
│                        │  & Push to Reg   │                         │
│                        └────────┬─────────┘                         │
│                                 │                                   │
│                                 ▼                                   │
│                        ┌──────────────────┐                         │
│                        │  Kubernetes      │                         │
│                        │  Rolling Deploy  │                         │
│                        │  (HPA enabled)   │                         │
│                        └────────┬─────────┘                         │
│                                 │                                   │
│                                 ▼                                   │
│                        ┌──────────────────┐                         │
│                        │  FastAPI         │                         │
│                        │  Inference API   │                         │
│                        │  /predict        │                         │
│                        └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
mlops-pipeline/
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py        # Raw data fetch / generation
│   ├── data_preprocessing.py   # Feature engineering, scaling, encoding
│   ├── train.py                 # Model training with MLflow logging
│   └── evaluate.py              # Test-set evaluation + quality gate
│
├── api/
│   ├── app.py                   # FastAPI inference service
│   └── requirements.txt         # API-specific dependencies
│
├── docker/
│   ├── Dockerfile.train         # Training container image
│   └── Dockerfile.api           # Multi-stage production API image
│
├── jenkins/
│   └── Jenkinsfile              # CI/CD pipeline (9 stages)
│
├── k8s/
│   ├── deployment.yaml          # Kubernetes Deployment (rolling update)
│   └── service.yaml             # Service + HPA + ConfigMap + Ingress
│
├── tests/
│   ├── test_pipeline.py         # Pytest unit tests (ingestion, prep, model)
│   └── smoke_test.py            # Post-deploy smoke tests
│
├── data/
│   ├── raw/                     # Raw CSV (DVC tracked)
│   └── processed/               # Processed features (DVC tracked)
│
├── models/                      # Serialized model artifacts (DVC tracked)
├── metrics/                     # DVC metrics JSON files
│
├── dvc.yaml                     # DVC pipeline DAG definition
├── params.yaml                  # All pipeline hyperparameters
├── docker-compose.yml           # Local dev stack
└── requirements.txt             # Training dependencies
```

---

## Tech Stack

| Tool | Role |
|---|---|
| **Python 3.11** | Core language |
| **Scikit-learn** | Model training (RandomForestClassifier) |
| **MLflow** | Experiment tracking, model registry |
| **DVC** | Data & model versioning, pipeline DAG |
| **FastAPI** | Model inference REST API |
| **Docker** | Container images (multi-stage build) |
| **Jenkins** | CI/CD automation (9-stage pipeline) |
| **Kubernetes** | Orchestration, autoscaling (HPA) |

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.11+, Docker, kubectl, DVC
pip install -r requirements.txt
```

### 2. Initialize DVC

```bash
git init
dvc init
dvc remote add -d myremote s3://your-bucket/dvc-cache
# or for local testing:
dvc remote add -d myremote /tmp/dvc-remote
```

### 3. Run Full Training Pipeline

```bash
# Run all pipeline stages (ingestion → preprocess → train → evaluate)
dvc repro

# View metrics
dvc metrics show

# Compare runs
dvc metrics diff HEAD~1
```

### 4. Start Local Dev Stack (Docker Compose)

```bash
# Start MLflow server + Inference API
docker-compose up -d

# Run full training pipeline inside Docker
docker-compose --profile train up trainer

# API is live at http://localhost:8000
# MLflow UI at http://localhost:5000
```

### 5. Test the Inference API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "feature_1": 1.23,
    "feature_2": -0.45,
    "feature_3": 0.78,
    "feature_4": 2.10,
    "feature_5": -1.05,
    "category_1": "A",
    "category_2": "B"
  }'

# Response:
# {
#   "prediction": 1,
#   "prediction_label": "1",
#   "probabilities": [0.12, 0.74, 0.14],
#   "model_version": "Production"
# }

# Interactive API docs
open http://localhost:8000/docs
```

### 6. Run Tests

```bash
# Unit tests with coverage
pytest tests/test_pipeline.py -v --cov=src --cov-report=term-missing

# Post-deploy smoke tests (requires running API)
API_URL=http://localhost:8000 python tests/smoke_test.py
```

---

## DVC Pipeline DAG

```
data_ingestion
      │
      ▼
data_preprocessing
      │
      ▼
    train
      │
      ▼
  evaluate ──▶ metrics/scores.json
```

```bash
# Visualize the pipeline
dvc dag

# Reproduce only changed stages
dvc repro

# Reproduce all stages regardless of cache
dvc repro --force
```

---

## MLflow Experiment Tracking

Each training run logs:

| Category | What's Logged |
|---|---|
| **Parameters** | n_estimators, max_depth, cv_folds, model_type, data shape |
| **Metrics** | CV F1, train/test accuracy, F1, precision, recall |
| **Artifacts** | model.joblib, feature_importances.json, classification_report.json |
| **Model** | Registered in MLflow Model Registry under `production-classifier` |

```bash
# Start MLflow UI locally
mlflow ui --port 5000

# View tracked experiments at http://localhost:5000
```

---

## Jenkins CI/CD Pipeline Stages

| Stage | Description |
|---|---|
| 1. Checkout | Clone repo, log commit SHA |
| 2. Install Dependencies | pip install requirements |
| 3. Lint & Test | flake8 + pytest with coverage (parallel) |
| 4. DVC Pull | Pull data/model cache from remote |
| 5. Train Pipeline | `dvc repro` — full pipeline execution |
| 6. Quality Gate | Block deploy if `test_f1_weighted < 0.80` |
| 7. Build & Push | Multi-stage Docker build → push to registry |
| 8. Deploy to K8s | `kubectl set image` + rollout status watch |
| 9. Smoke Test | POST /predict + /health validation |

Triggers: `git push` to `main` OR daily cron at 02:00.

---

## Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -f k8s/

# Watch rollout
kubectl rollout status deployment/mlops-model-api -n mlops

# Check HPA (autoscaling)
kubectl get hpa -n mlops

# Scale manually
kubectl scale deployment mlops-model-api --replicas=4 -n mlops

# Port-forward for local testing
kubectl port-forward svc/mlops-model-api 8000:80 -n mlops
```

**HPA Config:** Scales between 2–10 pods based on CPU (70%) and memory (80%) utilization.

---

## Configuration

All pipeline configuration lives in `params.yaml`:

```yaml
model:
  params:
    n_estimators: 100    # ← change and re-run dvc repro
    max_depth: 10

evaluate:
  threshold:
    f1_weighted: 0.80    # ← quality gate threshold
```

DVC tracks parameter changes and only re-runs affected downstream stages.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check + model status |
| POST | `/predict` | Single inference |
| POST | `/predict/batch` | Batch inference |
| GET | `/model/info` | Model metadata + params |
| GET | `/docs` | Swagger UI |

---

## Resume Bullets (Copy-Paste Ready)

```
• Built an end-to-end ML pipeline with MLflow for experiment tracking,
  logging parameters, metrics, and model artifacts across every run.

• Implemented data and model versioning using DVC with a 4-stage pipeline
  DAG (ingest → preprocess → train → evaluate), enabling reproducible runs.

• Containerized the model inference service using a multi-stage Docker build,
  reducing image size and separating build from runtime dependencies.

• Automated training and deployment using a 9-stage Jenkins CI/CD pipeline
  with quality gates, DVC cache, and Kubernetes rolling updates (HPA: 2–10 pods).
```
