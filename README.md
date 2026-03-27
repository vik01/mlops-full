[![CI](https://github.com/2026-IE-MLOps-Course/3-mlops-CICD/actions/workflows/ci.yml/badge.svg)](https://github.com/2026-IE-MLOps-Course/3-mlops-CICD/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![MLOps](https://img.shields.io/badge/MLOps-CI%2FCD-success)
![Docker](https://img.shields.io/badge/Docker-enabled-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Heart Disease Prediction MLOps Project

**Authors:** Vikram Bhatt, Teresa Alvarez, Alessandro Cristofolini, Matteo Khoueiri, Dominique Robson, & Pengchong Zhao
**Course:** MLOps: Master in Business Analytics and Data Science
**Repository:** `2026-IE-MLOps-Course/mlops-full`

## Overview

This repository demonstrates how to turn a notebook-first machine learning project into a modular, testable, reproducible, and deployable MLOps system.

The project predicts heart disease risk from structured clinical records. It includes:

- a modular `src/` pipeline
- centralized configuration in `config.yaml`
- unit tests with `pytest`
- experiment tracking and artifact logging with Weights & Biases
- a FastAPI inference service
- Docker packaging for serving
- Continuous Integration with GitHub Actions
- deployment to Render

## Business Objective

The goal is to identify patients with higher heart disease risk to support early screening and preventive intervention.

Healthcare professionals and screening platforms can submit patient clinical features and receive a heart disease risk prediction in real time via a web-accessible API endpoint.

This project is a decision-support example for teaching MLOps. It is not a clinical decision system, not a diagnostic tool, and not a substitute for medical judgment.

## Success Metrics

- **Business KPI:** Correctly identify ≥87% of positive cases (recall) while keeping the false positive rate below 12%, reducing both missed diagnoses and unnecessary referrals.
- **Technical Metrics:**
  - Logistic Regression test accuracy ≥ 88%
  - ROC-AUC ≥ 0.95
  - F1-score ≥ 0.86 on the test set
- **Acceptance Criteria:** The primary model (Logistic Regression) must outperform the Decision Tree baseline across precision, recall, specificity, and F1-score.

## What This Project Demonstrates

- moving from notebooks to a `src/` layout
- separating data loading, cleaning, validation, feature engineering, training, evaluation, and inference
- using one config file as the main source of non-secret runtime settings
- saving serialized model pipeline artifacts for consistent training and serving
- exposing model inference through a documented web API
- packaging the service in Docker
- validating quality with automated tests in Continuous Integration
- serving the API on Render
- logging training artifacts and model versions to Weights & Biases

## Project Architecture

The end-to-end workflow is:

1. Load raw data
2. Clean and standardize columns
3. Validate required columns and data quality constraints
4. Build features
5. Train classification pipelines (Logistic Regression, K-Means, Decision Tree)
6. Evaluate model quality
7. Save model artifacts
8. Serve the model through FastAPI locally or on Render

### Main Modules

- `src/load_data.py` loads raw data
- `src/clean_data.py` standardizes and cleans the dataframe
- `src/validate.py` applies fail-fast checks before expensive steps run
- `src/features.py` defines feature transformations
- `src/main.py` orchestrates the full pipeline
- `src/api.py` exposes the model as a FastAPI service
- `src/logger.py` centralizes logging setup
- `src/utils.py` shared utility helpers

## Repository Structure

```text
.
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
├── config.yaml
├── pytest.ini
├── .env                         # Secrets (never commit)
├── data/
│   ├── raw/
│   │   └── train.csv
│   ├── processed/
│   │   └── clean.csv
│   └── inference/
├── logs/
│   └── pipeline.log
├── models/
│   ├── kmeans_model.joblib
│   ├── logit_model.joblib
│   └── dtrees_model.joblib
├── notebooks/
│   └── Group_4_Heart_Disease_Prediction.ipynb
├── reports/
│   ├── kmeans_predictions.csv
│   ├── logit_predictions.csv
│   └── dtrees_predictions.csv
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── clean_data.py
│   ├── features.py
│   ├── load_data.py
│   ├── logger.py
│   ├── main.py
│   ├── utils.py
│   └── validate.py
└── tests/
    ├── test_clean_data.py
    ├── test_features.py
    ├── test_load_data.py
    ├── test_utils.py
    └── test_validate.py
```

## The Data

- **Source:** Synthetic heart disease dataset (630,000 observations, 13 clinical features).
- **Target Variable:** `Heart Disease` — binary classification (`Presence` / `Absence`).
- **Features:**

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Patient age |
| Sex | Categorical | Patient sex (binary) |
| Chest pain type | Categorical | Type of chest pain (1–4) |
| BP | Numeric | Resting blood pressure |
| Cholesterol | Numeric | Serum cholesterol |
| FBS over 120 | Categorical | Fasting blood sugar > 120 mg/dl (binary) |
| EKG results | Categorical | Resting electrocardiographic results |
| Max HR | Numeric | Maximum heart rate achieved |
| Exercise angina | Categorical | Exercise-induced angina (binary) |
| ST depression | Numeric | ST depression induced by exercise |
| Slope of ST | Categorical | Slope of the peak exercise ST segment |
| Number of vessels fluro | Categorical | Number of major vessels colored by fluoroscopy |
| Thallium | Categorical | Thallium stress test result |

> ⚠️ **WARNING:** Raw data files must NEVER be committed to GitHub. Ensure `data/` is in your `.gitignore`.

## Models

Three modeling approaches were applied and compared:

### K-Means Clustering (Unsupervised)
- Used for patient segmentation and risk profiling.
- Features scaled with `StandardScaler`; optimal k=2 selected via Elbow Method and Silhouette Score.
- Ex-post label mapping accuracy: 87.4%

### Logistic Regression (Primary Model)
- Binary classification with L2 regularization (`lbfgs` solver, `max_iter=2000`).
- Optimal threshold selected via Youden's J statistic on the ROC curve.
- **ROC-AUC: 0.9502**
- **Top 5 predictors:** Exercise angina, Sex, Chest pain type, Number of vessels fluro, Slope of ST.

### Decision Tree (Baseline)
- `max_depth=4`, `min_samples_leaf=30` for controlled complexity.
- Serves as an interpretable reference point.

### Results Comparison

| Metric | Logistic Regression | Clustering (ex-post) | Decision Tree |
|---|---|---|---|
| **Accuracy** | **0.882** | 0.874 | 0.854 |
| **Precision** | 0.862 | **0.895** | 0.828 |
| **Recall** | **0.876** | 0.815 | 0.853 |
| **Specificity** | **0.886** | 0.860 | 0.856 |
| **F1-Score** | **0.869** | 0.853 | 0.840 |

**Winner: Logistic Regression** — best overall balance across all metrics.

## Tech Stack

- Python 3.9+
- pip / virtualenv for environment management
- scikit-learn for model training
- FastAPI and Pydantic for serving and request validation
- Weights & Biases for experiment tracking and model artifacts
- Docker for containerized serving
- GitHub Actions for Continuous Integration
- Render for cloud deployment

## Configuration

`config.yaml` is the single source of truth for non-secret runtime settings.

It defines:

- file paths for raw data, processed data, model artifacts, predictions output, and logs
- classification problem settings (target column, valid targets)
- data split parameters
- training settings for each model
- feature groups (numeric passthrough, ordinal encode, min-max scaler, etc.)
- validation rules (max null rate, binary columns, age bounds)
- inference threshold
- Weights & Biases project and artifact settings

Secrets must stay in `.env`, not in `config.yaml`.

Example `.env` file:

```env
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_wandb_entity
```

## Setup

### 1) Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows (Command Prompt)
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the tests

```bash
pytest -q
```

### 4) Run the full pipeline

```bash
python -m src.main
```

Expected outputs:

- `data/processed/clean.csv`
- `models/logit_model.joblib`
- `models/kmeans_model.joblib`
- `models/dtrees_model.joblib`
- `reports/logit_predictions.csv`
- `logs/pipeline.log`

## FastAPI Service

The inference service is defined in `src/api.py`.

### Run Locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Local URLs

- Home: `http://127.0.0.1:8000/`
- Health check: `http://127.0.0.1:8000/health`
- Interactive docs: `http://127.0.0.1:8000/docs`

## Accessing the Render Deployment

The deployed service is available at:

```text
https://mlops-full.onrender.com
```

### API Docs

```text
https://mlops-full.onrender.com/docs
```

### Health Check

```text
https://mlops-full.onrender.com/health
```

## Prediction Request Format

The API expects a JSON payload with a `data` list. Each record must match the Pydantic schema defined in `src/api.py`.

### Example Request Body

```json
{
  "data": [
    {
      "Age": 52,
      "Sex": 1,
      "Chest pain type": 3,
      "BP": 130,
      "Cholesterol": 254,
      "FBS over 120": 0,
      "EKG results": 2,
      "Max HR": 147,
      "Exercise angina": 0,
      "ST depression": 1.4,
      "Slope of ST": 2,
      "Number of vessels fluro": 1,
      "Thallium": 7
    }
  ]
}
```

### Test with curl Against Render

```bash
curl -X POST "https://mlops-full.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "Age": 52,
        "Sex": 1,
        "Chest pain type": 3,
        "BP": 130,
        "Cholesterol": 254,
        "FBS over 120": 0,
        "EKG results": 2,
        "Max HR": 147,
        "Exercise angina": 0,
        "ST depression": 1.4,
        "Slope of ST": 2,
        "Number of vessels fluro": 1,
        "Thallium": 7
      }
    ]
  }'
```

### Test with Python

```python
import requests

url = "https://mlops-full.onrender.com/predict"
payload = {
    "data": [
        {
            "Age": 52,
            "Sex": 1,
            "Chest pain type": 3,
            "BP": 130,
            "Cholesterol": 254,
            "FBS over 120": 0,
            "EKG results": 2,
            "Max HR": 147,
            "Exercise angina": 0,
            "ST depression": 1.4,
            "Slope of ST": 2,
            "Number of vessels fluro": 1,
            "Thallium": 7,
        }
    ]
}

response = requests.post(url, json=payload, timeout=30)
print(response.status_code)
print(response.json())
```

## Expected API Behavior

- `GET /health` returns service health status
- `POST /predict` returns binary predictions and probability scores for each input record

## Weights & Biases

This project uses Weights & Biases for experiment tracking, model artifact versioning, and metrics logging.

**WandB Report:** [View the experiment report](https://api.wandb.ai/links/pengchongzhao1113-ie-university/op2lby3g)

The configuration in `config.yaml` defines the WandB project as `heart-disease-mlops` and the registered model artifact as `heart_disease_model_logit`.

## Docker

### Build the Image

```bash
docker build -t mlops-api:latest .
```

### Run the Container

```bash
docker run -p 8000:8000 --env-file .env --name mlops-api-container mlops-api:latest
```

### Test the Containerized API

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## Continuous Integration

The repository includes a GitHub Actions workflow under `.github/workflows/ci.yml` that automatically validates code quality and project integrity on each push or pull request.

## Testing Strategy

The `tests/` folder covers the major pipeline modules:

- data loading
- cleaning
- validation
- feature logic
- utilities

## Reproducibility Rules

- use `config.yaml` instead of hardcoding paths and settings
- keep secrets in `.env`
- run the orchestrator with `python -m src.main`
- run tests before pushing code
- serialize deployable model artifacts
- keep notebooks for exploration, not production orchestration

## License

This project includes an MIT License. Review `LICENSE` for details.
