# Telecom Customer Churn Prediction — MLOps Core Pipeline

End-to-end reproducible ML pipeline for telecom customer churn prediction.
This repository covers the **core MLOps milestone**: data ingestion, validation,
versioning with DVC, experiment tracking with MLflow, model registration, and a
fully reproducible `dvc.yaml` pipeline. Docker, REST API, cloud deployment, and
monitoring are intentionally excluded and will be added in later milestones.

## Team

- Rafia Afzal (23L-0821)
- Izma Asif (23L-0539)
- Adeeba Haq (23L-2540)

## Project Overview

- **Problem**: Predict whether a telecom customer will churn based on service,
  account, and demographic attributes.
- **Dataset**: [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
  (~7,043 rows, 21 columns).
- **Models**:
  - **Baseline**: Logistic Regression
  - **Main**: Random Forest
- **Selection metric**: ROC-AUC
- **MLOps tools**: DVC (DagsHub remote), MLflow (DagsHub tracking server), Git/GitHub.

## Folder Structure

```text
.
├── data
│   ├── raw/              # Raw CSV downloaded by DVC pipeline (DVC-tracked)
│   ├── processed/        # Train/test arrays + preprocessor (DVC-tracked)
│   └── external/         # Optional external data
├── models/               # Trained model artifacts (DVC-tracked)
├── metrics/              # Validation & evaluation reports (DVC-tracked)
├── notebooks/            # Exploratory notebooks
├── src
│   ├── data
│   │   ├── ingest.py     # Download dataset
│   │   ├── validate.py   # Schema & distribution checks
│   │   └── preprocess.py # Feature engineering & train/test split
│   └── models
│       ├── baseline.py   # Logistic Regression training + MLflow logging
│       ├── main_model.py # Random Forest training + MLflow logging
│       └── evaluate.py   # Compare models & register best in MLflow
├── tests/                # Pytest unit tests
├── dvc.yaml              # DVC pipeline definition
├── params.yaml           # Centralized configuration
├── requirements.txt
├── .env.example          # Template for secrets
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <your-github-repo-url>
cd telecom-churn-mlops
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

### 3. Configure DagsHub / MLflow credentials

Copy `.env.example` to `.env` and fill in your DagsHub details:

```bash
cp .env.example .env
```

Edit `.env`:

```env
DAGSHUB_REPO_OWNER=your_dagshub_username
DAGSHUB_REPO_NAME=your_dagshub_repo_name
DAGSHUB_TOKEN=your_dagshub_token
MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

> Do **not** commit `.env` to Git.

### 4. Configure DVC remote (DagsHub)

```bash
dvc remote add origin https://dagshub.com/<owner>/<repo>.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <token>
```

Pull existing data if any:

```bash
dvc pull
```

## How to Run the Pipeline

Reproduce the entire pipeline from scratch (activate the virtual environment first):

```bash
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

dvc repro
```

This executes the following stages:

1. `ingest` — downloads the Telco churn CSV.
2. `validate` — runs schema, null, range, and churn-rate checks.
3. `preprocess` — cleans data, encodes categoricals, scales features, splits train/test.
4. `train_baseline` — trains Logistic Regression and logs to MLflow.
5. `train_main` — trains Random Forest and logs to MLflow.
6. `evaluate` — compares both models by ROC-AUC and registers the best to the
   MLflow Model Registry with stage `Staging`.

### Visualize the pipeline

```bash
dvc dag
```

### Run individual stages

```bash
dvc repro ingest
dvc repro validate
dvc repro preprocess
dvc repro train_baseline
dvc repro train_main
dvc repro evaluate
```

Each stage is invoked as a Python module (e.g., `python -m src.data.ingest`),
so imports resolve correctly from the project root.

### Push data and pipeline state

```bash
dvc push
git add .
git commit -m "Run core pipeline"
git push
```

## MLflow Tracking

- Open your DagsHub MLflow UI: `https://dagshub.com/<owner>/<repo>/#/experiments`
- Each training run logs hyperparameters, metrics (accuracy, precision, recall,
  F1, ROC-AUC), and model artifacts.
- The `evaluate` stage registers the best-performing model to the Model Registry
  under the name `telco_churn_classifier` with stage `Staging`.

## Testing

Run the test suite with:

```bash
pytest tests/
```

## Data Validation Checks

Before any model training, `src/data/validate.py` enforces:

- Expected row and column counts.
- Presence of the target column `Churn`.
- Missing-value percentage below threshold.
- `TotalCharges` blank/missing count below threshold.
- Churn rate within expected range.
- Non-negative numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`).
- No duplicate `customerID` values.

If any check fails, the pipeline exits with a non-zero status.

## Configuration

All tunable settings live in `params.yaml`:

- `base.random_state`, `test_size`
- `data.source_url`, paths, expected shape
- `validation.*` thresholds
- `preprocessing.*` options
- `models.logistic_regression.*` and `models.random_forest.*`
- `mlflow.experiment_name`, `registered_model_name`, `metric_for_selection`

## Notes

- Raw data, processed data, models, and metrics are tracked by **DVC**, not Git.
- `.gitignore` excludes artifacts, virtual environments, and secrets.
- For this milestone the LLM explanation layer, REST API, Docker, Airflow,
  drift detection, and monitoring are out of scope and will follow in later
  deliverables.
