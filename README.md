# IDS568 Milestone 3 — Workflow Automation & Experiment Tracking
![MLOps CI/CD Pipeline](https://github.com/kiranmusu14/ids568-mlops-project-mil3/actions/workflows/train_and_validate.yml/badge.svg)

## Overview
This project implements a complete automated ML workflow integrating **Apache Airflow** orchestration, **GitHub Actions** CI/CD, and **MLflow** experiment tracking on the scikit-learn Diabetes dataset.

---

## Repository Structure
```
ids568-milestone3-[netid]/
├── .github/
│   └── workflows/
│       └── train_and_validate.yml   # CI/CD pipeline
├── dags/
│   └── train_pipeline.py            # Airflow DAG
├── mlruns/                          # MLflow tracking data
├── tracking_evidence/               # JSON run exports + model .pkl files
├── model_validation.py              # Quality gate script
├── train_model.py                   # Training script with MLflow logging
├── requirements.txt                 # Pinned dependencies
├── lineage_report.md                # Experiment analysis report
└── README.md                        # This file
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ids568-milestone3-<netid>.git
cd ids568-milestone3-<netid>
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize Airflow
```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init
airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname User \
  --role Admin --email admin@example.com
```

---

## How to Run the Pipeline

### Option A — Run training manually (recommended for testing)
```bash
# Generate 5 MLflow runs with different hyperparameters
python train_model.py 50
python train_model.py 100
python train_model.py 150 5
python train_model.py 200 10 5
python train_model.py 300 15 4

# Validate the best model
python model_validation.py

# View MLflow UI
mlflow ui --workers 1
# Open http://127.0.0.1:5000
```

### Option B — Run via Airflow DAG
```bash
# Terminal 1: Start scheduler
airflow scheduler

# Terminal 2: Start webserver
airflow webserver --port 8080

# Trigger the DAG manually
airflow dags trigger train_pipeline
# Open http://localhost:8080
```

### Option C — CI/CD via GitHub Actions
Push to `main` branch — the workflow automatically:
1. Installs dependencies
2. Runs all 5 training experiments
3. Verifies MLflow logging
4. Runs the quality gate (`model_validation.py`)
5. Uploads tracking evidence as a build artifact

---

## Architecture Explanation

```
┌─────────────────────────────────────────────┐
│              Airflow DAG                    │
│                                             │
│  preprocess_data → train_model → register   │
│                                             │
│  • retries=2, retry_delay=5min              │
│  • on_failure_callback for alerting         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│              MLflow Tracking                │
│                                             │
│  Experiment: diabetes_rf_experiment         │
│  • Logs: params, metrics, artifacts, tags   │
│  • Registry: None → Staging → Production    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           GitHub Actions CI/CD              │
│                                             │
│  Push → Train → Validate → Gate → Deploy   │
│  Quality gate: R² ≥ 0.40, RMSE ≤ 65.0     │
└─────────────────────────────────────────────┘
```

---

## DAG Idempotency & Lineage Guarantees

### Idempotency
Every task in the DAG is designed to be safely re-runnable:

- **preprocess_data**: Always uses `random_state=42` and fixed `test_size=0.2`. Re-running produces identical train/test splits saved to the same numpy files.
- **train_model**: Output files are named by hyperparameters (e.g., `model_150_5_2.pkl`). Re-running with the same params overwrites with identical output.
- **register_model**: MLflow's `register_model` creates a new version on each run — it never overwrites existing versions, so re-runs are safe and traceable.

### Lineage Guarantees
Every model in the registry can be traced back to:
- **Code**: Source file (`train_model.py`) logged as MLflow run source
- **Data**: Fixed dataset (`sklearn_diabetes v1.0`) logged as parameter
- **Hyperparameters**: All params logged (`n_estimators`, `max_depth`, `min_samples_split`, `random_state`)
- **Artifact integrity**: SHA-256 hash logged as `model_hash` tag for every run

---

## CI-Based Model Governance

The `model_validation.py` script enforces three quality gates before any model can be promoted:

| Gate | Threshold | Reason |
|---|---|---|
| R² Score | ≥ 0.40 | Model must explain at least 40% of variance |
| RMSE | ≤ 65.0 | Error ceiling on diabetes progression scale |
| Model hash | Must exist | Artifact integrity verification |

If any gate fails, the CI pipeline exits with code `1`, blocking promotion to staging.

---

## Experiment Tracking Methodology

- All runs logged to a single MLflow experiment: `diabetes_rf_experiment`
- Each run tracks: hyperparameters, metrics (R², RMSE, MAE, MSE), model artifact, and SHA-256 hash
- Runs are named automatically by MLflow for reproducibility
- Best run selected by highest R² score and registered to `DiabetesRFModel` registry
- Version progression: `None → Staging → Production`

---

## Operational Notes

### Retry Strategy & Failure Handling
- Each Airflow task retries up to **2 times** with a **5-minute delay** between attempts
- `on_failure_callback` prints task ID, DAG ID, and execution date for immediate diagnosis
- CI pipeline fails fast on quality gate breach with a descriptive error message

### Monitoring & Alerting Recommendations
- Monitor R² score trend across runs — degradation signals data drift
- Alert if RMSE exceeds 65.0 on new data
- Set up MLflow webhook or Slack alert when a model is promoted to Production
- Track model hash changes to detect unexpected artifact modifications

### Rollback Procedures
1. In MLflow UI → Models → `DiabetesRFModel`
2. Find the previously promoted version
3. Click **"Transition to Production"** on the previous version
4. Archive the bad version
5. Or via CLI:
```bash
python -c "
from mlflow.tracking import MlflowClient
client = MlflowClient()
# Roll back to version 1
client.transition_model_version_stage('DiabetesRFModel', 1, 'Production')
# Archive bad version
client.transition_model_version_stage('DiabetesRFModel', 2, 'Archived')
"
```