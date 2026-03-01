import sys
import os
import json
import hashlib
import pickle

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Config ────────────────────────────────────────────────────────────────────
os.makedirs("tracking_evidence", exist_ok=True)

n_estimators  = int(sys.argv[1])   if len(sys.argv) > 1 else 100
max_depth     = int(sys.argv[2])   if len(sys.argv) > 2 else None
min_samples   = int(sys.argv[3])   if len(sys.argv) > 3 else 2
RANDOM_STATE  = 42

print(f"--- Starting Run: n_estimators={n_estimators}, max_depth={max_depth}, "
      f"min_samples_split={min_samples} ---")

# ── Data ──────────────────────────────────────────────────────────────────────
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=RANDOM_STATE
)

# ── Train ─────────────────────────────────────────────────────────────────────
model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples,
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# ── Metrics ───────────────────────────────────────────────────────────────────
r2   = r2_score(y_test, preds)
mse  = mean_squared_error(y_test, preds)
mae  = mean_absolute_error(y_test, preds)
rmse = mse ** 0.5

print(f"  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

# ── Artifact hash ─────────────────────────────────────────────────────────────
model_bytes  = pickle.dumps(model)
model_hash   = hashlib.sha256(model_bytes).hexdigest()

model_path = f"tracking_evidence/model_{n_estimators}_{max_depth}_{min_samples}.pkl"
with open(model_path, "wb") as f:
    f.write(model_bytes)

# ── MLflow run ────────────────────────────────────────────────────────────────
mlflow.set_experiment("diabetes_rf_experiment")

with mlflow.start_run() as run:
    # Parameters
    mlflow.log_params({
        "n_estimators":      n_estimators,
        "max_depth":         str(max_depth),   # None is not JSON-safe otherwise
        "min_samples_split": min_samples,
        "random_state":      RANDOM_STATE,
        "test_size":         0.2,
        "dataset":           "sklearn_diabetes",
        "data_version":      "v1.0",
    })

    # Metrics
    mlflow.log_metrics({
        "r2_score": round(r2,   6),
        "mse":      round(mse,  6),
        "mae":      round(mae,  6),
        "rmse":     round(rmse, 6),
    })

    # Tags
    mlflow.set_tags({
        "model_hash":  model_hash,
        "model_type":  "RandomForestRegressor",
        "stage":       "experimental",
    })

    # Log model artifact to MLflow
    mlflow.sklearn.log_model(model, "model")

    # Also log the .pkl file directly
    mlflow.log_artifact(model_path, artifact_path="pkl")

    run_id = run.info.run_id
    print(f"  MLflow run_id: {run_id}")

# ── JSON evidence (for offline / CI use) ─────────────────────────────────────
run_data = {
    "run_id":  run_id,
    "params":  {
        "n_estimators":      n_estimators,
        "max_depth":         max_depth,
        "min_samples_split": min_samples,
        "random_state":      RANDOM_STATE,
    },
    "metrics": {
        "r2_score": round(r2,   6),
        "mse":      round(mse,  6),
        "mae":      round(mae,  6),
        "rmse":     round(rmse, 6),
    },
    "tags": {
        "model_hash": model_hash,
        "model_file": model_path,
    },
    "status": "FINISHED",
}

evidence_file = f"tracking_evidence/run_{n_estimators}_{max_depth}_{min_samples}.json"
with open(evidence_file, "w") as f:
    json.dump(run_data, f, indent=4)

# Keep a "latest" pointer so model_validation.py always finds something
with open("tracking_evidence/latest_run.json", "w") as f:
    json.dump(run_data, f, indent=4)

print(f"✅ Done! Evidence → {evidence_file}")
print(f"   Model hash: {model_hash[:16]}...")