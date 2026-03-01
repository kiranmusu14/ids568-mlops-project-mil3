from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator


# ── Failure callback ──────────────────────────────────────────────────────────
def on_failure_callback(context):
    task_id   = context.get('task_instance').task_id
    dag_id    = context.get('task_instance').dag_id
    exec_date = context.get('execution_date')
    print(f"[ALERT] Task '{task_id}' in DAG '{dag_id}' failed at {exec_date}.")
    print("Action: Check logs and retry manually if needed.")


# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 27),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': on_failure_callback,
}


# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    'train_pipeline',
    default_args=default_args,
    description='MLOps Pipeline: Preprocess → Train → Register',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['mlops', 'milestone3'],
) as dag:

    # ── Task 1: Preprocess ────────────────────────────────────────────────────
    # Idempotent: always writes same output for same input data
    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command=(
            'python -c "'
            'from sklearn.datasets import load_diabetes; '
            'from sklearn.model_selection import train_test_split; '
            'import numpy as np, os; '
            'os.makedirs(\'tracking_evidence\', exist_ok=True); '
            'data = load_diabetes(); '
            'X_train, X_test, y_train, y_test = train_test_split('
            'data.data, data.target, test_size=0.2, random_state=42); '
            'np.save(\'tracking_evidence/X_train.npy\', X_train); '
            'np.save(\'tracking_evidence/X_test.npy\', X_test); '
            'np.save(\'tracking_evidence/y_train.npy\', y_train); '
            'np.save(\'tracking_evidence/y_test.npy\', y_test); '
            'print(\'Preprocessing complete. Shapes: \', X_train.shape, X_test.shape)'
            '"'
        ),
    )

    # ── Task 2: Train ─────────────────────────────────────────────────────────
    # Idempotent: same hyperparams always produce same model file name
    train_model = BashOperator(
        task_id='train_model',
        bash_command='python train_model.py 150 5 2',
    )

    # ── Task 3: Register ──────────────────────────────────────────────────────
    # Idempotent: MLflow register_model creates new version safely on re-run
    register_model = BashOperator(
        task_id='register_model',
        bash_command=(
            'python -c "'
            'import mlflow, glob, json; '
            'mlflow.set_tracking_uri(\'./mlruns\'); '
            'files = glob.glob(\'tracking_evidence/run_*.json\'); '
            'best = max(files, key=lambda f: json.load(open(f))[\'metrics\'][\'r2_score\']); '
            'run_data = json.load(open(best)); '
            'run_id = run_data[\'run_id\']; '
            'result = mlflow.register_model(f\'runs:/{run_id}/model\', \'DiabetesRFModel\'); '
            'print(f\'Registered model version: {result.version}\')'
            '"'
        ),
    )

    # ── Dependencies ──────────────────────────────────────────────────────────
    preprocess_data >> train_model >> register_model