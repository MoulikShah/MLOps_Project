#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <MLFLOW_RUN_ID>"
  exit 1
fi
RUN_ID=$1
REGISTER_NAME="${2-}"           # may be empty if you don't pass it

# ─── Paths & MLflow URI ───────────────────────────────────────────────────────
MLFLOW_URI="http://129.114.27.48:8000"
export MLFLOW_TRACKING_URI="$MLFLOW_URI"

DEST_DIR="MLOps_Project/eval_testing/tests"
DEST_FILE="$DEST_DIR/backbone.pth"
TMP_DIR="tmp_model_${RUN_ID}"

VENV_DIR=".mlflow_venv"

# ─── 1) Bootstrap venv & install dependencies ─────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
  echo "▶ Creating virtualenv in $VENV_DIR…"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "▶ Installing MLflow & pytest into the venv if missing…"
pip install --upgrade pip
pip install mlflow pytest

# ─── 2) Download the model via run‐level API ─────────────────────────────────
echo "▶ Downloading model for run_id=$RUN_ID…"
mlflow artifacts download \
  --artifact-uri "runs:/$RUN_ID/model" \
  --dst-path "$TMP_DIR"

PTH=$(find "$TMP_DIR" -type f -name "*.pth" | head -n1)
if [[ -z "$PTH" ]]; then
  echo "❌ No .pth found under $TMP_DIR"
  rm -rf "$TMP_DIR"
  deactivate
  exit 1
fi

mkdir -p "$DEST_DIR"
cp "$PTH" "$DEST_FILE"
echo "✔ Model copied to $DEST_FILE"
rm -rf "$TMP_DIR"

# ─── 3) Run pytest and log results in MLflow ─────────────────────────────────
echo "▶ Running offline tests and logging results in MLflow…"

# Use the venv’s python explicitly
"$VENV_DIR/bin/python" << 'PYCODE'
import os, sys
import mlflow, pytest

# auto‐create or switch to the Offline_Testing experiment
mlflow.set_experiment("Offline_Testing")

# run pytest with JUnit XML output
junit = "pytest_results.xml"
ret = pytest.main(["-q", "--disable-warnings", f"--junitxml={junit}", "MLOps_Project/eval_testing/tests"])

with mlflow.start_run(run_name="offline_tests") as run:
    # log the XML report
    if os.path.isfile(junit):
        mlflow.log_artifact(junit, artifact_path="pytest_reports")
    # log metrics
    mlflow.log_metric("tests_passed", 1 if ret == 0 else 0)
    mlflow.log_metric("pytest_exit_code", ret)
    # fail the run (and script) if tests failed
    if ret != 0:
        sys.exit(ret)

print("✅ Offline tests passed and logged in MLflow")
PYCODE

# Note: pytest run created and closed the MLflow run; tests passed if we reach here

# ───────────────────────────────────────────────────────────────
# 4) Optionally register & promote to Staging
# ───────────────────────────────────────────────────────────────
if [[ -n "$REGISTER_NAME" ]]; then
  echo "▶ Registering model as '$REGISTER_NAME' and promoting to Staging…"
  # Use the same venv python to call the MLflow client
  export REGISTER_NAME RUN_ID
  "$VENV_DIR/bin/python" << 'PYCODE'
import os
from mlflow.tracking import MlflowClient

uri = os.environ["MLFLOW_TRACKING_URI"]
run_id = os.environ["RUN_ID"]
name   = os.environ["REGISTER_NAME"]
client = MlflowClient(uri)

# create registered model if needed
try:
    client.create_registered_model(name)
except Exception:
    pass

# register this run's artifact
model_uri = f"runs:/{run_id}/model"
mv = client.create_model_version(name=name, source=model_uri, run_id=run_id)

# promote to Staging, archiving any existing Staging versions
client.transition_model_version_stage(
    name=name,
    version=mv.version,
    stage="Staging",
    archive_existing_versions=True
)
print(f"✅ Registered '{name}' version {mv.version} and promoted to Staging")
PYCODE
fi

deactivate
echo "🎉 All done."
