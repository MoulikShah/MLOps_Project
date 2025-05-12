#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────────────────────────────────────────────────────────────
# 0) Configuration
# ───────────────────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/MoulikShah/MLOps_Project.git"
CLONE_DIR="MLOps_Project"

VENV_DIR=".mlflow_venv"
MLFLOW_URI="http://129.114.27.48:8000"
export MLFLOW_TRACKING_URI="$MLFLOW_URI"

MODEL_NAME="Staging_demo"

FASTAPI_PTH_DIR="model_serve/fastapi_pt"
FASTAPI_PTH_PATH="$FASTAPI_PTH_DIR/model.pth"
TMP_DIR="tmp_model"

DOCKER_COMPOSE_FILE="model_serve/docker-compose-fastapi.yaml"

# ───────────────────────────────────────────────────────────────────────────────
# 1) Ensure Docker is installed (via get.docker.com) and user in docker group
# ───────────────────────────────────────────────────────────────────────────────
if ! command -v docker &> /dev/null; then
  echo "▶ Installing Docker..."
  curl -sSL https://get.docker.com/ | sudo sh
  sudo groupadd -f docker
  sudo usermod -aG docker "$USER"
  echo "✅ Docker installed. Please run 'newgrp docker' or re-login for group changes to take effect."
else
  echo "✅ Docker already installed."
fi

# ───────────────────────────────────────────────────────────────────────────────
# 1) Clone the repo if needed
# ───────────────────────────────────────────────────────────────────────────────
if [[ ! -d "$CLONE_DIR" ]]; then
  echo "▶ Cloning repository…"
  git clone "$REPO_URL" "$CLONE_DIR"
else
  echo "✔ Repository already present"
fi
cd "$CLONE_DIR"

# ───────────────────────────────────────────────────────────────────────────────
# 2) Bootstrap virtualenv & install MLflow
# ───────────────────────────────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
  echo "▶ Creating Python virtualenv…"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "▶ Installing MLflow CLI…"
pip install --upgrade pip
pip install mlflow

# ───────────────────────────────────────────────────────────────────────────────
# 3) Determine latest version of $MODEL_NAME
# ───────────────────────────────────────────────────────────────────────────────
export MODEL_NAME
MODEL_VERSION=$("$VENV_DIR/bin/python" << 'PYCODE'
import os
from mlflow.tracking import MlflowClient

client = MlflowClient(os.environ["MLFLOW_TRACKING_URI"])
mv_list = client.search_model_versions(f"name='{os.environ['MODEL_NAME']}'")
# pick the one with highest numeric version
versions = [int(mv.version) for mv in mv_list]
print(max(versions))
PYCODE
)

echo "✔ Found latest version of $MODEL_NAME: v$MODEL_VERSION"

# ───────────────────────────────────────────────────────────────────────────────
# 4) Download that model version into your FastAPI folder
# ───────────────────────────────────────────────────────────────────────────────
echo "▶ Downloading models:/$MODEL_NAME/$MODEL_VERSION …"
mlflow artifacts download \
  --artifact-uri "models:/$MODEL_NAME/$MODEL_VERSION" \
  --dst-path "$TMP_DIR"

PTH_FILE=$(find "$TMP_DIR" -type f -name "*.pth" | head -n1)
if [[ -z "$PTH_FILE" ]]; then
  echo "❌ No .pth found in $TMP_DIR"
  rm -rf "$TMP_DIR"
  deactivate
  exit 1
fi

mkdir -p "$FASTAPI_PTH_DIR"
cp "$PTH_FILE" "$FASTAPI_PTH_PATH"
echo "✔ Copied model to $FASTAPI_PTH_PATH"
rm -rf "$TMP_DIR"

# ───────────────────────────────────────────────────────────────────────────────
# 5) Launch FastAPI + Jupyter
# ───────────────────────────────────────────────────────────────────────────────
echo "▶ Starting FastAPI + Jupyter via $DOCKER_COMPOSE_FILE …"
sudo docker compose -f "$DOCKER_COMPOSE_FILE" up -d --build

echo "🎉 Done!"
echo "  • FastAPI: http://localhost:8000"
echo "  • Jupyter: http://localhost:8888"

deactivate
