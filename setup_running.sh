#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$PROJECT_DIR"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CREATE_VENV="${CREATE_VENV:-1}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_TORCH="${INSTALL_TORCH:-auto}"
INSTALL_OPTIONAL_METRICS="${INSTALL_OPTIONAL_METRICS:-0}"
DOWNLOAD_BEST_MODEL="${DOWNLOAD_BEST_MODEL:-0}"
BEST_MODEL_FILE_ID="${BEST_MODEL_FILE_ID:-1fpGYgxU-9AxJMVhB9XGr_rc0IgfAUcNQ}"
BEST_MODEL_PATH="${BEST_MODEL_PATH:-checkpoints_content_unit_lrs2_10k_units32_l6_smooth5/best_model.pth}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
TORCH_CPU_INDEX_URL="${TORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

echo "[setup] project: $PROJECT_DIR"

if [[ "$CREATE_VENV" == "1" && -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[setup] creating virtualenv: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel

if [[ "$INSTALL_TORCH" != "0" ]]; then
  if [[ "$INSTALL_TORCH" == "1" ]] || ! python - <<'PY' >/dev/null 2>&1
import torch
print(torch.__version__)
PY
  then
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "[setup] installing PyTorch CUDA wheels from $TORCH_CUDA_INDEX_URL"
      python -m pip install --upgrade torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX_URL"
    else
      echo "[setup] installing PyTorch CPU wheels from $TORCH_CPU_INDEX_URL"
      python -m pip install --upgrade torch torchvision torchaudio --index-url "$TORCH_CPU_INDEX_URL"
    fi
  else
    echo "[setup] PyTorch already installed; set INSTALL_TORCH=1 to reinstall"
  fi
fi

echo "[setup] installing project requirements"
python -m pip install --upgrade -r requirements.txt

if [[ "$INSTALL_OPTIONAL_METRICS" == "1" ]]; then
  echo "[setup] installing optional PESQ + SpeechBrain HiFi-GAN dependencies"
  python -m pip install --upgrade pesq speechbrain || {
    echo "[warn] optional metric packages failed to install; inference still works without them"
  }
fi

if [[ "$DOWNLOAD_BEST_MODEL" == "1" ]]; then
  echo "[setup] downloading best ContentUnit checkpoint"
  python -m pip install --upgrade gdown
  mkdir -p "$(dirname "$BEST_MODEL_PATH")"
  python -m gdown "https://drive.google.com/uc?id=$BEST_MODEL_FILE_ID" -O "$BEST_MODEL_PATH"
  echo "[setup] checkpoint saved to: $PROJECT_DIR/$BEST_MODEL_PATH"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[warn] ffmpeg not found. Install it if you need broad audio/video format support:"
  echo "       sudo apt-get update && sudo apt-get install -y ffmpeg"
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "[setup] checking imports"
python - <<'PY'
import cv2
import google.protobuf
import librosa
import mediapipe as mp
import numpy as np
import sklearn
import torch
import transformers

from data.build_cache import LipLandmarkExtractor
from inference.infer_video import build_content_unit_model
from training import train_content_unit

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("numpy", np.__version__)
print("protobuf", google.protobuf.__version__)
print("mediapipe", mp.__version__)
print("imports ok")
PY

echo
echo "[setup] done"
echo "[setup] activate later with: source $PROJECT_DIR/$VENV_DIR/bin/activate"
echo "[setup] run commands from the repository root with: export PYTHONPATH=$PROJECT_DIR"
echo "[setup] optional metrics: INSTALL_OPTIONAL_METRICS=1 ./setup_running.sh"
echo "[setup] download checkpoint: DOWNLOAD_BEST_MODEL=1 ./setup_running.sh"
