#!/bin/bash

set -euo pipefail

echo "=========================================="
echo "       Gemma 4 12B Server"
echo "=========================================="

echo
echo "[1] Python version"
python --version

echo
echo "[2] Existing NVIDIA PyTorch"
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo
echo "[3] GPU"
nvidia-smi

echo
echo "[4] Creating virtual environment"

python -m venv --system-site-packages .venv

source .venv/bin/activate

echo
echo "[5] Virtual environment"
which python
python --version

echo
echo "[6] Checking NVIDIA PyTorch inside venv"

python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo
echo "[7] Installing Gemma dependencies"

unset PIP_CONSTRAINT
echo $PIP_CONSTRAINT
python -m pip config list -v
env | grep -i pip

python -m pip install --upgrade pip

python -m pip install \
    --no-cache-dir \
    -r requirements.txt

echo
echo "[8] Starting Gemma server"

exec python main.py