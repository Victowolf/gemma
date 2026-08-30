#!/bin/bash

set -euo pipefail

echo "=========================================="
echo "       Gemma 4 12B Server"
echo "=========================================="

echo
echo "[1] Python version"
python --version

echo
echo "[2] GPU"
nvidia-smi

echo
echo "[3] Installing requirements"

python -m pip install --upgrade pip

python -m pip install \
    --no-cache-dir \
    -r requirements.txt

echo
echo "[4] Starting main.py"

exec python main.py