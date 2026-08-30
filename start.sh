#!/bin/bash

set -euo pipefail

echo "=========================================="
echo "       Gemma 4 12B Server"
echo "=========================================="

echo
echo "[1] Python version"
python --version

echo
echo "[2] Existing PyTorch"
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo
echo "[3] GPU"
nvidia-smi

echo
echo "[4] Installing dependencies"

python -m pip install \
    --no-cache-dir \
    -r requirements.txt

echo
echo "[5] Starting main.py"

exec python main.py