#!/usr/bin/env bash
# Open Drift Studio on Linux (the Windows equivalent is start_studio.bat).
#   ./start_studio.sh          (first time: chmod +x start_studio.sh if needed)
# First run creates the Python environment (venv/) and installs packages.
# BeamNG doesn't need starting first: Drift Studio's "Launch BeamNG" button (and
# training) start it through Proton with the API port open.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PY="venv/bin/python"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 isn't installed: sudo apt install python3 python3-venv python3-tk"
    exit 1
fi
if ! python3 -c "import tkinter" >/dev/null 2>&1; then
    echo "Drift Studio needs Tk for Python: sudo apt install python3-tk"
    exit 1
fi

if [ ! -x "$PY" ]; then
    echo "First run: creating the Python environment..."
    if ! python3 -m venv venv; then
        echo "Couldn't create the environment: sudo apt install python3-venv"
        exit 1
    fi
fi

if ! "$PY" -c "import beamngpy, numpy, scipy, matplotlib, PIL, gymnasium, stable_baselines3, torch, tensorboard" >/dev/null 2>&1; then
    echo "Installing Python packages - this can take several minutes the first time..."
    "$PY" -m pip install --upgrade pip
    # CPU-only PyTorch: the default Linux wheel bundles NVIDIA CUDA (several GB), useless without an NVIDIA GPU.
    if ! "$PY" -c "import torch" >/dev/null 2>&1; then
        "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi
    "$PY" -m pip install -r requirements.txt
fi

echo "Starting Drift Studio..."
exec "$PY" scripts/drift_studio.py "$@"
