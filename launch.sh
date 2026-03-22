#!/bin/bash
# PINN Research Platform — Unix launcher
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Checking dependencies..."
python3 -c "import torch, fastapi, uvicorn, numpy" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

echo "Starting PINN Research Platform..."
python3 run.py "$@"
