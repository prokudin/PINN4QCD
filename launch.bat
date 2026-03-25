@echo off
REM PINN Research Platform — Windows launcher
cd /d "%~dp0"

echo Checking dependencies...
python -c "import torch, fastapi, uvicorn, numpy" 2>nul || (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo Starting PINN Research Platform...
python run.py %*
