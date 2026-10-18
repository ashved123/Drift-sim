@echo off
rem Double-click to open Drift Studio: tracks, training, models and live dashboards (Windows).
rem BeamNG is started when you first load a track or start training. Don't launch the game
rem from Steam first: a normal launch doesn't open the port Python controls it with.
rem First run creates the Python environment (venv) and installs packages.
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo First run: creating the Python environment...
    where py >nul 2>nul && (py -3 -m venv venv) || (python -m venv venv)
    if not exist "venv\Scripts\python.exe" (
        echo.
        echo Could not create the environment. Install Python 3.10-3.12 from python.org
        echo and tick "Add python.exe to PATH" during setup, then run this again.
        pause
        exit /b 1
    )
)

"venv\Scripts\python.exe" -c "import beamngpy, numpy, scipy, matplotlib, PIL, gymnasium, stable_baselines3, torch" >nul 2>nul
if errorlevel 1 (
    echo Installing Python packages - this can take several minutes the first time...
    "venv\Scripts\python.exe" -m pip install --upgrade pip
    "venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Package install failed - see the messages above.
        pause
        exit /b 1
    )
)

echo Starting Drift Studio...
"venv\Scripts\python.exe" scripts\drift_studio.py %*
if errorlevel 1 pause
