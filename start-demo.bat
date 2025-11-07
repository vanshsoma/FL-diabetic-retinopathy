@echo off
ECHO ==========================================================
ECHO  Starting Federated DR Server (Flower + Streamlit)
ECHO ==========================================================
ECHO.

ECHO 1. Activating Python virtual environment...
CALL server\venv\Scripts\activate
IF %ERRORLEVEL% NEQ 0 (
    ECHO ERROR: Virtual environment 'server\venv' not found.
    ECHO Please run the one-time setup steps to create the venv.
    pause
    exit /b
)
ECHO venv activated.
ECHO.

ECHO 2. Starting Flower Server (Port 8080) in background...
ECHO A new window for the Flower server will open.

:: This is the fix: We must explicitly tell the new 'start' command
:: to activate the venv before running the python script.
start "Flower Server (FL)" cmd /k "CALL server\venv\Scripts\activate && python server\flower_server.py"

ECHO.
ECHO 3. Starting Streamlit Dashboard (Port 8501)...
ECHO.
streamlit run server\dashboard.py --server.port 8501