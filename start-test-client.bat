@echo off
ECHO ==================================
ECHO  Starting Federated Test Client...
ECHO ==================================
ECHO.

ECHO 1. Activating Python virtual environment...
CALL server\venv\Scripts\activate
IF %ERRORLEVEL% NEQ 0 (
    ECHO ERROR: Virtual environment 'server\venv' not found.
    ECHO Please run 'start_demo.bat' first to create the venv.
    pause
    exit /b
)
ECHO Venv activated.
ECHO.

ECHO 2. Checking for test_client.py...
IF NOT EXIST test_client.py (
    ECHO ERROR: test_client.py not found in this directory.
    pause
    exit /b
)
ECHO Found test_client.py.
ECHO.

ECHO 3. Running the test client...
ECHO This will connect to the server, simulate 10 rounds, and then exit.
ECHO ------------------------------------------------------------
python test_client.py
ECHO ------------------------------------------------------------
ECHO.
ECHO.
ECHO ✅ Test client has finished all 10 rounds.
ECHO This window will close in 15 seconds.
timeout /t 15