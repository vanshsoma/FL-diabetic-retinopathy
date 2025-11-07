@echo off
ECHO ======================================
ECHO  Starting Federated Test Clients (2x)
ECHO ======================================
ECHO.

ECHO 1. Activating Python virtual environment...
CALL server\venv\Scripts\activate
IF %ERRORLEVEL% NEQ 0 (
    ECHO ERROR: Virtual environment 'server\venv' not found.
    pause
    exit /b
)
ECHO Venv activated.
ECHO.

ECHO 2. Running 2 separate clients with unique IDs (CID 1 and 2)...
ECHO ------------------------------------------------------------

:: This loop runs the test_client.py 2 times, passing the CID as an argument
FOR /L %%i IN (1,1,2) DO (
    ECHO Starting Client ID: %%i
    :: The 'start' command runs each client in a NEW, separate window
    start "MobileNetV2 Client %%i" cmd /k "CALL server\venv\Scripts\activate && python test_client.py %%i"
    timeout /t 1 /nobreak >nul
)

ECHO ------------------------------------------------------------
ECHO.
ECHO.
ECHO ✅ Both client windows have launched.
pause