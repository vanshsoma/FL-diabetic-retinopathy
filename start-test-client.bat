@echo off
cls
ECHO ==================================
ECHO  Netra.ai Client Startup
ECHO ==================================
ECHO.

:: --- Pre-flight Checks ---
ECHO 1. Verifying setup...
IF NOT EXIST "server\venv\Scripts\activate.bat" (
    ECHO [ERROR] Virtual environment 'venv' not found.
    ECHO          Please run these commands first:
    ECHO          1. py -3.11 -m venv venv
    ECHO          2. .\venv\Scripts\activate
    ECHO          3. pip install -r requirements.txt
    pause
    exit /b
)
IF NOT EXIST "test_client.py" (
    ECHO [ERROR] 'test_client.py' not found in this folder.
    pause
    exit
)
IF NOT EXIST "data_zips" (
    ECHO [ERROR] 'data_zips' folder not found.
    ECHO          Please create it and add your hospital_N_data.zip file.
    pause
    exit /b
)
ECHO    ... Setup verified.
ECHO.

:: --- Activate Environment ---
ECHO 2. Activating Python virtual environment...
CALL venv\Scripts\activate
ECHO    ... venv activated.
ECHO.

:: --- Get User Input ---
:ask_id
set /p HOSPITAL_ID="Enter your Hospital ID (e.g., 1, 2, 3, or 4): "
IF "%HOSPITAL_ID%"=="" (
    ECHO Please enter a valid ID.
    goto ask_id
)
ECHO.

:ask_ip
set /p SERVER_IP="Enter the Server's IP Address (e.g., 192.168.1.10): "
IF "%SERVER_IP%"=="" (
    ECHO Please enter a valid IP.
    goto ask_ip
)
ECHO.

:: --- Run the Client ---
ECHO ==================================
ECHO Starting client %HOSPITAL_ID%...
ECHO Connecting to server at %SERVER_IP%:8080...
ECHO ==================================
ECHO.

python test_client.py %HOSPITAL_ID% %SERVER_IP%

ECHO.
ECHO ==================================
ECHO Client has disconnected.
ECHO ==================================
pause