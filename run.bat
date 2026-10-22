
@echo off
setlocal
cd /d "%~dp0"

title MediaIntel - AI Competitive Intelligence

echo ==========================================
echo       MEDIAINTEL PROJECT LAUNCHER
echo ==========================================
echo.

where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 --version >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=py -3.12"
        goto :python_found
    )
)

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Install Python 3.12 from python.org
    pause
    exit /b 1
)
set "PYTHON=python"

:python_found
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :failed
)

set "VENV_PYTHON=.venv\Scripts\python.exe"

if not exist ".venv\.mediaintel_installed" (
    echo Installing dependencies...
    "%VENV_PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto :failed

    "%VENV_PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 goto :failed

    "%VENV_PYTHON%" -m pip install --upgrade yt-dlp
    if errorlevel 1 goto :failed

    type nul > ".venv\.mediaintel_installed"
)

if not exist ".env" (
    echo WARNING: .env file is missing.
    echo Configure environment variables before running.
    pause
    exit /b 1
)

echo.
echo Starting MediaIntel...
echo.

"%VENV_PYTHON%" main_orchestrator.py %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo MediaIntel exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%

:failed
echo.
echo ERROR: Setup failed. Check the messages above.
pause
exit /b 1
