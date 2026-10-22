
@echo off
setlocal
cd /d "%~dp0"

title MediaIntel Dashboard

echo ====================================
echo     MEDIAINTEL DASHBOARD
echo ====================================

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo Run run.bat first to install dependencies.
    pause
    exit /b 1
)

echo Starting dashboard...
echo Open http://localhost:8501
echo.

".venv\Scripts\python.exe" -m streamlit run dashboard/dashboard_app.py --server.address 127.0.0.1 --server.port 8501

pause
