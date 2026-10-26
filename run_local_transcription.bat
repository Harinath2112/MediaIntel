@echo off
setlocal
cd /d "%~dp0"

title MediaIntel - Hybrid Local Transcription Worker

echo ======================================================================
echo           MEDIAINTEL HYBRID LOCAL TRANSCRIPTION WORKER
echo ======================================================================
echo.
echo Responsibilities of this local worker:
echo   [1] Extract YouTube captions via your residential internet connection.
echo   [2] Fall back to local Faster-Whisper CPU audio transcription.
echo   [3] Persist completed transcripts directly to Supabase PostgreSQL.
echo.
echo Note: Cloud AI processing, sentiment analysis, topic generation, and
echo       RPA reporting remain 100%% cloud-based.
echo ======================================================================
echo.

:: 1. Check virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Local Python virtual environment not found at:
    echo         %~dp0.venv\Scripts\python.exe
    echo.
    echo Please ensure the .venv environment is set up.
    echo.
    pause
    exit /b 1
)

set "PYTHON_EXE=.venv\Scripts\python.exe"

:: 2. Check local configuration
if not exist ".env" (
    echo [ERROR] Configuration file '.env' is missing.
    echo Please copy .env.example to .env and configure MEDIAINTEL_DB_URL.
    echo.
    pause
    exit /b 1
)

:: 3. Verify core dependencies without printing credentials
echo Verifying Python environment and required modules...
"%PYTHON_EXE%" -c "
import sys
missing = []
for mod in ['sqlalchemy', 'faster_whisper', 'yt_dlp', 'youtube_transcript_api', 'pandas']:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
pg_found = False
for pg_mod in ['psycopg', 'psycopg2']:
    try:
        __import__(pg_mod)
        pg_found = True
        break
    except Exception:
        pass
if not pg_found:
    missing.append('psycopg/psycopg2')
if missing:
    print('[ERROR] Missing required Python package(s): ' + ', '.join(missing))
    sys.exit(1)
"
if errorlevel 1 (
    echo.
    echo [ERROR] Required dependencies are missing from .venv.
    echo Run 'run.bat' or activate .venv and run:
    echo     pip install -r requirements-local.txt
    echo.
    pause
    exit /b 1
)

echo Dependencies verified. Starting local transcription worker...
echo.

:: 4. Execute the local worker (passing any CLI arguments like --dry-run or --video-id)
"%PYTHON_EXE%" -m core.local_transcription_worker %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if %EXIT_CODE% equ 0 (
    echo [SUCCESS] Local transcription completed successfully.
) else (
    echo [NOTICE] Worker exited with code %EXIT_CODE%. Check the output above.
)
echo.
echo ======================================================================
echo Press any key to close this window.
echo ======================================================================
pause >nul
exit /b %EXIT_CODE%
