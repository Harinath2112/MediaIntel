
#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "       MEDIAINTEL PROJECT LAUNCHER"
echo "=========================================="

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON=python3.12
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    echo "ERROR: Python 3 is not installed."
    exit 1
fi

if [ ! -f ".venv/bin/python" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi

VENV_PYTHON=".venv/bin/python"

if [ ! -f ".venv/.mediaintel_installed" ]; then
    echo "Installing dependencies..."
    "$VENV_PYTHON" -m pip install --upgrade pip
    "$VENV_PYTHON" -m pip install -r requirements.txt
    "$VENV_PYTHON" -m pip install --upgrade yt-dlp
    touch ".venv/.mediaintel_installed"
fi

if [ ! -f ".env" ]; then
    echo "ERROR: .env file is missing."
    echo "Create it from .env.example and configure it."
    exit 1
fi

echo "Starting MediaIntel..."

"$VENV_PYTHON" main_orchestrator.py "$@"

echo "MediaIntel finished."
