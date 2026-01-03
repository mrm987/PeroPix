@echo off
title PeroPix

echo Starting PeroPix...
echo.

:: Check required packages
python -c "import piexif" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install piexif -q
)

echo PeroPix is running at http://127.0.0.1:8765
echo Press Ctrl+C or close this window to stop.
echo.

:: Run backend (browser opens automatically when server is ready)
python backend.py

:: Show error message if failed
if errorlevel 1 (
    echo.
    echo [ERROR] PeroPix failed to start. See error above.
    pause
)
