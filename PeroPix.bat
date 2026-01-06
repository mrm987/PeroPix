@echo off
title PeroPix
cd /d "%~dp0"

echo.
echo  ============================================
echo   PeroPix - Starting...
echo   (First run may take a few minutes to setup)
echo  ============================================
echo.

:: Python path (same as build)
set PYTHON_EXE=%~dp0python\python.exe

:: Use python if exists, otherwise use system Python
if exist "%PYTHON_EXE%" (
    echo Using python\python.exe (Python 3.11.9^)
    "%PYTHON_EXE%" backend.py
) else (
    :: Fallback: System Python (first-run auto-install will create python/)
    echo Using system Python...
    echo First-run auto-install will set up python environment...
    echo.

    :: Check required packages
    python -c "import piexif" 2>nul
    if errorlevel 1 (
        echo Installing required packages...
        pip install piexif -q
    )

    python backend.py
)

echo.
echo PeroPix is running at http://127.0.0.1:8765
echo Press Ctrl+C or close this window to stop.

:: Show error message if failed
if errorlevel 1 (
    echo.
    echo [ERROR] PeroPix failed to start. See error above.
    pause
)
