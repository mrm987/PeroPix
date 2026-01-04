@echo off
title PeroPix

echo Starting PeroPix...
echo.

:: Embedded Python path (with CUDA torch for Local generation)
set EMBEDDED_PYTHON=%~dp0python_env\python\python.exe

:: Use embedded Python if exists (CUDA support)
if exist "%EMBEDDED_PYTHON%" (
    echo Using embedded Python with CUDA support...
    "%EMBEDDED_PYTHON%" backend.py
) else (
    :: Fallback: System Python (NAI only)
    echo Using system Python...
    
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
