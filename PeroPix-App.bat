@echo off
title PeroPix Server

echo Starting PeroPix...

:: Embedded Python path (with CUDA torch for Local generation)
set EMBEDDED_PYTHON=%~dp0python_env\python\python.exe
set EMBEDDED_PYTHONW=%~dp0python_env\python\pythonw.exe

:: Start backend in background - prefer embedded Python
if exist "%EMBEDDED_PYTHONW%" (
    echo Using embedded Python with CUDA support...
    start /B "%EMBEDDED_PYTHONW%" backend.py
) else if exist "%EMBEDDED_PYTHON%" (
    start /B "%EMBEDDED_PYTHON%" backend.py
) else (
    echo Using system Python...
    start /B pythonw backend.py 2>nul || start /B python backend.py
)

:: Wait for server to start
timeout /t 2 /nobreak >nul

:: Open in Chrome app mode (borderless)
start "" chrome --app=http://127.0.0.1:8765 2>nul || start "" msedge --app=http://127.0.0.1:8765 2>nul || start "" "http://127.0.0.1:8765"

echo.
echo PeroPix running. Close this window to stop the server.
pause >nul
