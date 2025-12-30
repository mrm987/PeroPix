@echo off
title PeroPix Server

echo Starting PeroPix...

:: 백엔드 시작 (백그라운드)
start /B pythonw backend.py 2>nul || start /B python backend.py

:: 서버 시작 대기
timeout /t 2 /nobreak >nul

:: Chrome 앱 모드로 열기 (테두리 없이)
start "" chrome --app=http://127.0.0.1:8765 2>nul || start "" msedge --app=http://127.0.0.1:8765 2>nul || start "" "http://127.0.0.1:8765"

echo.
echo PeroPix running. Close this window to stop the server.
pause >nul
