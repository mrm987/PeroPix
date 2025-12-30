#!/usr/bin/env python3
"""
PeroPix - PyWebView Desktop App
NAI + Local Diffusers Image Generator
"""
import sys
import os
import threading
import time
import socket

# 경로 설정 (PyInstaller 빌드 대응)
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(APP_DIR)


def is_port_in_use(port: int) -> bool:
    """포트 사용 중인지 확인"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def wait_for_server(port: int, timeout: int = 30) -> bool:
    """서버가 준비될 때까지 대기"""
    start = time.time()
    while time.time() - start < timeout:
        if is_port_in_use(port):
            return True
        time.sleep(0.1)
    return False


def start_backend():
    """FastAPI 백엔드 시작 (별도 스레드)"""
    import uvicorn
    from backend import app

    # uvicorn 로그 설정
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8765,
        log_level="warning",  # 로그 최소화
        access_log=False
    )


def main():
    import webview

    # 이미 서버가 실행 중인지 확인
    server_already_running = is_port_in_use(8765)

    if not server_already_running:
        # 백엔드 서버 시작 (daemon 스레드)
        server_thread = threading.Thread(target=start_backend, daemon=True)
        server_thread.start()

        # 서버 준비 대기
        print("Starting backend server...")
        if not wait_for_server(8765, timeout=30):
            print("Error: Backend server failed to start")
            sys.exit(1)
        print("Backend server ready!")
    else:
        print("Backend server already running")

    # PyWebView 창 생성
    window = webview.create_window(
        title="PeroPix",
        url="http://127.0.0.1:8765",
        width=1400,
        height=900,
        min_size=(800, 600),
        resizable=True,
        frameless=False,
        easy_drag=False,
        text_select=True,
    )

    # WebView 시작 (메인 스레드에서 실행)
    webview.start(
        debug=False,  # 릴리즈에서는 False
        http_server=False,  # 이미 FastAPI 사용 중
    )


if __name__ == "__main__":
    main()
