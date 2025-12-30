#!/usr/bin/env python3
"""
PeroPix PyInstaller Build Script
Creates a standalone executable for Windows/macOS/Linux
"""
import subprocess
import sys
import platform
import shutil
from pathlib import Path

APP_NAME = "PeroPix"
MAIN_SCRIPT = "app.py"

def build():
    # PyInstaller 확인
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # 기존 빌드 정리
    for folder in ["build", "dist"]:
        if Path(folder).exists():
            shutil.rmtree(folder)

    # PyInstaller 옵션
    options = [
        MAIN_SCRIPT,
        f"--name={APP_NAME}",
        "--onedir",  # 폴더 모드 (onefile보다 빠름)
        "--windowed",  # 콘솔 창 숨김
        "--noconfirm",
        # 필요한 데이터 파일 포함
        "--add-data=index.html:.",
        "--add-data=backend.py:.",
        # Hidden imports (pywebview + FastAPI)
        "--hidden-import=uvicorn.logging",
        "--hidden-import=uvicorn.loops",
        "--hidden-import=uvicorn.loops.auto",
        "--hidden-import=uvicorn.protocols",
        "--hidden-import=uvicorn.protocols.http",
        "--hidden-import=uvicorn.protocols.http.auto",
        "--hidden-import=uvicorn.protocols.websockets",
        "--hidden-import=uvicorn.protocols.websockets.auto",
        "--hidden-import=uvicorn.lifespan",
        "--hidden-import=uvicorn.lifespan.on",
        "--hidden-import=webview",
        "--hidden-import=clr",  # Windows WebView2
    ]

    # Windows 아이콘
    if platform.system() == "Windows":
        if Path("icon.ico").exists():
            options.append("--icon=icon.ico")

    # macOS 아이콘
    if platform.system() == "Darwin":
        if Path("icon.icns").exists():
            options.append("--icon=icon.icns")

    # 빌드 실행
    print(f"Building {APP_NAME}...")
    print(f"Platform: {platform.system()}")

    cmd = [sys.executable, "-m", "PyInstaller"] + options
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\nBuild successful!")
        print(f"Output: dist/{APP_NAME}/")

        # 추가 파일 복사 (outputs, presets 폴더 구조)
        dist_path = Path("dist") / APP_NAME
        for folder in ["outputs", "presets", "prompts", "models"]:
            (dist_path / folder).mkdir(exist_ok=True)

        # prompts 하위 폴더
        for subfolder in ["base", "negative", "character"]:
            (dist_path / "prompts" / subfolder).mkdir(exist_ok=True)

        # models 하위 폴더
        for subfolder in ["checkpoints", "loras", "upscale_models"]:
            (dist_path / "models" / subfolder).mkdir(exist_ok=True)

        print(f"Created folder structure in dist/{APP_NAME}/")
    else:
        print("Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    build()
