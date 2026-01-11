#!/usr/bin/env python3
"""
NAI + Local Diffusers Image Generator Backend
Standalone version for Tauri sidecar
"""
import os
import sys
import io
import re
import base64
import json
import asyncio
import datetime
import math
import random
import ctypes
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import piexif
import piexif.helper

# 경로 설정 (앱 디렉토리 기준)
if getattr(sys, 'frozen', False):
    # PyInstaller로 빌드된 경우
    APP_DIR = Path(sys.executable).parent
else:
    APP_DIR = Path(__file__).parent

# local_engine 모듈 경로 추가
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

MODELS_DIR = APP_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LORA_DIR = MODELS_DIR / "loras"
UPSCALE_DIR = MODELS_DIR / "upscale_models"
OUTPUT_DIR = APP_DIR / "outputs"
PRESETS_DIR = APP_DIR / "presets"
PROMPTS_DIR = APP_DIR / "prompts"
DATA_DIR = APP_DIR / "data"  # 데이터 파일 (tags.json 등)
CONFIG_FILE = APP_DIR / "config.json"
PYTHON_ENV_DIR = APP_DIR / "python"  # Python 환경 (빌드와 동일 경로)
CENSOR_MODELS_DIR = MODELS_DIR / "censor"  # 검열 모델 (YOLO)
CENSORING_DIR = APP_DIR / "censoring"  # 검열 작업 폴더 (레거시, 더 이상 사용 안함)
UNCENSORED_DIR = CENSORING_DIR / "uncensored"  # 검열 전 이미지 (레거시, 더 이상 사용 안함)
CENSORED_DIR = OUTPUT_DIR / "censored"  # 검열 후 이미지 (새 경로: outputs/censored)
LOGS_DIR = APP_DIR / "logs"
VERSION_FILE = APP_DIR / "version.json"

# 로그 디렉토리 생성
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 버전 정보 로드
def load_version():
    """version.json에서 버전 정보 로드"""
    if VERSION_FILE.exists():
        try:
            return json.loads(VERSION_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {"version": "1.0.0", "build_date": "unknown"}

APP_VERSION = load_version()

# 로깅 설정
def setup_logging():
    """파일 및 콘솔 로깅 설정"""
    log_file = LOGS_DIR / f"peropix_{datetime.datetime.now().strftime('%Y%m%d')}.log"

    # 로그 포맷
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 파일 핸들러 (5MB 제한, 최대 5개 파일 보관)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)

    return logging.getLogger('peropix')

# 로거 초기화
logger = setup_logging()
logger.info(f"PeroPix v{APP_VERSION['version']} 시작")

# 카테고리별 이미지 순번 - 매번 실제 폴더 스캔
def get_next_image_number(category: str, save_dir: Path = None, ext: str = 'png') -> int:
    """해당 카테고리의 다음 순번 반환 - 실제 폴더에서 스캔 (모든 확장자 검색)"""
    if save_dir is None:
        save_dir = OUTPUT_DIR

    max_num = 0
    # 모든 지원 포맷에서 순번 검색 (포맷 변경 시에도 연속 번호 유지)
    pattern = re.compile(rf'^{re.escape(category)}_(\d{{7}})\.(png|jpg|webp)$')

    if save_dir.exists():
        for f in save_dir.iterdir():
            try:
                if f.suffix.lower() in ['.png', '.jpg', '.webp']:
                    match = pattern.match(f.name)
                    if match:
                        num = int(match.group(1))
                        if num > max_num:
                            max_num = num
            except Exception:
                pass

    return max_num + 1

# 디렉토리 생성
for d in [CHECKPOINTS_DIR, LORA_DIR, UPSCALE_DIR, OUTPUT_DIR, PRESETS_DIR, DATA_DIR, CENSOR_MODELS_DIR, CENSORED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Prompts 하위 폴더 생성
for subdir in ['base', 'negative', 'character']:
    (PROMPTS_DIR / subdir).mkdir(parents=True, exist_ok=True)

# Config 관리
def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
    return {"nai_token": "", "checkpoints_dir": str(CHECKPOINTS_DIR), "lora_dir": str(LORA_DIR)}

def save_config(config):
    CONFIG_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')

CONFIG = load_config()

# ============================================================
# Imports (lazy loading for local generation only)
# ============================================================
from PIL import Image  # PIL은 NAI에서도 필요하므로 바로 import

torch = None
np = None

def lazy_imports():
    """Local generation 전용 - torch, numpy 로드"""
    global torch, np
    if torch is None:
        import torch as _torch
        import numpy as _np
        torch = _torch
        np = _np

# ============================================================
# Character Reference Helper Functions
# ============================================================
ACCEPTED_CR_SIZES = [(1024, 1536), (1536, 1024), (1472, 1472)]

def _choose_cr_canvas(w: int, h: int) -> tuple:
    """캐릭터 레퍼런스용 캔버스 크기 선택 (원본 비율에 가장 가까운 것)"""
    aspect = w / h
    best = None
    best_diff = float('inf')
    for cw, ch in ACCEPTED_CR_SIZES:
        diff = abs((cw / ch) - aspect)
        if diff < best_diff:
            best_diff = diff
            best = (cw, ch)
    return best

def decode_base64_image(base64_image: str):
    """base64 문자열을 PIL Image로 디코딩"""
    from PIL import Image as PILImage
    image_data = base64.b64decode(base64_image)
    return PILImage.open(io.BytesIO(image_data))


def pad_image_to_canvas_base64(base64_image: str, target_size: tuple) -> str:
    """base64 이미지를 캔버스 크기에 맞게 letterbox 패딩 후 base64 반환 (NAIS2 방식: JPEG 95%)"""
    from PIL import Image as PILImage

    # base64 디코딩
    image_data = base64.b64decode(base64_image)
    pil_img = PILImage.open(io.BytesIO(image_data))

    # JPEG는 알파 채널 미지원, RGB로 변환
    if pil_img.mode == 'RGBA':
        # 알파 채널이 있으면 검은 배경에 합성
        background = PILImage.new('RGB', pil_img.size, (0, 0, 0))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    elif pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    W, H = pil_img.size
    tw, th = target_size

    # 비율 유지하면서 리사이즈
    import math
    scale = min(tw / W, th / H)
    new_w = min(tw, max(1, math.ceil(W * scale)))
    new_h = min(th, max(1, math.ceil(H * scale)))
    # BILINEAR (브라우저 Canvas drawImage와 유사)
    pil_resized = pil_img.resize((new_w, new_h), PILImage.BILINEAR)

    # 검은 캔버스에 중앙 배치
    canvas = PILImage.new('RGB', (tw, th), (0, 0, 0))
    offset = ((tw - new_w) // 2, (th - new_h) // 2)
    canvas.paste(pil_resized, offset)

    # NAIS2 방식: JPEG 95% 품질
    buffer = io.BytesIO()
    canvas.save(buffer, format='JPEG', quality=95)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_image_size_from_base64(base64_image: str) -> tuple:
    """base64 이미지의 크기 반환 (width, height)"""
    from PIL import Image as PILImage
    image_data = base64.b64decode(base64_image)
    pil_img = PILImage.open(io.BytesIO(image_data))
    return pil_img.size

def resize_image_base64(base64_image: str, max_size: int = 1024) -> str:
    """base64 이미지를 최대 크기로 리사이즈 (비율 유지)"""
    from PIL import Image as PILImage
    
    image_data = base64.b64decode(base64_image)
    pil_img = PILImage.open(io.BytesIO(image_data))
    
    # RGB로 변환
    if pil_img.mode == 'RGBA':
        # RGBA면 흰 배경에 합성
        background = PILImage.new('RGB', pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    elif pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    W, H = pil_img.size
    
    # 이미 작으면 그대로
    if W <= max_size and H <= max_size:
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 비율 유지하면서 리사이즈
    scale = min(max_size / W, max_size / H)
    new_w = int(W * scale)
    new_h = int(H * scale)
    pil_resized = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
    
    buffer = io.BytesIO()
    pil_resized.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def resize_image_to_size_base64(base64_image: str, target_width: int, target_height: int) -> str:
    """base64 이미지를 지정된 크기로 리사이즈 (Vibe Transfer용) - PNG 포맷"""
    from PIL import Image as PILImage

    image_data = base64.b64decode(base64_image)
    pil_img = PILImage.open(io.BytesIO(image_data))

    # RGB로 변환
    if pil_img.mode == 'RGBA':
        background = PILImage.new('RGB', pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    elif pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    # 지정된 크기로 리사이즈
    pil_resized = pil_img.resize((target_width, target_height), PILImage.LANCZOS)

    # PNG로 저장 (bedovyy 방식)
    buffer = io.BytesIO()
    pil_resized.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def ensure_png_base64(base64_image: str, force_reencode: bool = False) -> str:
    """base64 이미지를 PNG 포맷으로 변환 (필요한 경우에만 재인코딩)

    RGBA 이미지는 그대로 유지 (NAI API가 RGBA 지원, 바이브 인코딩 호환성)
    """
    from PIL import Image as PILImage

    image_data = base64.b64decode(base64_image)
    pil_img = PILImage.open(io.BytesIO(image_data))

    # 이미 PNG이고 RGB 또는 RGBA이면 원본 그대로 반환
    if not force_reencode and pil_img.format == 'PNG' and pil_img.mode in ('RGB', 'RGBA'):
        return base64_image

    # PNG가 아니거나 다른 모드인 경우에만 변환
    if pil_img.mode not in ('RGB', 'RGBA'):
        pil_img = pil_img.convert('RGB')

    # PNG로 저장
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def binarize_mask(base64_mask: str, threshold: int = 1) -> str:
    """마스크를 이진화하고 원본 PNG 형식 유지

    PIL 재인코딩이 NAI와 호환 안될 수 있으므로,
    이미 순수 흑백이면 원본 그대로 반환.
    회색 픽셀이 있으면 이진화 후 반환.
    """
    from PIL import Image as PILImage

    # 마스크 로드
    mask_data = base64.b64decode(base64_mask)
    mask_img = PILImage.open(io.BytesIO(mask_data))

    # 픽셀값 확인
    if mask_img.mode == 'RGBA':
        # R 채널만 확인 (흑백이면 R=G=B)
        r_channel = list(mask_img.split()[0].getdata())
        unique_values = set(r_channel)
    else:
        gray = mask_img.convert('L')
        unique_values = set(gray.getdata())

    # 이미 순수 흑백(0, 255만 있음)이면 원본 그대로 반환
    if unique_values <= {0, 255}:
        return base64_mask

    # 회색 픽셀이 있으면 이진화 필요
    mask_gray = mask_img.convert('L')
    mask_binary = mask_gray.point(lambda x: 255 if x >= threshold else 0, mode='L')
    alpha = PILImage.new('L', mask_binary.size, 255)
    mask_rgba = PILImage.merge('RGBA', (mask_binary, mask_binary, mask_binary, alpha))

    # PNG로 저장
    buffer = io.BytesIO()
    mask_rgba.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ============================================================
# Windows Explorer 폴더 열기 (포커스 포함)
# ============================================================

def _force_foreground_window(hwnd) -> bool:
    """Windows 제한을 우회하여 창을 강제로 foreground로 가져오기"""
    try:
        import win32gui
        import win32process
        import win32con
        import ctypes
        
        # 최소화 상태면 복원
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        
        # 이미 foreground면 완료
        if win32gui.GetForegroundWindow() == hwnd:
            return True
        
        # 현재 foreground 창의 스레드와 대상 창의 스레드 ID 획득
        foreground_hwnd = win32gui.GetForegroundWindow()
        foreground_thread = win32process.GetWindowThreadProcessId(foreground_hwnd)[0]
        target_thread = win32process.GetWindowThreadProcessId(hwnd)[0]
        
        # 스레드가 다르면 AttachThreadInput으로 권한 우회
        if foreground_thread != target_thread:
            ctypes.windll.user32.AttachThreadInput(target_thread, foreground_thread, True)
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)
            ctypes.windll.user32.AttachThreadInput(target_thread, foreground_thread, False)
        else:
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetForegroundWindow(hwnd)
        
        return True
    except Exception as e:
        print(f"[ForceForeground] Error: {e}")
        return False


def open_folder_in_explorer(path: str) -> bool:
    """폴더를 파일 탐색기로 열기 (크로스 플랫폼)
    
    Windows:
    - 이미 열린 폴더면 기존 창을 foreground로 활성화
    - 새 폴더면 열고 foreground로 가져옴
    """
    import platform
    import subprocess
    
    try:
        path = os.path.normpath(path)
        system = platform.system()
        
        if system == "Windows":
            try:
                import win32com.client
                import urllib.parse
                
                # Shell.Application으로 열린 탐색기 창 검색
                shell = win32com.client.Dispatch("Shell.Application")
                windows = shell.Windows()
                
                # 이미 열린 폴더인지 확인
                for window in windows:
                    try:
                        # LocationURL에서 경로 추출 (file:///C:/path 형식)
                        location = str(window.LocationURL)
                        if location.startswith("file:///"):
                            window_path = location[8:].replace("/", "\\")  # file:/// 제거
                            window_path = urllib.parse.unquote(window_path)
                            
                            if os.path.normcase(window_path) == os.path.normcase(path):
                                # 이미 열려있음 - 해당 창을 foreground로
                                _force_foreground_window(window.HWND)
                                return True
                    except:
                        continue
                
                # 열려있지 않음 - 새로 열기
                os.startfile(path)
                
                # 새로 열린 창을 foreground로 가져오기 (약간의 딜레이 필요)
                import time
                time.sleep(0.3)
                
                # 다시 창 목록에서 찾아서 foreground로
                windows = shell.Windows()
                for window in windows:
                    try:
                        location = str(window.LocationURL)
                        if location.startswith("file:///"):
                            window_path = location[8:].replace("/", "\\")
                            window_path = urllib.parse.unquote(window_path)
                            if os.path.normcase(window_path) == os.path.normcase(path):
                                _force_foreground_window(window.HWND)
                                break
                    except:
                        continue
                
                return True
                
            except ImportError:
                # pywin32 없으면 기본 방식
                os.startfile(path)
                return True
                
        elif system == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except Exception as e:
        print(f"[OpenFolder] Error: {e}")
        return False


# ============================================================
# EXIF 메타데이터 헬퍼 함수
# ============================================================

def save_metadata_to_exif(image_bytes: bytes, metadata_dict: dict, image_format: str = 'PNG') -> bytes:
    """이미지에 EXIF UserComment로 메타데이터 저장

    Args:
        image_bytes: 원본 이미지 바이트
        metadata_dict: 저장할 메타데이터 딕셔너리
        image_format: 이미지 포맷 (PNG, JPEG, WEBP)

    Returns:
        메타데이터가 포함된 이미지 바이트
    """
    from PIL import Image as PILImage

    try:
        # JSON으로 직렬화
        metadata_json = json.dumps(metadata_dict, ensure_ascii=False)

        # EXIF 데이터 생성
        # UserComment에 저장 (UTF-8 인코딩)
        user_comment = piexif.helper.UserComment.dump(metadata_json, encoding="unicode")

        exif_dict = {
            "0th": {},
            "Exif": {
                piexif.ExifIFD.UserComment: user_comment
            },
            "GPS": {},
            "1st": {},
            "thumbnail": None
        }

        exif_bytes = piexif.dump(exif_dict)

        # 이미지에 EXIF 삽입
        img = PILImage.open(io.BytesIO(image_bytes))

        output = io.BytesIO()

        if image_format.upper() == 'PNG':
            # PNG: EXIF + PNG tEXt 청크 둘 다 저장 (NAI 호환성)
            from PIL.PngImagePlugin import PngInfo
            pnginfo = PngInfo()
            pnginfo.add_text("Comment", metadata_json)

            # PNG는 EXIF를 직접 지원하지 않으므로 tEXt로만 저장
            img.save(output, format='PNG', pnginfo=pnginfo)
        elif image_format.upper() in ('JPEG', 'JPG'):
            # JPEG: EXIF 직접 지원
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.save(output, format='JPEG', exif=exif_bytes, quality=95)
        elif image_format.upper() == 'WEBP':
            # WebP: EXIF 직접 지원
            img.save(output, format='WEBP', exif=exif_bytes, quality=95)
        else:
            # 기타 포맷은 메타데이터 없이 저장
            img.save(output, format=image_format)

        return output.getvalue()

    except Exception as e:
        print(f"[EXIF] Error saving metadata: {e}")
        # 실패시 원본 반환
        return image_bytes


def read_metadata_from_image(image_bytes: bytes) -> dict:
    """이미지에서 메타데이터 읽기 (EXIF UserComment 또는 PNG Comment)

    Args:
        image_bytes: 이미지 바이트

    Returns:
        메타데이터 딕셔너리 (없으면 빈 딕셔너리)
    """
    from PIL import Image as PILImage

    metadata = None

    try:
        img = PILImage.open(io.BytesIO(image_bytes))

        # 1. PNG Comment 확인 (NAI 호환)
        if hasattr(img, 'info') and 'Comment' in img.info:
            try:
                comment = img.info['Comment']
                if isinstance(comment, bytes):
                    comment = comment.decode('utf-8')
                metadata = json.loads(comment)
                return metadata
            except:
                pass

        # 2. 레거시 peropix 필드 확인
        if hasattr(img, 'info') and 'peropix' in img.info:
            try:
                legacy = json.loads(img.info['peropix'])
                # 레거시 형식을 NAI 호환 형식으로 변환
                metadata = {
                    "prompt": legacy.get("prompt", ""),
                    "uc": legacy.get("negative_prompt", ""),
                    "seed": legacy.get("seed"),
                    "width": legacy.get("width"),
                    "height": legacy.get("height"),
                    "steps": legacy.get("steps"),
                    "scale": legacy.get("cfg"),
                    "sampler": legacy.get("sampler"),
                    "noise_schedule": legacy.get("scheduler"),
                    "request_type": legacy.get("nai_model"),
                    "ucPreset": legacy.get("uc_preset"),
                    "qualityToggle": legacy.get("quality_tags"),
                    "cfg_rescale": legacy.get("cfg_rescale"),
                    "peropix": {
                        "version": 0,
                        "provider": legacy.get("provider", "nai"),
                        "character_prompts": legacy.get("character_prompts", []),
                        "variety_plus": legacy.get("variety_plus", False),
                        "furry_mode": legacy.get("furry_mode", False),
                        "local_model": legacy.get("model", "")
                    }
                }
                return metadata
            except:
                pass

        # 3. EXIF UserComment 확인 - getexif() 사용 (Pillow 9.0+, WebP/JPEG 모두 지원)
        try:
            exif = img.getexif()
            if exif:
                # UserComment는 EXIF IFD에 있으므로 get_ifd로 접근
                exif_ifd = exif.get_ifd(0x8769)  # ExifIFD
                if exif_ifd and 0x9286 in exif_ifd:  # UserComment tag
                    user_comment_bytes = exif_ifd[0x9286]
                    user_comment = piexif.helper.UserComment.load(user_comment_bytes)
                    metadata = json.loads(user_comment)
                    return metadata
        except Exception as e:
            pass

        # 4. piexif로 직접 읽기 시도 (JPEG 호환)
        try:
            exif_dict = piexif.load(image_bytes)
            if "Exif" in exif_dict and piexif.ExifIFD.UserComment in exif_dict["Exif"]:
                user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
                metadata = json.loads(user_comment)
                return metadata
        except:
            pass

    except Exception as e:
        print(f"[EXIF] Error reading metadata: {e}")

    return metadata or {}


# ============================================================
# LyCORIS Loader (LokR/LoHa support)
# Diffusers가 지원하지 않는 LyCORIS 형식을 직접 처리
# 참고: ComfyUI weight_adapter 구현
# ============================================================
class LyCORISLoader:
    """LyCORIS (LokR/LoHa) 전용 로더 - Diffusers와 독립적으로 동작"""

    SUPPORTED_TYPES = {'lokr', 'loha'}

    @staticmethod
    def detect_type(state_dict: dict) -> str | None:
        """
        LoRA 파일의 타입 감지
        Returns: 'lokr', 'loha', or None (일반 LoRA)
        """
        for key in state_dict.keys():
            # LokR 감지: lokr_w1, lokr_w2, lokr_w1_a 등
            if '.lokr_w1' in key or '.lokr_w2' in key:
                return 'lokr'
            # LoHa 감지: hada_w1_a, hada_w1_b 등
            if '.hada_w1_a' in key or '.hada_w2_a' in key:
                return 'loha'
        return None  # 일반 LoRA

    @staticmethod
    def get_layer_name_mapping(state_dict: dict, model_type: str = 'sdxl') -> dict:
        """
        LyCORIS 키를 모델 레이어 이름으로 매핑
        Returns: {lycoris_prefix: model_layer_name}
        """
        # LyCORIS 키에서 고유 prefix 추출
        prefixes = set()
        for key in state_dict.keys():
            # 예: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lokr_w1
            # prefix: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q
            for suffix in ['.lokr_w1', '.lokr_w2', '.lokr_w1_a', '.lokr_w1_b',
                          '.lokr_w2_a', '.lokr_w2_b', '.lokr_t2', '.alpha',
                          '.hada_w1_a', '.hada_w1_b', '.hada_w2_a', '.hada_w2_b',
                          '.hada_t1', '.hada_t2']:
                if key.endswith(suffix):
                    prefixes.add(key[:-len(suffix)])
                    break

        # prefix를 모델 레이어 이름으로 변환
        mapping = {}
        for prefix in prefixes:
            # lora_unet_xxx -> unet.xxx (언더스코어를 점으로)
            # lora_te1_xxx -> text_encoder.xxx
            # lora_te2_xxx -> text_encoder_2.xxx
            layer_name = prefix

            if layer_name.startswith('lora_unet_'):
                layer_name = 'unet.' + layer_name[10:].replace('_', '.')
            elif layer_name.startswith('lora_te1_'):
                layer_name = 'text_encoder.' + layer_name[9:].replace('_', '.')
            elif layer_name.startswith('lora_te2_'):
                layer_name = 'text_encoder_2.' + layer_name[9:].replace('_', '.')
            elif layer_name.startswith('lora_te_'):
                layer_name = 'text_encoder.' + layer_name[8:].replace('_', '.')

            # 숫자 패턴 수정: down.blocks.0 -> down_blocks.0 등
            # Diffusers 모델 구조에 맞게 변환
            import re
            # blocks.0.attentions -> blocks[0].attentions 스타일이 아닌
            # down_blocks.0 형태 유지
            layer_name = re.sub(r'\.(\d+)\.', r'[\1].', layer_name)

            mapping[prefix] = layer_name

        return mapping

    @staticmethod
    def calculate_lokr_weight(
        state_dict: dict,
        prefix: str,
        device: str,
        dtype
    ):
        """
        LokR weight delta 계산
        delta = kron(w1, w2) * (alpha / dim)
        """
        lazy_imports()

        # weight 키 이름들
        w1_key = f"{prefix}.lokr_w1"
        w2_key = f"{prefix}.lokr_w2"
        w1_a_key = f"{prefix}.lokr_w1_a"
        w1_b_key = f"{prefix}.lokr_w1_b"
        w2_a_key = f"{prefix}.lokr_w2_a"
        w2_b_key = f"{prefix}.lokr_w2_b"
        t2_key = f"{prefix}.lokr_t2"
        alpha_key = f"{prefix}.alpha"

        # alpha 값
        alpha = state_dict.get(alpha_key, torch.tensor(1.0)).item() if alpha_key in state_dict else 1.0

        # w1 계산
        if w1_key in state_dict:
            w1 = state_dict[w1_key].to(device=device, dtype=dtype)
            dim = w1.shape[0]  # 또는 다른 차원
        elif w1_a_key in state_dict and w1_b_key in state_dict:
            w1_a = state_dict[w1_a_key].to(device=device, dtype=dtype)
            w1_b = state_dict[w1_b_key].to(device=device, dtype=dtype)
            w1 = torch.mm(w1_a, w1_b)
            dim = w1_b.shape[0]
        else:
            return None

        # w2 계산
        if w2_key in state_dict:
            w2 = state_dict[w2_key].to(device=device, dtype=dtype)
        elif w2_a_key in state_dict and w2_b_key in state_dict:
            w2_a = state_dict[w2_a_key].to(device=device, dtype=dtype)
            w2_b = state_dict[w2_b_key].to(device=device, dtype=dtype)

            # Tucker decomposition 체크
            if t2_key in state_dict:
                t2 = state_dict[t2_key].to(device=device, dtype=dtype)
                w2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2_b, w2_a)
            else:
                w2 = torch.mm(w2_a, w2_b)
            dim = w2_b.shape[0]
        else:
            return None

        # Kronecker product 계산
        # Conv2d의 경우 4D 처리
        if len(w2.shape) == 4:
            w1 = w1.unsqueeze(2).unsqueeze(2)

        # alpha 스케일링
        scale = alpha / dim if dim > 0 else 1.0

        delta = torch.kron(w1, w2) * scale
        return delta

    @staticmethod
    def calculate_loha_weight(
        state_dict: dict,
        prefix: str,
        device: str,
        dtype
    ):
        """
        LoHa weight delta 계산
        delta = (w1_a @ w1_b) * (w2_a @ w2_b) * (alpha / dim)
        Hadamard product (element-wise multiplication)
        """
        lazy_imports()

        # weight 키 이름들
        w1_a_key = f"{prefix}.hada_w1_a"
        w1_b_key = f"{prefix}.hada_w1_b"
        w2_a_key = f"{prefix}.hada_w2_a"
        w2_b_key = f"{prefix}.hada_w2_b"
        t1_key = f"{prefix}.hada_t1"
        t2_key = f"{prefix}.hada_t2"
        alpha_key = f"{prefix}.alpha"

        # 필수 키 체크
        if w1_a_key not in state_dict or w1_b_key not in state_dict:
            return None
        if w2_a_key not in state_dict or w2_b_key not in state_dict:
            return None

        # alpha 값
        alpha = state_dict.get(alpha_key, torch.tensor(1.0)).item() if alpha_key in state_dict else 1.0

        w1_a = state_dict[w1_a_key].to(device=device, dtype=dtype)
        w1_b = state_dict[w1_b_key].to(device=device, dtype=dtype)
        w2_a = state_dict[w2_a_key].to(device=device, dtype=dtype)
        w2_b = state_dict[w2_b_key].to(device=device, dtype=dtype)

        dim = w1_b.shape[0]

        # Tucker decomposition 체크
        if t1_key in state_dict and t2_key in state_dict:
            t1 = state_dict[t1_key].to(device=device, dtype=dtype)
            t2 = state_dict[t2_key].to(device=device, dtype=dtype)
            m1 = torch.einsum('i j k l, j r, i p -> p r k l', t1, w1_b, w1_a)
            m2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2_b, w2_a)
        else:
            m1 = torch.mm(w1_a, w1_b)
            m2 = torch.mm(w2_a, w2_b)

        # Hadamard product (element-wise)
        scale = alpha / dim if dim > 0 else 1.0
        delta = (m1 * m2) * scale

        return delta


class ModelCache:
    def __init__(self):
        self.pipe = None
        self.model_path = None
        self.loaded_loras = {}  # {name: {'type': 'lora'|'lokr'|'loha', 'scale': float, ...}}
        self._device = None
        self._dtype = None
        # LyCORIS 적용을 위한 원본 weight 백업
        self._original_weights = {}  # {layer_name: original_weight_tensor}
        self._lycoris_deltas = {}    # {lora_name: {layer_name: delta_tensor}}
    
    @property
    def device(self):
        if self._device is None:
            lazy_imports()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device
    
    @property
    def dtype(self):
        if self._dtype is None:
            lazy_imports()
            # bfloat16 on Ampere+ (compute capability >= 8.0) for better VAE stability
            if self.device == "cuda":
                try:
                    cc = torch.cuda.get_device_capability()
                    if cc[0] >= 8:  # Ampere (RTX 30xx) or newer
                        self._dtype = torch.bfloat16
                        print(f"[GPU] Using bfloat16 (compute capability {cc[0]}.{cc[1]})")
                    else:
                        self._dtype = torch.float16
                except:
                    self._dtype = torch.float16
            else:
                self._dtype = torch.float32
        return self._dtype
    
    def load_model(self, model_path: str):
        lazy_imports()
        
        if self.model_path == model_path and self.pipe is not None:
            print(f"[Cache] Using cached model: {model_path}")
            return self.pipe
        
        print(f"[Load] Loading model: {model_path}")
        
        if self.pipe is not None:
            del self.pipe
            torch.cuda.empty_cache()
        
        from diffusers import StableDiffusionXLPipeline
        
        checkpoints_dir = Path(CONFIG.get("checkpoints_dir", CHECKPOINTS_DIR))
        full_path = checkpoints_dir / model_path
        
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            str(full_path),
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        self.pipe.to(self.device)

        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_tiling()
            # xformers for memory efficient attention
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
        
        self.model_path = model_path
        self.loaded_loras = {}
        
        print(f"[Load] Model loaded: {model_path}")
        return self.pipe
    
    def load_lora(self, lora_name: str, scale: float = 1.0):
        if self.pipe is None:
            raise ValueError("Model not loaded")

        if lora_name in self.loaded_loras:
            return

        lora_dir = Path(CONFIG.get("lora_dir", LORA_DIR))
        lora_path = lora_dir / lora_name

        if not lora_path.exists():
            raise ValueError(f"LoRA not found: {lora_path}")

        # safetensors 파일 로드하여 타입 감지
        from safetensors.torch import load_file
        state_dict = load_file(str(lora_path))
        lora_type = LyCORISLoader.detect_type(state_dict)

        if lora_type in ('lokr', 'loha'):
            # LyCORIS 형식 - 직접 처리
            print(f"[Load] Loading {lora_type.upper()}: {lora_name} (scale={scale})")
            self._load_lycoris(lora_name, state_dict, lora_type, scale)
        else:
            # 일반 LoRA - Diffusers 방식
            # adapter_name에 '.'이 포함되면 PEFT에서 모듈 경로 파싱 오류 발생
            safe_adapter_name = lora_name.replace(".", "_")
            print(f"[Load] Loading LoRA: {lora_name} (scale={scale})")
            self.pipe.load_lora_weights(str(lora_path), adapter_name=safe_adapter_name)
            self.loaded_loras[lora_name] = {'type': 'lora', 'scale': scale}

    def _load_lycoris(self, lora_name: str, state_dict: dict, lora_type: str, scale: float):
        """LyCORIS (LokR/LoHa) weight를 모델에 직접 적용"""
        # delta 계산 함수 선택
        if lora_type == 'lokr':
            calc_fn = LyCORISLoader.calculate_lokr_weight
        else:  # loha
            calc_fn = LyCORISLoader.calculate_loha_weight

        # LyCORIS 키에서 prefix 추출
        prefixes = set()
        for key in state_dict.keys():
            for suffix in ['.lokr_w1', '.lokr_w2', '.lokr_w1_a', '.lokr_w1_b',
                          '.lokr_w2_a', '.lokr_w2_b', '.lokr_t2', '.alpha',
                          '.hada_w1_a', '.hada_w1_b', '.hada_w2_a', '.hada_w2_b',
                          '.hada_t1', '.hada_t2']:
                if key.endswith(suffix):
                    prefixes.add(key[:-len(suffix)])
                    break

        # 각 prefix에 대해 delta 계산 및 저장
        deltas = {}
        applied_count = 0
        skipped_count = 0

        for prefix in prefixes:
            delta = calc_fn(state_dict, prefix, self.device, self.dtype)
            if delta is not None:
                deltas[prefix] = delta
                applied_count += 1
            else:
                skipped_count += 1

        self._lycoris_deltas[lora_name] = deltas
        self.loaded_loras[lora_name] = {
            'type': lora_type,
            'scale': scale,
            'prefixes': list(prefixes)
        }

        print(f"[Load] {lora_type.upper()} loaded: {applied_count} layers, {skipped_count} skipped")

    def _apply_lycoris_to_model(self):
        """저장된 LyCORIS delta들을 모델에 적용"""
        if not self._lycoris_deltas:
            return

        # UNet과 Text Encoder의 state_dict 가져오기
        unet_sd = self.pipe.unet.state_dict()
        te1_sd = self.pipe.text_encoder.state_dict() if hasattr(self.pipe, 'text_encoder') else {}
        te2_sd = self.pipe.text_encoder_2.state_dict() if hasattr(self.pipe, 'text_encoder_2') else {}

        for lora_name, deltas in self._lycoris_deltas.items():
            lora_info = self.loaded_loras.get(lora_name, {})
            scale = lora_info.get('scale', 1.0)

            for prefix, delta in deltas.items():
                # prefix를 실제 weight 키로 변환
                weight_key = self._lycoris_prefix_to_weight_key(prefix)
                if weight_key is None:
                    continue

                # 해당 모듈 찾기 및 weight 업데이트
                try:
                    self._apply_delta_to_weight(prefix, weight_key, delta, scale)
                except Exception as e:
                    print(f"[LyCORIS] Failed to apply {prefix}: {e}")

    def _lycoris_prefix_to_weight_key(self, prefix: str) -> str | None:
        """LyCORIS prefix를 모델 weight 키로 변환"""
        # lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q
        # -> down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight

        if prefix.startswith('lora_unet_'):
            key = prefix[10:]  # 'lora_unet_' 제거
        elif prefix.startswith('lora_te1_'):
            key = prefix[9:]
        elif prefix.startswith('lora_te2_'):
            key = prefix[9:]
        elif prefix.startswith('lora_te_'):
            key = prefix[8:]
        else:
            return None

        # 언더스코어를 점으로 변환하되, 숫자 앞뒤는 유지
        import re
        # 패턴: word_word -> word.word, word_0_word -> word.0.word
        parts = key.split('_')
        result = []
        i = 0
        while i < len(parts):
            part = parts[i]
            result.append(part)
            i += 1

        return '.'.join(result) + '.weight'

    def _apply_delta_to_weight(self, prefix: str, weight_key: str, delta, scale: float):
        """delta를 실제 모델 weight에 적용"""
        # 어떤 모듈에 적용할지 결정
        if prefix.startswith('lora_unet_'):
            module = self.pipe.unet
            key_in_module = weight_key
        elif prefix.startswith('lora_te1_') or prefix.startswith('lora_te_'):
            module = self.pipe.text_encoder
            key_in_module = weight_key
        elif prefix.startswith('lora_te2_'):
            module = self.pipe.text_encoder_2
            key_in_module = weight_key
        else:
            return

        # 해당 레이어 찾기
        try:
            # nested module 접근
            parts = key_in_module.replace('.weight', '').split('.')
            target = module
            for part in parts:
                if part.isdigit():
                    target = target[int(part)]
                else:
                    target = getattr(target, part)

            # 원본 백업 (최초 1회)
            full_key = f"{prefix}.weight"
            if full_key not in self._original_weights:
                self._original_weights[full_key] = target.weight.data.clone()

            # delta 적용
            scaled_delta = delta * scale
            if scaled_delta.shape != target.weight.data.shape:
                scaled_delta = scaled_delta.reshape(target.weight.data.shape)

            target.weight.data = self._original_weights[full_key] + scaled_delta.to(target.weight.data.dtype)

        except Exception as e:
            # 매핑 실패 - 디버그용 로그
            print(f"[LyCORIS] Could not find layer for {prefix}: {e}")
    
    def set_lora_scales(self, lora_configs: List[dict]):
        if not lora_configs:
            # 모든 LoRA 언로드
            self._unload_all_loras()
            return

        # 일반 LoRA와 LyCORIS 분리
        diffusers_adapters = []  # (name, scale) for Diffusers
        lycoris_configs = []     # (name, scale) for LyCORIS

        for config in lora_configs:
            name = config["name"]
            scale = config.get("scale", 1.0)

            if name not in self.loaded_loras:
                self.load_lora(name, scale)

            lora_info = self.loaded_loras.get(name, {})
            lora_type = lora_info.get('type', 'lora')

            if lora_type in ('lokr', 'loha'):
                lycoris_configs.append((name, scale))
                # scale 변경 시 업데이트
                if lora_info.get('scale') != scale:
                    self.loaded_loras[name]['scale'] = scale
            else:
                safe_adapter_name = name.replace(".", "_")
                diffusers_adapters.append((safe_adapter_name, scale))

        # Diffusers LoRA 적용
        if diffusers_adapters:
            adapter_names = [a[0] for a in diffusers_adapters]
            adapter_weights = [a[1] for a in diffusers_adapters]
            self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

        # LyCORIS 적용 (scale 변경 포함)
        if lycoris_configs:
            self._apply_lycoris_to_model()

    def _unload_all_loras(self):
        """모든 LoRA 및 LyCORIS 언로드"""
        # Diffusers LoRA 언로드
        has_diffusers_lora = any(
            info.get('type') == 'lora'
            for info in self.loaded_loras.values()
            if isinstance(info, dict)
        )
        if has_diffusers_lora:
            try:
                self.pipe.unload_lora_weights()
            except Exception as e:
                print(f"[LoRA] Unload warning: {e}")

        # LyCORIS weight 복원
        self._restore_original_weights()

        # 상태 초기화
        self.loaded_loras = {}
        self._lycoris_deltas = {}

    def _restore_original_weights(self):
        """LyCORIS로 변경된 weight를 원본으로 복원"""
        if not self._original_weights:
            return

        restored_count = 0
        for full_key, original_weight in self._original_weights.items():
            # full_key: "lora_unet_xxx.weight" 형태
            prefix = full_key.replace('.weight', '')

            try:
                if prefix.startswith('lora_unet_'):
                    module = self.pipe.unet
                elif prefix.startswith('lora_te1_') or prefix.startswith('lora_te_'):
                    module = self.pipe.text_encoder
                elif prefix.startswith('lora_te2_'):
                    module = self.pipe.text_encoder_2
                else:
                    continue

                # weight 키 변환 및 레이어 접근
                weight_key = self._lycoris_prefix_to_weight_key(prefix)
                if weight_key is None:
                    continue

                parts = weight_key.replace('.weight', '').split('.')
                target = module
                for part in parts:
                    if part.isdigit():
                        target = target[int(part)]
                    else:
                        target = getattr(target, part)

                target.weight.data = original_weight.clone()
                restored_count += 1

            except Exception as e:
                print(f"[LyCORIS] Restore failed for {prefix}: {e}")

        if restored_count > 0:
            print(f"[LyCORIS] Restored {restored_count} original weights")

        self._original_weights = {}

    def clear(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.model_path = None
        self.loaded_loras = {}
        self._original_weights = {}
        self._lycoris_deltas = {}
        if torch is not None:
            torch.cuda.empty_cache()


model_cache = ModelCache()


# ============================================================
# Upscale Model Cache
# ============================================================
class UpscaleModelCache:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.scale = 2  # 기본값
        self.dtype = None  # lazy init
    
    def load_model(self, model_name: str):
        lazy_imports()
        
        if self.model_name == model_name and self.model is not None:
            print(f"[Cache] Using cached upscale model: {model_name}")
            # 캐시된 모델이 CPU에 있을 수 있으므로 GPU로 이동
            self.model.to(model_cache.device)
            return self.model
        
        print(f"[Load] Loading upscale model: {model_name}")
        
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        
        from spandrel import ModelLoader
        
        upscale_path = UPSCALE_DIR / model_name
        if not upscale_path.exists():
            raise ValueError(f"Upscale model not found: {upscale_path}")
        
        model_desc = ModelLoader().load_from_file(str(upscale_path))
        self.scale = model_desc.scale if hasattr(model_desc, 'scale') else 2
        
        self.model = model_desc.to(model_cache.device)
        
        # fp16 지원 여부 확인 후 변환
        if model_cache.device == "cuda":
            try:
                self.model = self.model.half()
                self.dtype = torch.float16
            except Exception:
                # fp16 미지원 모델은 fp32 유지
                self.dtype = torch.float32
                print(f"[Load] Model doesn't support fp16, using fp32")
        else:
            self.dtype = torch.float32
        
        self.model.eval()
        
        self.model_name = model_name
        print(f"[Load] Upscale model loaded: {model_name} (scale: {self.scale}x)")
        return self.model
    
    def clear(self):
        if self.model is not None:
            del self.model
            self.model = None
        self.model_name = None
        self.scale = 2
        self.dtype = None
        if torch is not None:
            torch.cuda.empty_cache()


upscale_cache = UpscaleModelCache()


# ============================================================
# FastAPI App
# ============================================================
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

class GenerateRequest(BaseModel):
    provider: str = "nai"
    prompt: str
    negative_prompt: str = ""
    character_prompts: List[str] = []
    width: int = 832
    height: int = 1216
    steps: int = 28
    cfg: float = 5.0
    seed: int = -1
    sampler: str = "euler_ancestral"
    scheduler: str = "normal"
    
    # NAI
    nai_model: str = "nai-diffusion-4-5-full"
    smea: str = "none"
    uc_preset: str = "Heavy"
    quality_tags: bool = True
    furry_mode: bool = False
    cfg_rescale: float = 0.0
    variety_plus: bool = False

    # NAI Vibe Transfer (최대 16개)
    vibe_transfer: List[dict] = []  # [{"image": base64, "info_extracted": 1.0, "strength": 0.6}, ...]

    # NAI Character Reference (V4.5 only)
    character_reference: Optional[dict] = None  # {"image": base64, "fidelity": 0.5, "style_aware": True}

    # Base Image (img2img / inpaint)
    base_image: Optional[str] = None  # base64 encoded image
    base_mask: Optional[str] = None   # base64 encoded mask (white = inpaint area)
    base_mode: str = "inpaint"        # "img2img" | "inpaint"
    base_strength: float = 0.7        # 변형 강도
    base_noise: float = 0.0           # 노이즈

    # Local
    model: str = ""
    loras: List[dict] = []
    
    # Upscale (Local only)
    enable_upscale: bool = False
    upscale_model: str = ""
    downscale_ratio: float = 0.7
    upscale_steps: int = 15
    upscale_cfg: float = 5.0
    upscale_denoise: float = 0.5
    size_alignment: str = "none"  # "none", "8", "64"

    # Save Options
    save_format: str = "png"  # png, jpg, webp
    jpg_quality: int = 95
    strip_metadata: bool = False
    output_folder: str = ""


class PromptItem(BaseModel):
    name: str = ""
    content: str = ""
    slotIndex: int = 0


class MultiGenerateRequest(BaseModel):
    provider: str = "nai"
    base_prompt: str
    negative_prompt: str = ""
    character_prompts: List[str] = []
    prompt_list: List[PromptItem] = []
    width: int = 832
    height: int = 1216
    steps: int = 28
    cfg: float = 5.0
    seed: int = -1
    random_seed_per_image: bool = False
    sampler: str = "euler_ancestral"
    scheduler: str = "normal"
    nai_model: str = "nai-diffusion-4-5-full"
    smea: str = "none"
    uc_preset: str = "Heavy"
    quality_tags: bool = True
    furry_mode: bool = False
    cfg_rescale: float = 0.0
    variety_plus: bool = False
    model: str = ""
    loras: List[dict] = []
    output_folder: str = ""  # 비어있으면 outputs에 직접 저장, 있으면 outputs/폴더명에 저장

    # NAI Vibe Transfer
    vibe_transfer: List[dict] = []

    # NAI Character Reference (V4.5 only)
    character_reference: Optional[dict] = None

    # Base Image (img2img / inpaint)
    base_image: Optional[str] = None
    base_mask: Optional[str] = None
    base_mode: str = "inpaint"
    base_strength: float = 0.7
    base_noise: float = 0.0

    # Upscale (Local only)
    enable_upscale: bool = False
    upscale_model: str = ""
    downscale_ratio: float = 0.7
    upscale_steps: int = 15
    upscale_cfg: float = 5.0
    upscale_denoise: float = 0.5
    size_alignment: str = "none"

    # Save Options
    save_format: str = "png"  # png, jpg, webp
    jpg_quality: int = 95
    strip_metadata: bool = False


class ConfigUpdate(BaseModel):
    nai_token: Optional[str] = None
    checkpoints_dir: Optional[str] = None
    lora_dir: Optional[str] = None


# ============================================================
# NAI API
# ============================================================
# KSampler -> NAI sampler 변환
NAI_SAMPLER_MAP = {
    "euler_ancestral": "k_euler_ancestral",
    "euler": "k_euler",
    "dpmpp_2m": "k_dpmpp_2m",
    "dpmpp_2m_sde": "k_dpmpp_2m_sde",
    "dpmpp_sde": "k_dpmpp_sde",
    "dpmpp_3m_sde": "k_dpmpp_sde",  # fallback
    "dpmpp_2s_ancestral": "k_dpmpp_2s_ancestral",
    "ddim": "ddim",
    "uni_pc": "k_euler",  # fallback
    "lcm": "k_euler",  # fallback
}

# KSampler -> NAI scheduler 변환
NAI_SCHEDULER_MAP = {
    "normal": "karras",  # fallback
    "karras": "karras",
    "exponential": "exponential",
    "sgm_uniform": "karras",  # fallback
    "simple": "karras",  # fallback
    "ddim_uniform": "karras",  # fallback
    "beta": "karras",  # fallback
}

# Vibe 캐시 디렉토리
VIBE_CACHE_DIR = APP_DIR / "vibe_cache"
VIBE_CACHE_DIR.mkdir(exist_ok=True)

# 갤러리 디렉토리
GALLERY_DIR = APP_DIR / "gallery"
GALLERY_DIR.mkdir(exist_ok=True)


def get_vibe_cache_key(image_base64: str, model: str, info_extracted: float) -> str:
    """이미지 + 모델 + info_extracted로 캐시 키 생성"""
    import hashlib
    # 이미지의 해시값 생성 (base64 전체를 해싱)
    image_hash = hashlib.sha256(image_base64.encode()).hexdigest()[:16]
    # info_extracted는 소수점 2자리까지만 (0.70 -> "070")
    info_str = f"{int(info_extracted * 100):03d}"
    # 모델명 간소화
    model_short = model.replace("nai-diffusion-", "").replace("-", "")
    return f"{image_hash}_{model_short}_{info_str}"


def get_next_vibe_number() -> int:
    """다음 vibe 파일 번호 반환"""
    existing = list(VIBE_CACHE_DIR.glob("*.png"))
    if not existing:
        return 1
    max_num = 0
    for f in existing:
        try:
            # 파일명: imagename_0.6_0.7_0000001.png -> 마지막 숫자 추출
            parts = f.stem.split('_')
            if len(parts) >= 4:
                num = int(parts[-1])
                max_num = max(max_num, num)
        except:
            pass
    return max_num + 1


# Vibe 캐시 (메모리)
_vibe_key_map_cache = None
_vibe_key_map_mtime = 0
_vibe_data_cache = {}  # cache_key -> vibe_data

def get_cached_vibe(cache_key: str) -> Optional[str]:
    """캐시된 vibe 데이터 반환 (PNG 메타데이터에서 읽기, 메모리 캐시 사용)"""
    global _vibe_key_map_cache, _vibe_key_map_mtime, _vibe_data_cache
    from PIL import Image as PILImage

    # 메모리 캐시에서 먼저 확인
    if cache_key in _vibe_data_cache:
        return _vibe_data_cache[cache_key]

    # 키 매핑 파일을 메모리 캐시에서 읽기
    key_map_file = VIBE_CACHE_DIR / "key_map.json"
    if key_map_file.exists():
        try:
            current_mtime = key_map_file.stat().st_mtime
            if _vibe_key_map_cache is None or current_mtime > _vibe_key_map_mtime:
                _vibe_key_map_cache = json.loads(key_map_file.read_text(encoding='utf-8'))
                _vibe_key_map_mtime = current_mtime

            if _vibe_key_map_cache and cache_key in _vibe_key_map_cache:
                png_file = VIBE_CACHE_DIR / _vibe_key_map_cache[cache_key]
                if png_file.exists():
                    img = PILImage.open(png_file)
                    if 'vibe_data' in img.info:
                        vibe_data = img.info['vibe_data']
                        _vibe_data_cache[cache_key] = vibe_data  # 메모리 캐시에 저장
                        return vibe_data
        except:
            pass

    # 레거시 .vibe 파일 확인
    legacy_file = VIBE_CACHE_DIR / f"{cache_key}.vibe"
    if legacy_file.exists():
        vibe_data = legacy_file.read_text(encoding='utf-8')
        _vibe_data_cache[cache_key] = vibe_data
        return vibe_data

    return None


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """파일명에서 사용할 수 없는 문자 제거"""
    import re
    # 확장자 제거
    name = Path(name).stem
    # 특수문자 제거 (알파벳, 숫자, 한글, 언더스코어, 하이픈만 허용)
    name = re.sub(r'[^\w\s가-힣-]', '', name)
    # 공백을 언더스코어로
    name = name.replace(' ', '_')
    # 길이 제한
    return name[:max_length] if name else "vibe"


def save_vibe_cache(cache_key: str, encoded_vibe: str, original_image_base64: str,
                    strength: float, info_extracted: float, model: str, image_name: str = "vibe"):
    """vibe 데이터를 PNG 이미지로 저장 (썸네일 + 메타데이터)"""
    from PIL import Image as PILImage
    from PIL.PngImagePlugin import PngInfo

    # 원본 이미지 디코딩
    image_data = base64.b64decode(original_image_base64)
    pil_img = PILImage.open(io.BytesIO(image_data))

    # RGB로 변환
    if pil_img.mode == 'RGBA':
        background = PILImage.new('RGB', pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    elif pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    # 썸네일 크기로 리사이즈 (긴 변 512px)
    max_size = 512
    w, h = pil_img.size
    if w > max_size or h > max_size:
        scale = min(max_size / w, max_size / h)
        new_size = (int(w * scale), int(h * scale))
        pil_img = pil_img.resize(new_size, PILImage.LANCZOS)

    # 메타데이터에 vibe 정보 저장
    metadata = PngInfo()
    metadata.add_text("vibe_data", encoded_vibe)
    metadata.add_text("cache_key", cache_key)
    metadata.add_text("model", model)
    metadata.add_text("info_extracted", str(info_extracted))
    metadata.add_text("strength", str(strength))

    # 파일명 생성: {image_name}_{strength}_{info_extracted}_{number}.png
    safe_name = sanitize_filename(image_name)
    number = get_next_vibe_number()
    filename = f"{safe_name}_{strength:.1f}_{info_extracted:.1f}_{number:07d}.png"
    filepath = VIBE_CACHE_DIR / filename

    # PNG 저장
    pil_img.save(filepath, format='PNG', pnginfo=metadata)

    # 캐시 키 매핑 저장
    key_map_file = VIBE_CACHE_DIR / "key_map.json"
    try:
        if key_map_file.exists():
            key_map = json.loads(key_map_file.read_text(encoding='utf-8'))
        else:
            key_map = {}
        key_map[cache_key] = filename
        key_map_file.write_text(json.dumps(key_map, indent=2), encoding='utf-8')
    except:
        pass

    print(f"[NAI] Vibe saved: {filename}")


async def encode_vibe_v4(image_base64: str, model: str, info_extracted: float,
                        strength: float, token: str, image_name: str = "vibe") -> str:
    """V4+ 모델용 vibe 사전 인코딩 - /ai/encode-vibe 엔드포인트 사용 (캐시 지원)"""
    import httpx

    # 캐시 확인
    cache_key = get_vibe_cache_key(image_base64, model, info_extracted)
    cached = get_cached_vibe(cache_key)
    if cached:
        print(f"[NAI] Vibe cache hit: {cache_key}, data length: {len(cached)}")
        # 캐시 데이터 기본 검증 (base64 형식 확인)
        try:
            import base64 as b64
            decoded = b64.b64decode(cached)
            print(f"[NAI] Vibe cache data valid, decoded size: {len(decoded)} bytes")
        except Exception as e:
            print(f"[NAI] Vibe cache data invalid (not valid base64): {e}")
            # 손상된 캐시 무시하고 새로 인코딩
            cached = None

    if cached:
        return cached

    print(f"[NAI] Vibe cache miss, encoding...")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # NAI API는 정수 값일 때 정수로 전송 (1.0 -> 1)
    info_val = int(info_extracted) if info_extracted == int(info_extracted) else info_extracted

    # NAI 웹과 동일한 payload 구조 (information_extracted는 최상위 레벨)
    payload = {
        "image": image_base64,
        "information_extracted": info_val,
        "model": model
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://image.novelai.net/ai/encode-vibe",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            print(f"[NAI] encode-vibe error {response.status_code}: {response.text[:200]}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Vibe encoding failed: {response.text}"
            )

        # 응답은 바이너리 vibe 데이터 - base64로 인코딩
        encoded_vibe = base64.b64encode(response.content).decode('utf-8')
        print(f"[NAI] Vibe encoded successfully, length: {len(encoded_vibe)}")

        # 캐시에 PNG로 저장 (썸네일 이미지 + 메타데이터)
        save_vibe_cache(cache_key, encoded_vibe, image_base64, strength, info_extracted, model, image_name)

        return encoded_vibe


async def call_nai_api(req: GenerateRequest):
    import httpx

    token = CONFIG.get("nai_token", "")
    if not token:
        raise HTTPException(status_code=500, detail="NAI token not set. Go to Settings.")

    uc_preset_map = {"Heavy": 0, "Light": 1, "Human Focus": 2, "None": 3}
    uc_preset_value = uc_preset_map.get(req.uc_preset, 0)

    # SMEA는 V3 모델에서만 지원, V4+에서는 비활성화
    is_v4_model = "diffusion-4" in req.nai_model
    sm = req.smea in ["SMEA", "SMEA+DYN"] and not is_v4_model
    sm_dyn = req.smea == "SMEA+DYN" and not is_v4_model

    seed = req.seed if req.seed >= 0 else random.randint(0, 2**31 - 1)
    
    # NAI 값이면 그대로, KSampler 값이면 변환
    nai_sampler = NAI_SAMPLER_MAP.get(req.sampler, req.sampler)
    nai_scheduler = NAI_SCHEDULER_MAP.get(req.scheduler, req.scheduler)

    # Furry Mode: 프롬프트 앞에 "fur dataset, " 추가
    prompt_for_nai = f"fur dataset, {req.prompt}" if req.furry_mode else req.prompt

    # V4.5 Quality Tags (NAI 서버가 처리하지 않으므로 클라이언트에서 직접 추가)
    V45_QUALITY_TAGS = ", very aesthetic, masterpiece, no text"

    # V4.5 UC Presets (NAI 서버가 처리하지 않으므로 클라이언트에서 직접 추가)
    V45_UC_PRESETS = {
        "Heavy": "nsfw, lowres, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, dithering, halftone, screentone, multiple views, logo, too many watermarks, negative space, blank page",
        "Light": "nsfw, lowres, artistic error, scan artifacts, worst quality, bad quality, jpeg artifacts, multiple views, very displeasing, too many watermarks, negative space, blank page",
        "Furry Focus": "nsfw, {worst quality}, distracting watermark, unfinished, bad quality, {widescreen}, upscale, {sequence}, {{grandfathered content}}, blurred foreground, chromatic aberration, sketch, everyone, [sketch background], simple, [flat colors], ych (character), outline, multiple scenes, [[horror (theme)]], comic",
        "Human Focus": "nsfw, lowres, artistic error, film grain, scan artifacts, worst quality, bad quality, jpeg artifacts, very displeasing, chromatic aberration, dithering, halftone, screentone, multiple views, logo, too many watermarks, negative space, blank page, @_@, mismatched pupils, glowing eyes, bad anatomy",
    }

    # V4+ 모델에서만 클라이언트 태그 적용
    if is_v4_model:
        # Quality Tags 적용 (프롬프트 끝에 추가)
        if req.quality_tags:
            prompt_for_nai = prompt_for_nai + V45_QUALITY_TAGS

        # UC Preset 태그 적용 (네거티브 프롬프트 앞에 추가)
        uc_preset_tags = V45_UC_PRESETS.get(req.uc_preset, "")
        if uc_preset_tags:
            if req.negative_prompt:
                negative_for_nai = uc_preset_tags + ", " + req.negative_prompt
            else:
                negative_for_nai = uc_preset_tags
        else:
            negative_for_nai = req.negative_prompt
    else:
        # V3 이하 모델은 NAI 서버가 처리
        negative_for_nai = req.negative_prompt

    params = {
        "params_version": 3,
        "width": req.width,
        "height": req.height,
        "scale": req.cfg,
        "sampler": nai_sampler,
        "steps": req.steps,
        "seed": int(seed),
        "n_samples": 1,
        "ucPreset": uc_preset_value,
        "qualityToggle": req.quality_tags,
        "sm": sm,
        "sm_dyn": sm_dyn,
        "dynamic_thresholding": False,
        "controlnet_strength": 1.0,
        "legacy": False,
        "add_original_image": True,
        "cfg_rescale": req.cfg_rescale,
        "noise_schedule": nai_scheduler,
        "legacy_v3_extend": False,
        "uncond_scale": 1.0,
        "negative_prompt": negative_for_nai,
        "prompt": prompt_for_nai,
        "extra_noise_seed": int(seed),
        "use_coords": False,
        "characterPrompts": [{"prompt": cp, "uc": "", "center": {"x": 0.5, "y": 0.5}, "enabled": True} for cp in req.character_prompts] if req.character_prompts else [],
        "v4_prompt": {
            "use_coords": False,
            "use_order": True,
            "caption": {
                "base_caption": prompt_for_nai,
                "char_captions": [{"char_caption": cp, "centers": [{"x": 0.5, "y": 0.5}]} for cp in req.character_prompts] if req.character_prompts else []
            }
        },
        "v4_negative_prompt": {
            "legacy_uc": False,
            "caption": {
                "base_caption": negative_for_nai,
                "char_captions": [{"char_caption": "", "centers": [{"x": 0.5, "y": 0.5}]} for _ in req.character_prompts] if req.character_prompts else []
            }
        },
    }

    # Variety+ 옵션 (값이 있을 때만 추가)
    if req.variety_plus:
        params["skip_cfg_above_sigma"] = 19
    
    # k_euler_ancestral + non-native scheduler 조합에서 필수 파라미터
    if nai_sampler == "k_euler_ancestral" and nai_scheduler != "native":
        params["deliberate_euler_ancestral_bug"] = False
        params["prefer_brownian"] = True

    # Vibe Transfer - V4+ 모델은 /ai/encode-vibe로 사전 인코딩 필요
    # 파라미터는 vibe가 있을 때만 추가 (빈 배열 전송 방지)
    if req.vibe_transfer and len(req.vibe_transfer) > 0:
        vibe_images = []
        info_extracted_list = []
        strength_list = []

        # V4+ 모델인지 확인
        is_v4_plus = "diffusion-4" in req.nai_model

        for i, v in enumerate(req.vibe_transfer):
            try:
                info_extracted = v.get("info_extracted", 1.0)
                strength = v.get("strength", 0.6)

                if is_v4_plus:
                    # V4+ 모델: encode-vibe 엔드포인트로 사전 인코딩
                    # 이미 인코딩된 데이터가 있고, 모델이 일치하면 그대로 사용
                    encoded_model = v.get("encoded_model", "")
                    # 모델명에서 베이스 모델 추출 (예: nai-diffusion-4-5-full-inpainting -> nai-diffusion-4-5-full)
                    base_model = req.nai_model.replace("-inpainting", "")
                    encoded_base_model = encoded_model.replace("-inpainting", "") if encoded_model else ""

                    # info_extracted 값도 비교 (인코딩 시점의 값과 현재 값)
                    encoded_info_extracted = v.get("encoded_info_extracted", info_extracted)
                    info_match = abs(info_extracted - encoded_info_extracted) <= 0.001

                    if v.get("encoded") and encoded_base_model == base_model and info_match:
                        print(f"[NAI] Vibe {i+1}: using pre-encoded data (cached, model match)")
                        vibe_images.append(v["encoded"])
                    else:
                        orig_size = get_image_size_from_base64(v["image"])
                        png_image = ensure_png_base64(v["image"])
                        image_name = v.get("name", f"vibe_{i+1}")

                        if v.get("encoded") and encoded_base_model != base_model:
                            print(f"[NAI] Vibe {i+1}: model mismatch ({encoded_model} -> {req.nai_model}), re-encoding...")
                        elif v.get("encoded") and not info_match:
                            print(f"[NAI] Vibe {i+1}: info_extracted changed ({encoded_info_extracted} -> {info_extracted}), re-encoding...")
                        else:
                            print(f"[NAI] Vibe {i+1}: {orig_size[0]}x{orig_size[1]}, encoding for V4+...")

                        encoded_vibe = await encode_vibe_v4(png_image, req.nai_model, info_extracted, strength, token, image_name)
                        vibe_images.append(encoded_vibe)
                        print(f"[NAI] Vibe {i+1}: encoded successfully")
                else:
                    # V3 모델: 원본 이미지 그대로 전송
                    png_image = ensure_png_base64(v["image"])
                    vibe_images.append(png_image)
                    print(f"[NAI] Vibe {i+1}: using raw image for V3")

                # NAI API는 정수 값일 때 정수로 전송 (1.0 -> 1)
                if info_extracted == int(info_extracted):
                    info_extracted_list.append(int(info_extracted))
                else:
                    info_extracted_list.append(info_extracted)
                strength_list.append(strength)
            except Exception as e:
                print(f"[NAI] Vibe {i+1} error: {e}")
                raise

        params["reference_image_multiple"] = vibe_images
        params["reference_information_extracted_multiple"] = info_extracted_list
        params["reference_strength_multiple"] = strength_list

    # Character Reference (V4.5 only) - NAIS2 방식 참고
    # https://github.com/sunanakgo/NAIS2
    if req.character_reference and req.character_reference.get("image"):
        fidelity = req.character_reference.get("fidelity", 0.5)
        style_aware = req.character_reference.get("style_aware", True)
        caption_type = "character&style" if style_aware else "character"

        # 프론트엔드에서 이미 Canvas로 처리된 이미지 (JPEG, 패딩 완료)
        # 추가 처리 없이 그대로 사용
        processed_image = req.character_reference["image"]
        
        try:
            w, h = get_image_size_from_base64(processed_image)
            print(f"[NAI] Character Reference: {w}x{h} (pre-processed by frontend)")
        except Exception as e:
            print(f"[NAI] Character Reference: size check failed: {e}")

        # NAIS2 방식: director_reference_images 사용 (JPEG, 패딩된 이미지)
        params["director_reference_images"] = [processed_image]
        params["director_reference_information_extracted"] = [1.0]
        params["director_reference_strength_values"] = [1.0]
        params["director_reference_secondary_strength_values"] = [round(1.0 - fidelity, 2)]
        params["director_reference_descriptions"] = [{
            "caption": {
                "base_caption": caption_type,
                "char_captions": []
            },
            "legacy_uc": False
        }]

        print(f"[NAI] CharRef: fidelity={fidelity}, secondary={round(1.0 - fidelity, 2)}, caption={caption_type}, data_len={len(processed_image)}")

    # Base Image (img2img / inpaint) 처리
    action = "generate"
    model_to_use = req.nai_model
    if req.base_image:
        # 이미지를 PNG로 변환
        base_png = ensure_png_base64(req.base_image)
        params["image"] = base_png
        params["strength"] = req.base_strength

        if req.base_mode == "inpaint" and req.base_mask:
            action = "infill"

            # 이미지와 마스크 크기 확인
            from PIL import Image as PILImage
            img_data = base64.b64decode(base_png)
            img_pil = PILImage.open(io.BytesIO(img_data))
            print(f"[NAI] Source image size: {img_pil.size}")

            # 마스크를 이진화 (NAI는 순수 흑백만 지원, 회색 가장자리 제거)
            mask_png = binarize_mask(req.base_mask)
            params["mask"] = mask_png

            # 마스크 크기 확인
            mask_data = base64.b64decode(mask_png)
            mask_pil = PILImage.open(io.BytesIO(mask_data))
            print(f"[NAI] Mask size: {mask_pil.size}")

            if img_pil.size != mask_pil.size:
                print(f"[NAI] WARNING: Image and mask size mismatch!")

            # NAI 웹과 정확히 동일한 인페인트 파라미터
            params["strength"] = 0.7
            params["add_original_image"] = False
            params["image_format"] = "png"
            params["inpaintImg2ImgStrength"] = 1
            params["legacy"] = False
            params["legacy_v3_extend"] = False
            params["noise"] = 0  # 삭제가 아니라 0으로 설정

            # 인페인트는 바이브/캐릭터레퍼런스 미지원 (NAI 웹 확인)
            params_to_delete = [
                # Vibe Transfer
                "reference_image_multiple",
                "reference_information_extracted_multiple",
                "reference_strength_multiple",
                # Character Reference
                "director_reference_images",
                "director_reference_information_extracted",
                "director_reference_strength_values",
                "director_reference_secondary_strength_values",
                "director_reference_descriptions"
            ]
            for param in params_to_delete:
                if param in params:
                    del params[param]

            # 인페인트는 전용 모델 사용 (모델명 + "-inpainting")
            # 예: nai-diffusion-4-5-full → nai-diffusion-4-5-full-inpainting
            if not model_to_use.endswith("-inpainting"):
                model_to_use = f"{model_to_use}-inpainting"
            print(f"[NAI] Mode: Inpaint, model={model_to_use}, user_strength={req.base_strength}")
        else:
            action = "img2img"
            # img2img 파라미터 (NAI 웹 페이로드와 동일하게)
            params["noise"] = req.base_noise
            params["image_format"] = "png"
            params["inpaintImg2ImgStrength"] = 1
            print(f"[NAI] Mode: Img2Img, strength={req.base_strength}, noise={req.base_noise}")

    payload = {
        "input": prompt_for_nai,
        "model": model_to_use,
        "action": action,
        "parameters": params
    }

    # 로그
    vibe_count = len(params.get("reference_image_multiple", []))
    has_char_ref = "director_reference_images" in params
    print(f"[NAI] Generating: {req.width}x{req.height}, steps={req.steps}, model={model_to_use}, vibes={vibe_count}, charref={has_char_ref}")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://image.novelai.net/ai/generate-image",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            error_text = response.text[:500] if len(response.text) > 500 else response.text
            print(f"[NAI] Error {response.status_code}: {error_text}")
            print(f"[NAI] Headers: {dict(response.headers)}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"NAI API error: {error_text}"
            )
        
        import zipfile
        zip_buffer = io.BytesIO(response.content)
        try:
            with zipfile.ZipFile(zip_buffer, 'r') as zf:
                image_name = zf.namelist()[0]
                image_data = zf.read(image_name)
                return Image.open(io.BytesIO(image_data)), int(seed)
        except zipfile.BadZipFile:
            # ZIP이 아닌 응답 (에러 메시지일 수 있음)
            content_preview = response.content[:500].decode('utf-8', errors='replace')
            print(f"[NAI] Unexpected response (not ZIP): {content_preview}")
            raise HTTPException(status_code=500, detail=f"NAI returned invalid response: {content_preview}")


# ============================================================
# Local Engine (ComfyUI-based)
# ============================================================

# 로컬 엔진 캐시
_local_engine_generator = None

def get_local_engine_generator():
    """로컬 엔진 생성기 인스턴스 가져오기 (싱글톤)"""
    global _local_engine_generator
    return _local_engine_generator

def call_local_engine(req: GenerateRequest):
    """
    ComfyUI 기반 로컬 엔진으로 이미지 생성

    Diffusers 대신 ComfyUI의 샘플링 코드를 사용하여
    동일한 모델/프롬프트/시드/세팅에서 ComfyUI와 동일한 결과물 생성
    """
    global _local_engine_generator

    import torch
    from local_engine import SDXLGenerator

    if not req.model:
        raise HTTPException(status_code=400, detail="Model not specified")

    model_path = CHECKPOINTS_DIR / req.model
    if not model_path.exists():
        raise HTTPException(status_code=400, detail=f"Model not found: {req.model}")

    # dtype/device 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # 생성기 초기화 또는 재사용
    if _local_engine_generator is None or _local_engine_generator.model_path != str(model_path):
        print(f"[LocalEngine] Loading model: {model_path}")
        _local_engine_generator = SDXLGenerator(
            model_path=str(model_path),
            device=device,
            dtype=dtype
        )

    # LoRA 적용
    if req.loras:
        _local_engine_generator.remove_loras()
        for lora_config in req.loras:
            lora_name = lora_config.get("name", "")
            lora_scale = lora_config.get("scale", 1.0)
            lora_path = LORA_DIR / lora_name
            if lora_path.exists():
                print(f"[LocalEngine] Applying LoRA: {lora_name} (scale={lora_scale})")
                _local_engine_generator.apply_lora(str(lora_path), lora_scale)

    seed = req.seed if req.seed >= 0 else np.random.randint(0, 2**31 - 1)

    # Base Image 처리 (img2img / inpaint)
    base_image = None
    mask = None
    denoise = req.base_strength if req.base_strength else 1.0

    if req.base_image:
        base_image = decode_base64_image(req.base_image)
        base_image = base_image.resize((req.width, req.height), Image.LANCZOS)

        if req.base_mode == "inpaint" and req.base_mask:
            mask = decode_base64_image(req.base_mask).convert("L")
            mask = mask.resize((req.width, req.height), Image.NEAREST)

    print(f"[LocalEngine] Generating: {req.width}x{req.height}, steps={req.steps}, cfg={req.cfg}")
    print(f"[LocalEngine] sampler={req.sampler}, scheduler={req.scheduler}, seed={seed}")
    print(f"[LocalEngine] prompt={req.prompt[:50]}...")

    # 진행률 콜백
    def progress_callback(step, total):
        print(f"[LocalEngine] Step {step + 1}/{total}")

    # 이미지 생성
    image, used_seed = _local_engine_generator.generate(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        steps=req.steps,
        cfg_scale=req.cfg,
        seed=seed,
        sampler=req.sampler,
        scheduler=req.scheduler,
        base_image=base_image,
        mask=mask,
        denoise=denoise,
        callback=progress_callback
    )

    print(f"[LocalEngine] Done, seed={used_seed}")
    return image, int(used_seed)


# ============================================================
# Local Diffusers (Legacy)
# ============================================================
def call_local_diffusers(req: GenerateRequest):
    lazy_imports()
    
    if not req.model:
        raise HTTPException(status_code=400, detail="Model not specified")
    
    pipe = model_cache.load_model(req.model)
    model_cache.set_lora_scales(req.loras)
    
    seed = req.seed if req.seed >= 0 else np.random.randint(0, 2**31 - 1)
    
    # CPU에서는 generator를 cpu로 설정
    gen_device = "cpu"  # diffusers는 generator를 항상 cpu에서 생성
    generator = torch.Generator(device=gen_device).manual_seed(int(seed))
    
    # KSampler -> diffusers scheduler 변환
    scheduler_map = {
        "euler": "EulerDiscreteScheduler",
        "euler_ancestral": "EulerAncestralDiscreteScheduler",
        "dpmpp_2m": "DPMSolverMultistepScheduler",
        "dpmpp_2m_sde": "DPMSolverMultistepScheduler",
        "dpmpp_sde": "DPMSolverSDEScheduler",
        "dpmpp_3m_sde": "DPMSolverMultistepScheduler",
        "dpmpp_2s_ancestral": "DPMSolverSinglestepScheduler",
        "ddim": "DDIMScheduler",
        "uni_pc": "UniPCMultistepScheduler",
        "lcm": "LCMScheduler",
    }
    
    from diffusers import schedulers

    # use_karras_sigmas 지원 여부 확인
    # EulerAncestralDiscreteScheduler는 karras 미지원 → EulerDiscreteScheduler로 대체
    scheduler_name = scheduler_map.get(req.sampler, "EulerAncestralDiscreteScheduler")

    # karras 스케줄 사용 시, 지원하지 않는 스케줄러는 대체
    if req.scheduler == "karras" and scheduler_name == "EulerAncestralDiscreteScheduler":
        scheduler_name = "EulerDiscreteScheduler"
        print(f"[Generate] euler_ancestral + karras → EulerDiscreteScheduler 사용 (karras 지원)")

    scheduler_class = getattr(schedulers, scheduler_name, None)
    if scheduler_class:
        try:
            # noise schedule (karras, normal 등) 적용
            scheduler_config = dict(pipe.scheduler.config)

            # karras sigma 스케줄 적용 (지원하는 스케줄러만)
            if req.scheduler == "karras":
                scheduler_config["use_karras_sigmas"] = True
            elif req.scheduler == "normal":
                scheduler_config["use_karras_sigmas"] = False
            # sgm_uniform, simple 등은 Diffusers에서 직접 지원 안 함

            pipe.scheduler = scheduler_class.from_config(scheduler_config)
            print(f"[Generate] Scheduler set: {scheduler_name}, noise_schedule={req.scheduler}, use_karras={scheduler_config.get('use_karras_sigmas', False)}")

            # 디버그: 실제 sigma 값 확인
            pipe.scheduler.set_timesteps(req.steps)
            sigmas = pipe.scheduler.sigmas if hasattr(pipe.scheduler, 'sigmas') else None
            if sigmas is not None:
                print(f"[DEBUG] Sigmas (first 5): {sigmas[:5].tolist()}")
                print(f"[DEBUG] Sigmas (last 5): {sigmas[-5:].tolist()}")
        except Exception as e:
            print(f"[Warning] Failed to set scheduler {scheduler_name}: {e}")

    print(f"[Generate] sampler={req.sampler}, scheduler={req.scheduler}, steps={req.steps}, cfg={req.cfg}")
    print(f"[Generate] prompt={req.prompt[:50]}..., size={req.width}x{req.height}")

    # Compel로 프롬프트 인코딩 (긴 프롬프트 + 가중치 지원)
    try:
        from compel import Compel, ReturnedEmbeddingsType

        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,  # 긴 프롬프트 지원
        )
        prompt_embeds, pooled_prompt_embeds = compel(req.prompt)
        negative_embeds, negative_pooled_embeds = compel(req.negative_prompt)

        # prompt_embeds와 negative_embeds 길이 맞추기 (패딩)
        # SDXL은 두 임베딩의 sequence length가 같아야 함
        max_len = max(prompt_embeds.shape[1], negative_embeds.shape[1])
        if prompt_embeds.shape[1] < max_len:
            padding = torch.zeros(
                (prompt_embeds.shape[0], max_len - prompt_embeds.shape[1], prompt_embeds.shape[2]),
                dtype=prompt_embeds.dtype, device=prompt_embeds.device
            )
            prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)
        if negative_embeds.shape[1] < max_len:
            padding = torch.zeros(
                (negative_embeds.shape[0], max_len - negative_embeds.shape[1], negative_embeds.shape[2]),
                dtype=negative_embeds.dtype, device=negative_embeds.device
            )
            negative_embeds = torch.cat([negative_embeds, padding], dim=1)

        use_compel = True
        print(f"[Generate] Using Compel for long prompt support (seq_len={max_len})")
    except Exception as e:
        print(f"[Warning] Compel failed, using default encoding: {e}")
        prompt_embeds = None
        pooled_prompt_embeds = None
        negative_embeds = None
        negative_pooled_embeds = None
        use_compel = False

    # Base Image 처리 (img2img / inpaint)
    if req.base_image:
        base_img = decode_base64_image(req.base_image)

        if req.base_mode == "inpaint" and req.base_mask:
            # Inpaint 모드
            from diffusers import AutoPipelineForInpainting

            mask_img = decode_base64_image(req.base_mask)
            # 마스크를 그레이스케일로 변환 (흰색 = 인페인트 영역)
            mask_img = mask_img.convert("L")

            inpaint_pipe = AutoPipelineForInpainting.from_pipe(pipe)
            print(f"[Generate] Inpaint mode, strength={req.base_strength}")

            if use_compel:
                result = inpaint_pipe(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_embeds,
                    image=base_img,
                    mask_image=mask_img,
                    width=req.width,
                    height=req.height,
                    num_inference_steps=req.steps,
                    guidance_scale=req.cfg,
                    strength=req.base_strength,
                    generator=generator,
                )
            else:
                result = inpaint_pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    image=base_img,
                    mask_image=mask_img,
                    width=req.width,
                    height=req.height,
                    num_inference_steps=req.steps,
                    guidance_scale=req.cfg,
                    strength=req.base_strength,
                    generator=generator,
                )
        else:
            # Img2Img 모드
            from diffusers import AutoPipelineForImage2Image

            # 베이스 이미지를 타겟 크기로 리사이즈
            base_img = base_img.resize((req.width, req.height), Image.LANCZOS)

            img2img_pipe = AutoPipelineForImage2Image.from_pipe(pipe)
            print(f"[Generate] Img2Img mode, strength={req.base_strength}")

            if use_compel:
                result = img2img_pipe(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_embeds,
                    image=base_img,
                    num_inference_steps=req.steps,
                    guidance_scale=req.cfg,
                    strength=req.base_strength,
                    generator=generator,
                )
            else:
                result = img2img_pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    image=base_img,
                    num_inference_steps=req.steps,
                    guidance_scale=req.cfg,
                    strength=req.base_strength,
                    generator=generator,
                )

        image = result.images[0]
        print(f"[Generate] Done (base image mode), seed={seed}")
    else:
        # 일반 txt2img 생성
        if use_compel:
            result = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                negative_pooled_prompt_embeds=negative_pooled_embeds,
                width=req.width,
                height=req.height,
                num_inference_steps=req.steps,
                guidance_scale=req.cfg,
                generator=generator,
            )
        else:
            result = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                width=req.width,
                height=req.height,
                num_inference_steps=req.steps,
                guidance_scale=req.cfg,
                generator=generator,
            )

        image = result.images[0]
        print(f"[Generate] 1st pass done, seed={seed}")
    
    # 2nd pass - 업스케일 (옵션)
    if req.enable_upscale and req.upscale_model:
        print(f"[Upscale] Starting 2-pass upscale with {req.upscale_model}")
        
        # 1. 업스케일 모델로 확대 (타일링 처리)
        upscale_model = upscale_cache.load_model(req.upscale_model)
        
        # PIL -> Tensor (BCHW) - 모델의 dtype에 맞춤
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device=model_cache.device, dtype=upscale_cache.dtype)
        
        # 타일링 업스케일 (OOM 방지)
        with torch.no_grad():
            upscaled_tensor = tiled_upscale(upscale_model, img_tensor, tile_size=512, overlap=32)
        
        # 업스케일 모델 VRAM 해제
        upscale_cache.model.to("cpu")
        torch.cuda.empty_cache()
        
        upscaled_np = (upscaled_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
        upscaled_image = Image.fromarray(upscaled_np)
        
        upscaled_w, upscaled_h = upscaled_image.size
        print(f"[Upscale] Upscaled to {upscaled_w}x{upscaled_h}")
        
        # 2. Downscale + Size alignment (2nd pass 전에!)
        target_w = int(upscaled_w * req.downscale_ratio)
        target_h = int(upscaled_h * req.downscale_ratio)
        
        # 정렬 적용
        if req.size_alignment == "64":
            target_w = ((target_w + 32) // 64) * 64
            target_h = ((target_h + 32) // 64) * 64
            target_w = max(64, target_w)
            target_h = max(64, target_h)
        elif req.size_alignment == "8":
            target_w = ((target_w + 4) // 8) * 8
            target_h = ((target_h + 4) // 8) * 8
            target_w = max(8, target_w)
            target_h = max(8, target_h)
        
        resized_image = upscaled_image.resize((target_w, target_h), Image.LANCZOS)
        print(f"[Upscale] Resized to {target_w}x{target_h}")
        
        # 3. 2nd pass img2img
        from diffusers import AutoPipelineForImage2Image
        
        img2img_pipe = AutoPipelineForImage2Image.from_pipe(pipe)
        generator2 = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
        
        result2 = img2img_pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=resized_image,
            num_inference_steps=req.upscale_steps,
            guidance_scale=req.upscale_cfg,
            strength=req.upscale_denoise,
            generator=generator2,
        )
        
        image = result2.images[0]
        print(f"[Upscale] 2nd pass done, final size={image.size[0]}x{image.size[1]}")
    
    print(f"[Generate] Done, seed={seed}")
    return image, int(seed)


def tiled_upscale(model, img_tensor, tile_size=512, overlap=32):
    """타일링 업스케일 (OOM 방지)"""
    _, _, h, w = img_tensor.shape
    scale = model.scale if hasattr(model, 'scale') else 2
    
    # 작은 이미지는 그냥 처리
    if h <= tile_size and w <= tile_size:
        return model(img_tensor)
    
    # 출력 텐서
    out_h, out_w = h * scale, w * scale
    output = torch.zeros((1, 3, out_h, out_w), dtype=img_tensor.dtype, device=img_tensor.device)
    weight = torch.zeros((1, 1, out_h, out_w), dtype=img_tensor.dtype, device=img_tensor.device)
    
    # 타일 처리
    step = tile_size - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            # 타일 범위
            y1 = min(y, h - tile_size) if y + tile_size > h else y
            x1 = min(x, w - tile_size) if x + tile_size > w else x
            y2 = y1 + tile_size
            x2 = x1 + tile_size
            
            # 타일 추출 및 업스케일
            tile = img_tensor[:, :, y1:y2, x1:x2]
            upscaled_tile = model(tile)
            
            # 출력 위치
            oy1, oy2 = y1 * scale, y2 * scale
            ox1, ox2 = x1 * scale, x2 * scale
            
            output[:, :, oy1:oy2, ox1:ox2] += upscaled_tile
            weight[:, :, oy1:oy2, ox1:ox2] += 1
    
    # 평균화
    output = output / weight.clamp(min=1)
    return output.clamp(0, 1)


# ============================================================
# App Setup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global install_status
    
    print("=" * 50)
    print("NAI + Local Generator Backend")
    print(f"App dir: {APP_DIR}")
    print(f"Checkpoints: {CONFIG.get('checkpoints_dir', CHECKPOINTS_DIR)}")
    print(f"NAI Token: {'Set' if CONFIG.get('nai_token') else 'Not set'}")
    print("=" * 50)
    
    # 큐 처리 백그라운드 태스크 시작
    queue_task = asyncio.create_task(process_queue())

    # 서버 준비 완료 후 브라우저 열기 (짧은 지연으로 서버가 완전히 시작되도록)
    async def open_browser_delayed():
        await asyncio.sleep(0.5)
        import webbrowser
        webbrowser.open("http://127.0.0.1:8765")
    asyncio.create_task(open_browser_delayed())
    
    yield
    
    # 종료 시 태스크 취소
    queue_task.cancel()
    model_cache.clear()
    upscale_cache.clear()


app = FastAPI(title="NAI Generator Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 큐 시스템 (WebSocket 기반)
from collections import deque
import uuid

class GenerationQueue:
    def __init__(self):
        self.queue = deque()  # 대기 큐
        self.current_job = None  # 현재 실행 중인 작업
        self.current_job_id = None
        self.cancel_current = False  # 현재 작업 취소 플래그
        self.is_processing = False

        # WebSocket 클라이언트 관리
        self.clients: dict[str, WebSocket] = {}  # client_id → WebSocket

        # 진행 상황 추적
        self.completed_images = 0
        self.total_images = 0

        # 재연결 시 동기화용 - 최근 완료된 이미지 목록 (최대 100개 유지)
        self.recent_images: list[dict] = []
        self.image_sequence = 0  # 이미지 순번 (재연결 시 누락 감지용)

    def add_job(self, request):
        job_id = str(uuid.uuid4())[:8]
        job = {"id": job_id, "request": request}
        self.queue.append(job)
        return job_id

    def get_next_job(self):
        if self.queue:
            return self.queue.popleft()
        return None

    def clear_queue(self):
        """대기 큐만 비우기 (현재 작업은 유지)"""
        cleared_jobs = len(self.queue)
        cleared_images = 0
        for job in self.queue:
            req = job["request"]
            prompt_count = len(req.prompt_list) if req.prompt_list else 1
            cleared_images += prompt_count
        self.queue.clear()
        return cleared_jobs, cleared_images

    def cancel_current_job(self):
        """현재 작업 취소 신호"""
        self.cancel_current = True

    def get_status(self):
        return {
            "queue_length": len(self.queue),
            "current_job_id": self.current_job_id,
            "is_processing": self.is_processing,
            "queued_jobs": [{"id": j["id"], "prompts": len(j["request"].prompt_list or [''])} for j in self.queue],
            "completed_images": self.completed_images,
            "total_images": self.total_images,
            "image_sequence": self.image_sequence
        }

    def add_completed_image(self, image_data: dict):
        """완료된 이미지 기록 (재연결 동기화용)"""
        self.image_sequence += 1
        image_data["seq"] = self.image_sequence
        self.recent_images.append(image_data)
        # 최대 100개 유지
        if len(self.recent_images) > 100:
            self.recent_images.pop(0)

    def get_images_since(self, last_seq: int) -> list[dict]:
        """특정 순번 이후의 이미지 목록 반환"""
        return [img for img in self.recent_images if img.get("seq", 0) > last_seq]

    async def broadcast(self, data):
        """모든 WebSocket 클라이언트에 메시지 전송"""
        msg_type = data.get('type', 'unknown')
        if msg_type == 'image':
            print(f"[WS Broadcast] Sending image to {len(self.clients)} clients")

        disconnected = []
        for client_id, ws in self.clients.items():
            try:
                await ws.send_json(data)
            except Exception as e:
                print(f"[WS Broadcast] Failed to send to {client_id}: {e}")
                disconnected.append(client_id)

        # 연결 끊긴 클라이언트 제거
        for client_id in disconnected:
            self.clients.pop(client_id, None)

    async def send_to_client(self, client_id: str, data: dict):
        """특정 클라이언트에 메시지 전송"""
        ws = self.clients.get(client_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception as e:
                print(f"[WS] Failed to send to {client_id}: {e}")
                self.clients.pop(client_id, None)

gen_queue = GenerationQueue()


async def process_queue():
    """큐 처리 루프"""
    while True:
        if not gen_queue.is_processing and gen_queue.queue:
            job = gen_queue.get_next_job()
            if job:
                gen_queue.is_processing = True
                gen_queue.current_job = job
                gen_queue.current_job_id = job["id"]
                gen_queue.cancel_current = False

                try:
                    await process_job(job)
                except Exception as e:
                    print(f"[Error] Job failed: {e}")

                gen_queue.is_processing = False
                gen_queue.current_job = None
                gen_queue.current_job_id = None
                gen_queue.cancel_current = False

        await asyncio.sleep(0.1)


async def process_job(job):
    """단일 작업 처리"""
    import time
    job_start_time = time.time()

    req = job["request"]
    job_id = job["id"]

    # PromptItem 리스트 처리
    prompts = req.prompt_list if req.prompt_list else [PromptItem(name="", content="")]
    total_images = len(prompts)
    image_idx = 0

    # 시드 설정
    current_seed = req.seed if req.seed >= 0 else random.randint(0, 2**31 - 1)

    print(f"[Generate] Job {job_id} started - {total_images} image(s), {req.width}x{req.height}, {req.steps} steps")

    # 시작 알림 (total은 job_queued에서 이미 증가됨)
    await gen_queue.broadcast({
        "type": "job_start",
        "job_id": job_id,
        "job_total": total_images,
        "progress": {
            "completed": gen_queue.completed_images,
            "total": gen_queue.total_images,
            "queue_length": len(gen_queue.queue)
        }
    })

    for prompt_idx, prompt_item in enumerate(prompts):
        # 취소 체크
        if gen_queue.cancel_current:
            cancelled_images = total_images - image_idx
            # 취소된 이미지 수만큼 total에서 차감
            gen_queue.total_images -= cancelled_images
            await gen_queue.broadcast({
                "type": "job_cancelled",
                "job_id": job_id,
                "cancelled_images": cancelled_images,
                "progress": {
                    "completed": gen_queue.completed_images,
                    "total": gen_queue.total_images,
                    "queue_length": len(gen_queue.queue)
                }
            })
            return
        
        extra_prompt = prompt_item.content if hasattr(prompt_item, 'content') else str(prompt_item)
        prompt_name = prompt_item.name if hasattr(prompt_item, 'name') else ""
        slot_index = prompt_item.slotIndex if hasattr(prompt_item, 'slotIndex') else prompt_idx
        
        full_prompt = f"{req.base_prompt}, {extra_prompt}".strip(", ") if extra_prompt else req.base_prompt
        
        # 로컬인 경우 캐릭터 프롬프트를 메인 프롬프트에 합침
        if req.provider != "nai" and req.character_prompts:
            char_prompts_str = ", ".join(req.character_prompts)
            full_prompt = f"{full_prompt}, {char_prompts_str}".strip(", ")
        
        if req.random_seed_per_image and prompt_idx > 0:
            current_seed = random.randint(0, 2**31 - 1)
        
        single_req = GenerateRequest(
            provider=req.provider,
            prompt=full_prompt,
            negative_prompt=req.negative_prompt,
            character_prompts=req.character_prompts,
            width=req.width,
            height=req.height,
            steps=req.steps,
            cfg=req.cfg,
            seed=int(current_seed),
            sampler=req.sampler,
            scheduler=req.scheduler,
            nai_model=req.nai_model,
            smea=req.smea,
            uc_preset=req.uc_preset,
            quality_tags=req.quality_tags,
            furry_mode=req.furry_mode,
            cfg_rescale=req.cfg_rescale,
            variety_plus=req.variety_plus,
            model=req.model,
            loras=req.loras,
            # NAI Vibe Transfer & Character Reference
            vibe_transfer=req.vibe_transfer,
            character_reference=req.character_reference,
            # Base Image (img2img / inpaint)
            base_image=req.base_image,
            base_mask=req.base_mask,
            base_mode=req.base_mode,
            base_strength=req.base_strength,
            base_noise=req.base_noise,
            # Upscale params
            enable_upscale=req.enable_upscale,
            upscale_model=req.upscale_model,
            downscale_ratio=req.downscale_ratio,
            upscale_steps=req.upscale_steps,
            upscale_cfg=req.upscale_cfg,
            upscale_denoise=req.upscale_denoise,
            size_alignment=req.size_alignment,
            # Save Options
            save_format=req.save_format,
            jpg_quality=req.jpg_quality,
            strip_metadata=req.strip_metadata,
            output_folder=req.output_folder,
        )
        
        try:
            image_start_time = time.time()
            if req.provider == "nai":
                image, actual_seed = await call_nai_api(single_req)
            else:
                # ComfyUI 포팅 로컬 엔진 사용
                image, actual_seed = call_local_engine(single_req)
            image_time = time.time() - image_start_time
            
            # 파일명용 태그 결정: name > 첫 태그 (sanitize_filename 사용)
            file_tag = ""
            if prompt_name:
                file_tag = sanitize_filename(prompt_name, max_length=100)
            elif extra_prompt:
                tag = extra_prompt.split(",")[0].strip()
                file_tag = sanitize_filename(tag, max_length=100)
            
            image_idx += 1

            # 저장 경로 결정
            if req.output_folder:
                save_dir = OUTPUT_DIR / req.output_folder
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = OUTPUT_DIR

            # 카테고리 결정 (슬롯 인덱스 + 파일 태그)
            if file_tag:
                category = f"{slot_index+1:03d}_{file_tag}"
            else:
                category = f"{slot_index+1:03d}"

            # 저장 포맷 결정
            save_format = getattr(req, 'save_format', 'png').lower()
            if save_format not in ['png', 'jpg', 'webp']:
                save_format = 'png'
            ext = 'jpg' if save_format == 'jpg' else save_format

            file_num = get_next_image_number(category, save_dir, ext)
            filename = f"{category}_{file_num:07d}.{ext}"

            # 메타데이터 옵션
            strip_metadata = getattr(req, 'strip_metadata', False)
            jpg_quality = getattr(req, 'jpg_quality', 95)

            # NAI 호환 메타데이터 구조 생성
            # 기존 NAI Comment가 있으면 파싱, 없으면 새로 생성
            existing_comment = None
            if hasattr(image, 'info') and 'Comment' in image.info:
                try:
                    existing_comment = json.loads(image.info['Comment'])
                except:
                    pass

            # SMEA 설정 파싱
            sm = req.smea in ['SMEA', 'SMEA+DYN']
            sm_dyn = req.smea == 'SMEA+DYN'

            # Vibe Transfer 정보 (이미지 포함 - 복원용)
            vibe_info = []
            if req.vibe_transfer:
                for v in req.vibe_transfer:
                    vibe_entry = {
                        "strength": v.get("strength", 0.6),
                        "info_extracted": v.get("info_extracted", 1.0),
                        "name": v.get("name", "")
                    }
                    # 이미지 데이터 포함 (복원용)
                    if v.get("image"):
                        vibe_entry["image"] = v.get("image")
                    if v.get("encoded"):
                        vibe_entry["encoded"] = v.get("encoded")
                    if v.get("encoded_model"):
                        vibe_entry["encoded_model"] = v.get("encoded_model")
                    vibe_info.append(vibe_entry)

            # PeroPix 확장 필드
            peropix_ext = {
                "version": 1,
                "provider": req.provider,
                "character_prompts": req.character_prompts or [],
                "variety_plus": req.variety_plus,
                "furry_mode": req.furry_mode,
                "local_model": req.model if req.provider == 'local' else "",
                "vibe_transfer": vibe_info if vibe_info else None
            }

            if existing_comment:
                # NAI에서 생성된 이미지: 기존 Comment에 peropix 확장 추가
                existing_comment["peropix"] = peropix_ext
                existing_comment["request_type"] = req.nai_model
                unified_metadata = existing_comment
            else:
                # Local 또는 Comment 없는 경우: NAI 호환 형식으로 전체 생성
                unified_metadata = {
                    "prompt": full_prompt,
                    "uc": req.negative_prompt or "",
                    "steps": req.steps,
                    "width": req.width,
                    "height": req.height,
                    "scale": req.cfg,
                    "seed": actual_seed,
                    "sampler": req.sampler,
                    "noise_schedule": req.scheduler,
                    "sm": sm,
                    "sm_dyn": sm_dyn,
                    "ucPreset": req.uc_preset,
                    "qualityToggle": req.quality_tags,
                    "cfg_rescale": req.cfg_rescale,
                    "request_type": req.nai_model,
                    "peropix": peropix_ext
                }

            # 이미지를 바이트로 변환
            img_buffer = io.BytesIO()
            format_map = {'png': 'PNG', 'jpg': 'JPEG', 'webp': 'WEBP'}
            pil_format = format_map.get(save_format, 'PNG')

            # JPEG는 RGB만 지원
            save_image = image
            if save_format == 'jpg' and image.mode in ('RGBA', 'P'):
                save_image = image.convert('RGB')

            # strip_metadata 처리
            if strip_metadata:
                # 메타데이터 제거: 깨끗한 이미지로 저장
                # PIL의 info 딕셔너리를 제거하기 위해 새 이미지로 복사
                clean_image = Image.new(save_image.mode, save_image.size)
                clean_image.putdata(list(save_image.getdata()))

                if save_format == 'jpg':
                    clean_image.save(img_buffer, format=pil_format, quality=jpg_quality)
                elif save_format == 'webp':
                    clean_image.save(img_buffer, format=pil_format, quality=jpg_quality)
                else:
                    clean_image.save(img_buffer, format=pil_format)

                image_bytes = img_buffer.getvalue()
            else:
                # 메타데이터 유지: 먼저 깨끗하게 저장 후 새 메타데이터 추가
                clean_image = Image.new(save_image.mode, save_image.size)
                clean_image.putdata(list(save_image.getdata()))

                if save_format == 'jpg':
                    clean_image.save(img_buffer, format=pil_format, quality=jpg_quality)
                elif save_format == 'webp':
                    clean_image.save(img_buffer, format=pil_format, quality=jpg_quality)
                else:
                    clean_image.save(img_buffer, format=pil_format)

                image_bytes = img_buffer.getvalue()
                image_bytes = save_metadata_to_exif(image_bytes, unified_metadata, pil_format)

            # 파일로 저장
            save_path = save_dir / filename
            with open(save_path, 'wb') as f:
                f.write(image_bytes)

            # 진행 상황 업데이트
            gen_queue.completed_images += 1

            # 이미지 경로 (output_folder 포함)
            image_path = f"{req.output_folder}/{filename}" if req.output_folder else filename

            # 이미지 데이터 구성
            image_data = {
                "type": "image",
                "job_id": job_id,
                "slot_idx": slot_index,
                "seed": actual_seed,
                "image_path": image_path,
                "filename": filename,
                "prompt": full_prompt,
                "metadata": None if strip_metadata else unified_metadata
            }

            # 재연결 동기화용 기록
            gen_queue.add_completed_image(image_data.copy())

            # 진행 상황 추가
            image_data["progress"] = {
                "completed": gen_queue.completed_images,
                "total": gen_queue.total_images,
                "queue_length": len(gen_queue.queue)
            }

            print(f"[Generate] Image {image_idx}/{total_images} completed - {image_time:.1f}s - {filename}")
            await gen_queue.broadcast(image_data)
            
        except Exception as e:
            import traceback
            print(f"[Error] Generation failed: {e}")
            traceback.print_exc()
            image_idx += 1
            # 에러도 완료로 카운트 (프로그레스바 진행용)
            gen_queue.completed_images += 1
            await gen_queue.broadcast({
                "type": "error",
                "job_id": job_id,
                "slot_idx": slot_index,
                "error": str(e),
                "progress": {
                    "completed": gen_queue.completed_images,
                    "total": gen_queue.total_images,
                    "queue_length": len(gen_queue.queue)
                }
            })

    # 큐가 비면 카운터 리셋
    queue_empty = len(gen_queue.queue) == 0
    job_total_time = time.time() - job_start_time
    print(f"[Generate] Job {job_id} finished - {image_idx} image(s) in {job_total_time:.1f}s")

    await gen_queue.broadcast({
        "type": "job_done",
        "job_id": job_id,
        "progress": {
            "completed": gen_queue.completed_images,
            "total": gen_queue.total_images,
            "queue_length": len(gen_queue.queue)
        }
    })

    if queue_empty:
        gen_queue.completed_images = 0
        gen_queue.total_images = 0
        gen_queue.recent_images.clear()
        gen_queue.image_sequence = 0



def image_to_base64(img) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/version")
async def get_version():
    """현재 앱 버전 정보 반환"""
    return {
        "version": APP_VERSION.get("version", "1.0.0"),
        "build_date": APP_VERSION.get("build_date", "unknown")
    }


@app.get("/api/check-update")
async def check_update():
    """GitHub Releases에서 최신 버전 확인"""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.github.com/repos/mrm987/PeroPix/releases/latest",
                headers={"Accept": "application/vnd.github.v3+json"}
            )

            if response.status_code == 200:
                release_data = response.json()
                latest_version = release_data.get("tag_name", "").lstrip("v")
                current_version = APP_VERSION.get("version", "1.0.0")

                # 버전 비교 (semantic versioning)
                def parse_version(v):
                    try:
                        parts = v.split(".")
                        return tuple(int(p) for p in parts[:3])
                    except:
                        return (0, 0, 0)

                current = parse_version(current_version)
                latest = parse_version(latest_version)
                has_update = latest > current

                return {
                    "has_update": has_update,
                    "current_version": current_version,
                    "latest_version": latest_version,
                    "release_url": release_data.get("html_url", ""),
                    "release_notes": release_data.get("body", ""),
                    "published_at": release_data.get("published_at", "")
                }
            elif response.status_code == 404:
                return {
                    "has_update": False,
                    "current_version": APP_VERSION.get("version", "1.0.0"),
                    "error": "No releases found"
                }
            else:
                logger.warning(f"GitHub API 응답 오류: {response.status_code}")
                return {
                    "has_update": False,
                    "current_version": APP_VERSION.get("version", "1.0.0"),
                    "error": f"GitHub API error: {response.status_code}"
                }
    except Exception as e:
        logger.error(f"업데이트 확인 실패: {e}")
        return {
            "has_update": False,
            "current_version": APP_VERSION.get("version", "1.0.0"),
            "error": str(e)
        }


@app.post("/api/extract-metadata")
async def extract_metadata(request: dict):
    """이미지에서 메타데이터 추출 (PNG/JPG/WebP 지원)"""
    image_base64 = request.get("image")
    if not image_base64:
        return {"success": False, "error": "No image provided"}

    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))

        # PNG 텍스트 메타데이터 (바이브 파일 확인용)
        raw_metadata = {}
        if hasattr(image, 'info'):
            for key, value in image.info.items():
                if isinstance(value, str):
                    raw_metadata[key] = value

        # 바이브 파일 여부 확인
        is_vibe = 'vibe_data' in raw_metadata

        # 통합 메타데이터 읽기 (PNG Comment, 레거시 peropix, EXIF 순서로 시도)
        nai_metadata = read_metadata_from_image(image_data)

        return {
            "success": True,
            "is_vibe": is_vibe,
            "is_nai": bool(nai_metadata),
            "vibe_data": raw_metadata.get('vibe_data') if is_vibe else None,
            "vibe_model": raw_metadata.get('model') if is_vibe else None,
            "vibe_strength": float(raw_metadata.get('strength', 0.6)) if is_vibe else None,
            "vibe_info_extracted": float(raw_metadata.get('info_extracted', 1.0)) if is_vibe else None,
            "nai_metadata": nai_metadata if nai_metadata else None,
            "raw_metadata": raw_metadata
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# Gallery API
# ============================================================

def get_gallery_folder_path(folder: str = "") -> Path:
    """갤러리 폴더 경로 반환 (보안 검증 포함)"""
    if not folder or folder == "gallery":
        return GALLERY_DIR
    # 경로 조작 방지
    if ".." in folder or folder.startswith("/") or folder.startswith("\\"):
        raise ValueError("Invalid folder name")
    return GALLERY_DIR / folder


@app.get("/api/gallery/folders")
async def get_gallery_folders():
    """갤러리 하위 폴더 목록 조회"""
    folders = []
    try:
        for item in sorted(GALLERY_DIR.iterdir()):
            if item.is_dir():
                # 폴더 내 이미지 수 카운트 (PNG/JPG/WebP)
                image_count = 0
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
                    image_count += len(list(item.glob(ext)))
                folders.append({
                    "name": item.name,
                    "image_count": image_count
                })
        return {"success": True, "folders": folders}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/gallery/folders")
async def create_gallery_folder(request: dict):
    """갤러리 폴더 생성"""
    folder_name = request.get("name", "").strip()
    if not folder_name:
        return {"success": False, "error": "Folder name is required"}

    # 안전한 폴더명인지 확인
    if "/" in folder_name or "\\" in folder_name or ".." in folder_name:
        return {"success": False, "error": "Invalid folder name"}

    folder_path = GALLERY_DIR / folder_name
    if folder_path.exists():
        return {"success": False, "error": "Folder already exists"}

    try:
        folder_path.mkdir(parents=True)
        return {"success": True, "name": folder_name}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/api/gallery/folders/{folder_name}")
async def delete_gallery_folder(folder_name: str):
    """갤러리 폴더 삭제 (빈 폴더만)"""
    if not folder_name or folder_name == "gallery":
        return {"success": False, "error": "Cannot delete root gallery"}

    folder_path = GALLERY_DIR / folder_name
    if not folder_path.exists():
        return {"success": False, "error": "Folder not found"}

    # 폴더가 비어있는지 확인
    if any(folder_path.iterdir()):
        return {"success": False, "error": "Folder is not empty"}

    try:
        folder_path.rmdir()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/gallery")
async def get_gallery(folder: str = ""):
    """갤러리 이미지 목록 조회 (폴더별) - PNG/JPG/WebP 지원"""
    images = []

    try:
        gallery_path = get_gallery_folder_path(folder)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    if not gallery_path.exists():
        return {"images": [], "folder": folder}

    # 지원하는 모든 이미지 포맷 검색
    all_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
        all_files.extend(gallery_path.glob(ext))

    # 수정 시간순 정렬 (최신순)
    all_files = sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True)

    for filepath in all_files:
        try:
            # 파일을 바이트로 읽어서 메타데이터 추출
            with open(filepath, 'rb') as f:
                image_bytes = f.read()

            # 통합 메타데이터 읽기 (PNG Comment, 레거시 peropix, EXIF 순서로 시도)
            comment_meta = read_metadata_from_image(image_bytes)

            # seed와 prompt 추출
            seed = comment_meta.get("seed") if comment_meta else None
            prompt = (comment_meta.get("prompt", "") or "")[:100] if comment_meta else ""
            has_metadata = bool(comment_meta)

            # 썸네일 생성 (고해상도 디스플레이용 2x)
            image = Image.open(io.BytesIO(image_bytes))
            thumb = image.copy()
            thumb.thumbnail((520, 520), Image.LANCZOS)
            buffer = io.BytesIO()
            thumb.save(buffer, format="PNG")
            thumb_base64 = base64.b64encode(buffer.getvalue()).decode()

            images.append({
                "filename": filepath.name,
                "thumbnail": thumb_base64,
                "seed": seed,
                "prompt": prompt,
                "has_metadata": has_metadata
            })
        except Exception as e:
            print(f"[Gallery] Error loading {filepath}: {e}")
            continue

    return {"images": images, "folder": folder}


@app.post("/api/gallery/save")
async def save_to_gallery(request: dict):
    """이미지를 갤러리에 저장 (메타데이터 보존)"""
    from PIL.PngImagePlugin import PngInfo

    image_base64 = request.get("image")
    image_path = request.get("image_path")  # 파일 경로 (우선)
    filename = request.get("filename", "gallery_image.png")
    folder = request.get("folder", "")  # 저장할 폴더
    metadata = request.get("metadata")  # 프론트엔드에서 전달된 메타데이터

    try:
        # 폴더 경로 검증
        try:
            gallery_path = get_gallery_folder_path(folder)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        # 폴더가 없으면 생성
        gallery_path.mkdir(parents=True, exist_ok=True)

        # image_path가 있으면 파일에서 읽기, 없으면 base64 디코딩
        if image_path:
            file_path = OUTPUT_DIR / image_path
            if not file_path.exists():
                return {"success": False, "error": "Image file not found"}
            image = Image.open(file_path)
        elif image_base64:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
        else:
            return {"success": False, "error": "No image provided"}

        # 고유 파일명 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(filename).stem
        new_filename = f"{base_name}_{timestamp}.png"
        filepath = gallery_path / new_filename

        # 메타데이터 처리
        pnginfo = PngInfo()
        has_comment = False

        # 원본 이미지에서 메타데이터 복사
        if hasattr(image, 'info') and image.info:
            for key, value in image.info.items():
                if isinstance(value, str):
                    pnginfo.add_text(key, value)
                    if key == 'Comment':
                        has_comment = True

        # 파일에 Comment가 없으면 프론트엔드에서 전달된 metadata 사용
        if not has_comment and metadata:
            pnginfo.add_text("Comment", json.dumps(metadata))

        image.save(filepath, format="PNG", pnginfo=pnginfo)

        return {"success": True, "filename": new_filename}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/gallery/{filename}")
async def get_gallery_image(filename: str, folder: str = ""):
    """갤러리 이미지 조회 (전체 크기 + 메타데이터) - PNG/JPG/WebP EXIF 지원"""
    try:
        gallery_path = get_gallery_folder_path(folder)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    filepath = gallery_path / filename
    if not filepath.exists():
        return {"success": False, "error": "Image not found"}

    try:
        # 파일을 바이트로 읽어서 메타데이터 추출
        with open(filepath, 'rb') as f:
            image_bytes = f.read()

        # 통합 메타데이터 읽기 (PNG Comment, 레거시 peropix, EXIF 순서로 시도)
        comment_meta = read_metadata_from_image(image_bytes)

        # 이미지를 PNG로 변환하여 반환
        image = Image.open(io.BytesIO(image_bytes))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        return {
            "success": True,
            "image": image_base64,
            "metadata": comment_meta or {},
            "filename": filename
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/api/gallery/{filename}")
async def delete_gallery_image(filename: str, folder: str = ""):
    """갤러리 이미지 삭제 (폴더 미지정 시 전체 검색)"""
    # 폴더가 지정된 경우 해당 폴더에서만 검색
    if folder:
        try:
            gallery_path = get_gallery_folder_path(folder)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        filepath = gallery_path / filename
        if filepath.exists():
            try:
                filepath.unlink()
                return {"success": True, "found_in": folder}
            except Exception as e:
                return {"success": False, "error": str(e)}

    # 폴더 미지정 또는 지정된 폴더에 없는 경우 전체 검색
    # 루트 폴더 먼저 확인
    filepath = GALLERY_DIR / filename
    if filepath.exists():
        try:
            filepath.unlink()
            return {"success": True, "found_in": ""}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # 서브폴더 검색
    for subfolder in GALLERY_DIR.iterdir():
        if subfolder.is_dir():
            filepath = subfolder / filename
            if filepath.exists():
                try:
                    filepath.unlink()
                    return {"success": True, "found_in": subfolder.name}
                except Exception as e:
                    return {"success": False, "error": str(e)}

    return {"success": False, "error": "Image not found"}


@app.post("/api/gallery/{filename}/move")
async def move_gallery_image(filename: str, request: dict):
    """갤러리 이미지를 다른 폴더로 이동"""
    from_folder = request.get("from_folder", "")
    to_folder = request.get("to_folder", "")

    try:
        from_path = get_gallery_folder_path(from_folder)
        to_path = get_gallery_folder_path(to_folder)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # 대상 폴더가 없으면 생성
    to_path.mkdir(parents=True, exist_ok=True)

    src_file = from_path / filename
    dst_file = to_path / filename

    if not src_file.exists():
        return {"success": False, "error": "Image not found"}

    if dst_file.exists():
        # 파일명 충돌 시 타임스탬프 추가
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(filename).stem
        new_filename = f"{base_name}_{timestamp}.png"
        dst_file = to_path / new_filename
    else:
        new_filename = filename

    try:
        import shutil
        shutil.move(str(src_file), str(dst_file))
        return {"success": True, "new_filename": new_filename}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.patch("/api/gallery/{filename}")
async def rename_gallery_image(filename: str, request: Request):
    """갤러리 이미지 이름 변경"""
    data = await request.json()
    new_name = data.get("new_name", "").strip()
    folder = data.get("folder", "")

    if not new_name:
        return {"success": False, "error": "New name is required"}

    # 확장자 처리
    if not new_name.lower().endswith(".png"):
        new_name += ".png"

    # 안전한 파일명인지 확인
    if "/" in new_name or "\\" in new_name or ".." in new_name:
        return {"success": False, "error": "Invalid filename"}

    try:
        gallery_path = get_gallery_folder_path(folder)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    old_path = gallery_path / filename
    new_path = gallery_path / new_name

    if not old_path.exists():
        return {"success": False, "error": "Image not found"}

    if new_path.exists() and old_path != new_path:
        return {"success": False, "error": "File with this name already exists"}

    try:
        old_path.rename(new_path)
        return {"success": True, "new_filename": new_name}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/gallery/open-folder")
async def open_gallery_folder(request: dict = None):
    """갤러리 폴더 열기"""
    try:
        # 서브폴더 지정 시 해당 폴더 열기
        subfolder = request.get("folder", "") if request else ""
        target_folder = get_gallery_folder_path(subfolder)
        folder_path = str(target_folder.absolute())

        if open_folder_in_explorer(folder_path):
            return {"success": True}
        else:
            return {"success": False, "error": "Failed to open folder"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# === Vibe Cache API ===

@app.get("/api/vibe-cache")
async def get_vibe_cache():
    """바이브 캐시 목록 조회"""
    vibes = []

    if not VIBE_CACHE_DIR.exists():
        return {"vibes": []}

    for filepath in sorted(VIBE_CACHE_DIR.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            image = Image.open(filepath)
            metadata = {}

            if hasattr(image, 'info'):
                for key, value in image.info.items():
                    if isinstance(value, str):
                        metadata[key] = value

            # 썸네일 생성
            thumb = image.copy()
            thumb.thumbnail((260, 260), Image.LANCZOS)
            buffer = io.BytesIO()
            thumb.save(buffer, format="PNG")
            thumb_base64 = base64.b64encode(buffer.getvalue()).decode()

            vibes.append({
                "filename": filepath.name,
                "thumbnail": thumb_base64,
                "model": metadata.get("model", "unknown"),
                "strength": metadata.get("strength", "0.6"),
                "info_extracted": metadata.get("info_extracted", "1.0"),
                "has_vibe_data": "vibe_data" in metadata
            })
        except Exception as e:
            print(f"[VibeCache] Error loading {filepath}: {e}")
            continue

    return {"vibes": vibes}


@app.get("/api/vibe-cache/{filename}")
async def get_vibe_cache_file(filename: str):
    """바이브 캐시 파일 상세 정보 (vibe_data 포함)"""
    filepath = VIBE_CACHE_DIR / filename

    if not filepath.exists():
        return {"success": False, "error": "File not found"}

    try:
        image = Image.open(filepath)
        metadata = {}

        if hasattr(image, 'info'):
            for key, value in image.info.items():
                if isinstance(value, str):
                    metadata[key] = value

        # 원본 이미지 base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        return {
            "success": True,
            "filename": filename,
            "image": image_base64,
            "vibe_data": metadata.get("vibe_data"),
            "model": metadata.get("model", "unknown"),
            "strength": float(metadata.get("strength", 0.6)),
            "info_extracted": float(metadata.get("info_extracted", 1.0))
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/api/vibe-cache/{filename}")
async def delete_vibe_cache_file(filename: str):
    """바이브 캐시 파일 삭제"""
    filepath = VIBE_CACHE_DIR / filename

    if not filepath.exists():
        return {"success": False, "error": "File not found"}

    try:
        # key_map.json에서도 제거
        key_map_file = VIBE_CACHE_DIR / "key_map.json"
        if key_map_file.exists():
            try:
                key_map = json.loads(key_map_file.read_text(encoding='utf-8'))
                # filename과 일치하는 캐시 키 찾아서 제거
                keys_to_remove = [k for k, v in key_map.items() if v == filename]
                for k in keys_to_remove:
                    del key_map[k]
                key_map_file.write_text(json.dumps(key_map, indent=2), encoding='utf-8')
            except:
                pass

        filepath.unlink()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/")
async def serve_index():
    """Serve index.html (캐시 비활성화)"""
    index_path = APP_DIR / "index.html"
    if index_path.exists():
        return FileResponse(
            index_path,
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"error": "index.html not found"}


@app.get("/assets/{filepath:path}")
async def serve_assets(filepath: str):
    """Serve static assets"""
    file_path = APP_DIR / "assets" / filepath
    if not file_path.exists() or not file_path.is_file():
        return {"error": "File not found"}

    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    return FileResponse(file_path, media_type=media_type)


@app.get("/data/{filepath:path}")
async def serve_data(filepath: str):
    """Serve data files (e.g., tags.json)"""
    file_path = APP_DIR / "data" / filepath
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".json": "application/json",
        ".txt": "text/plain",
        ".csv": "text/csv",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    return FileResponse(file_path, media_type=media_type)


@app.get("/api/config")
async def get_config():
    return {
        "nai_token_set": bool(CONFIG.get("nai_token")),
        "checkpoints_dir": CONFIG.get("checkpoints_dir", str(CHECKPOINTS_DIR)),
        "lora_dir": CONFIG.get("lora_dir", str(LORA_DIR)),
    }


@app.post("/api/open-folder")
async def open_folder(request: dict):
    """폴더를 파일 탐색기로 열기 (Windows/macOS/Linux)"""
    folder_type = request.get("folder", "")
    subfolder = request.get("subfolder", "")  # outputs 서브폴더 지원

    folder_map = {
        "outputs": OUTPUT_DIR,
        "vibe_cache": VIBE_CACHE_DIR,
        "gallery": GALLERY_DIR,
        "checkpoints": Path(CONFIG.get("checkpoints_dir", CHECKPOINTS_DIR)),
        "loras": Path(CONFIG.get("lora_dir", LORA_DIR)),
        "uncensored": UNCENSORED_DIR,
        "censored": CENSORED_DIR,
    }

    folder_path = folder_map.get(folder_type)
    if not folder_path:
        return {"error": "Unknown folder type"}

    # 서브폴더가 지정된 경우
    if subfolder and folder_type in ["outputs", "uncensored", "censored"]:
        folder_path = folder_path / subfolder

    folder_path.mkdir(parents=True, exist_ok=True)

    try:
        abs_path = str(folder_path.resolve())
        if open_folder_in_explorer(abs_path):
            return {"success": True, "path": abs_path}
        else:
            return {"error": "Failed to open folder"}
    except Exception as e:
        print(f"[OpenFolder] Error: {e}")
        return {"error": str(e)}


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    global CONFIG
    if update.nai_token is not None:
        # 토큰 정리: 앞뒤 공백 제거
        token = update.nai_token.strip()
        # ASCII 문자만 허용 (JWT 토큰은 base64로 ASCII만 포함)
        try:
            token.encode('ascii')
        except UnicodeEncodeError:
            return {"success": False, "error": "NAI 토큰에 유효하지 않은 문자가 포함되어 있습니다. 토큰을 다시 복사해주세요."}
        CONFIG["nai_token"] = token
    if update.checkpoints_dir is not None:
        CONFIG["checkpoints_dir"] = update.checkpoints_dir
    if update.lora_dir is not None:
        CONFIG["lora_dir"] = update.lora_dir
    save_config(CONFIG)
    return {"success": True}


@app.get("/api/nai/subscription")
async def get_nai_subscription():
    """NAI 구독 정보 및 Anlas 잔액 조회"""
    import httpx

    token = CONFIG.get("nai_token", "")
    if not token:
        return {"error": "NAI token not set", "anlas": None}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                "https://api.novelai.net/user/subscription",
                headers={"Authorization": f"Bearer {token}"}
            )

            if response.status_code != 200:
                return {"error": f"API error: {response.status_code}", "anlas": None}

            data = response.json()

            # trainingStepsLeft 구조 확인
            training_steps = data.get("trainingStepsLeft", {})

            # trainingStepsLeft가 숫자인 경우 (구버전 API)
            if isinstance(training_steps, (int, float)):
                total_anlas = int(training_steps)
                subscription_anlas = total_anlas
                fixed_anlas = 0
            # trainingStepsLeft가 dict인 경우 (현재 API)
            elif isinstance(training_steps, dict):
                subscription_anlas = training_steps.get("fixedTrainingStepsLeft", 0) or 0
                fixed_anlas = training_steps.get("purchasedTrainingSteps", 0) or 0
                total_anlas = subscription_anlas + fixed_anlas
            else:
                total_anlas = 0
                subscription_anlas = 0
                fixed_anlas = 0

            return {
                "anlas": total_anlas,
                "subscription_anlas": subscription_anlas,
                "fixed_anlas": fixed_anlas,
                "tier": data.get("tier", 0),
                "active": data.get("active", False)
            }
    except Exception as e:
        return {"error": str(e), "anlas": None}


def calculate_anlas_cost(width: int, height: int, steps: int, is_opus: bool = False,
                         vibe_count: int = 0, has_char_ref: bool = False,
                         strength: float = 1.0) -> int:
    """NAI 이미지 생성 Anlas 소모량 계산"""
    pixels = width * height
    base_pixels = 1024 * 1024

    # Opus 무료 조건: 1MP 이하, 28 steps 이하, vibe/char_ref 없음
    if is_opus and pixels <= base_pixels and steps <= 28 and vibe_count <= 0 and not has_char_ref:
        return 0

    # 기본 비용 계산: ceil(MP * 20)
    if is_opus and pixels <= base_pixels and steps <= 28:
        # Opus는 기본 생성 무료, 추가 기능만 비용
        base_cost = 0
    else:
        # NAI 공식: ceil(megapixels * 20)
        base_cost = math.ceil(pixels / base_pixels * 20)

    # Steps 보정 (28 초과시)
    if steps > 28 and base_cost > 0:
        base_cost = int(base_cost * (steps / 28))

    # img2img strength 보정 (strength < 1.0이면 비용 감소)
    if strength < 1.0 and base_cost > 0:
        base_cost = max(math.ceil(base_cost * strength), 2)

    # Vibe Transfer (첫 사용 시 2 Anlas, 이후 무료)
    if vibe_count >= 1:
        base_cost += 2

    # Character Reference (Opus: 5 Anlas, 일반: 15 Anlas)
    if has_char_ref:
        base_cost += 5 if is_opus else 15

    return base_cost


@app.post("/api/nai/calculate-cost")
async def calculate_cost(request: dict):
    """Anlas 소모량 계산 (vibe 캐시 상태 포함)"""
    width = request.get("width", 832)
    height = request.get("height", 1216)
    steps = request.get("steps", 28)
    is_opus = request.get("is_opus", False)
    vibe_count = request.get("vibe_count", 0)
    has_char_ref = request.get("has_char_ref", False)
    count = request.get("count", 1)  # 생성 횟수

    # Vibe 캐시 체크 (vibes 배열이 제공된 경우)
    vibes = request.get("vibes", [])
    model = request.get("model", "nai-diffusion-4-5-full")
    uncached_vibe_count = 0

    if vibes and "diffusion-4" in model:
        base_model = model.replace("-inpainting", "")
        for v in vibes:
            # 이미 인코딩된 바이브: 모델 및 info_extracted 일치 여부 확인
            if v.get("encoded"):
                encoded_model = v.get("encoded_model", "")
                encoded_base_model = encoded_model.replace("-inpainting", "") if encoded_model else ""

                # 현재 info_extracted vs 인코딩 시 사용된 값 비교
                current_info = v.get("info_extracted", 1.0)
                encoded_info = v.get("encoded_info_extracted", current_info)

                # 모델 또는 info_extracted 불일치 시 재인코딩 필요
                if encoded_base_model != base_model or abs(current_info - encoded_info) > 0.001:
                    uncached_vibe_count += 1
                continue

            image_base64 = v.get("image", "")
            info_extracted = v.get("info_extracted", 1.0)
            if image_base64:
                # PNG로 변환 후 캐시 키 생성
                try:
                    png_image = ensure_png_base64(image_base64)
                    cache_key = get_vibe_cache_key(png_image, model, info_extracted)
                    if not get_cached_vibe(cache_key):
                        uncached_vibe_count += 1
                except:
                    uncached_vibe_count += 1
    else:
        uncached_vibe_count = vibe_count

    strength = request.get("strength", 1.0)
    cost_per_image = calculate_anlas_cost(width, height, steps, is_opus, uncached_vibe_count, has_char_ref, strength)
    total_cost = cost_per_image * count

    # Vibe 인코딩 비용 (캐시되지 않은 것만, 첫 이미지에서만 발생)
    vibe_encoding_cost = uncached_vibe_count * 2 if uncached_vibe_count > 0 else 0

    return {
        "cost_per_image": cost_per_image,
        "total_cost": total_cost,
        "count": count,
        "is_free": cost_per_image == 0,
        "vibe_encoding_cost": vibe_encoding_cost,
        "cached_vibes": len(vibes) - uncached_vibe_count if vibes else 0,
        "uncached_vibes": uncached_vibe_count
    }


@app.get("/api/models")
async def list_models():
    checkpoints_dir = Path(CONFIG.get("checkpoints_dir", CHECKPOINTS_DIR))
    models = []
    if checkpoints_dir.exists():
        for f in checkpoints_dir.iterdir():
            if f.suffix in (".safetensors", ".ckpt"):
                models.append(f.name)
    return {"models": sorted(models)}


@app.get("/api/loras")
async def list_loras():
    lora_dir = Path(CONFIG.get("lora_dir", LORA_DIR))
    loras = []
    if lora_dir.exists():
        for f in lora_dir.iterdir():
            if f.suffix == ".safetensors":
                loras.append(f.name)
    return {"loras": sorted(loras)}


@app.get("/api/upscale_models")
async def list_upscale_models():
    models = []
    if UPSCALE_DIR.exists():
        for f in UPSCALE_DIR.iterdir():
            if f.suffix in (".pth", ".safetensors", ".pt"):
                models.append(f.name)
    return {"models": sorted(models)}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    try:
        if req.provider == "nai":
            image, seed = await call_nai_api(req)
        else:
            # ComfyUI 포팅 로컬 엔진 사용
            image, seed = call_local_engine(req)

        return {
            "success": True,
            "image": image_to_base64(image),
            "seed": seed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/multi")
async def generate_multi(req: MultiGenerateRequest):
    """큐에 작업 추가"""
    job_id = gen_queue.add_job(req)

    # 이 job의 이미지 수 계산
    prompt_count = len(req.prompt_list) if req.prompt_list else 1

    # 즉시 total에 반영 (job 시작 전에도 프로그레스바에 표시)
    gen_queue.total_images += prompt_count

    # 모든 클라이언트에게 큐 추가 알림
    await gen_queue.broadcast({
        "type": "job_queued",
        "job_id": job_id,
        "job_images": prompt_count,
        "progress": {
            "completed": gen_queue.completed_images,
            "total": gen_queue.total_images,
            "queue_length": len(gen_queue.queue)
        }
    })

    return {
        "success": True,
        "job_id": job_id,
        "queue_length": len(gen_queue.queue),
        "message": f"Job {job_id} added to queue"
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, clientId: str = None):
    """WebSocket 연결 - 실시간 이미지 생성 업데이트"""
    # 클라이언트 ID 생성 또는 재사용
    client_id = clientId or str(uuid.uuid4())[:8]

    await websocket.accept()
    print(f"[WS] Client connected: {client_id}")

    # 기존 연결이 있으면 교체 (재연결)
    old_ws = gen_queue.clients.get(client_id)
    if old_ws:
        try:
            await old_ws.close()
        except:
            pass
    gen_queue.clients[client_id] = websocket

    # 초기 상태 전송
    await websocket.send_json({
        "type": "connected",
        "client_id": client_id,
        "status": gen_queue.get_status()
    })

    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "sync":
                # 재연결 시 누락된 이미지 동기화
                last_seq = data.get("last_seq", 0)
                missed_images = gen_queue.get_images_since(last_seq)
                await websocket.send_json({
                    "type": "sync",
                    "images": missed_images,
                    "status": gen_queue.get_status()
                })
                print(f"[WS] Sync requested by {client_id}: last_seq={last_seq}, sending {len(missed_images)} images")

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {client_id}")
    except Exception as e:
        print(f"[WS] Error with client {client_id}: {e}")
    finally:
        gen_queue.clients.pop(client_id, None)


@app.post("/api/cancel-current")
async def cancel_current():
    """현재 작업 취소 (현재 이미지 완료 후)"""
    if gen_queue.is_processing:
        gen_queue.cancel_current_job()
        return {"success": True, "message": "Current job will be cancelled after current image"}
    return {"success": False, "message": "No job is currently running"}


@app.post("/api/clear-queue")
async def clear_queue():
    """대기 큐 비우기 (현재 작업은 계속)"""
    cleared_jobs, cleared_images = gen_queue.clear_queue()
    # total에서 대기 중이던 이미지 수 차감
    gen_queue.total_images -= cleared_images
    if gen_queue.total_images < gen_queue.completed_images:
        gen_queue.total_images = gen_queue.completed_images
    await gen_queue.broadcast({
        "type": "queue_cleared",
        "cleared_jobs": cleared_jobs,
        "cleared_images": cleared_images,
        "progress": {
            "completed": gen_queue.completed_images,
            "total": gen_queue.total_images,
            "queue_length": 0
        }
    })
    return {"success": True, "cleared_jobs": cleared_jobs, "cleared_images": cleared_images}


@app.get("/api/queue")
async def get_queue():
    """큐 상태 조회"""
    return gen_queue.get_status()


@app.post("/api/cache/clear")
async def clear_cache():
    model_cache.clear()
    return {"success": True}


@app.get("/api/status")
async def status():
    return {
        "device": model_cache.device if torch else "unknown",
        "cached_model": model_cache.model_path,
        "loaded_loras": list(model_cache.loaded_loras.keys()),
        "nai_available": bool(CONFIG.get("nai_token")),
    }


@app.get("/api/outputs")
async def list_outputs():
    files = []
    for f in sorted(OUTPUT_DIR.iterdir(), reverse=True)[:100]:
        if f.suffix == ".png":
            files.append(f.name)
    return {"files": files}


@app.get("/api/outputs/{filepath:path}")
async def get_output_image(filepath: str):
    """출력 이미지 파일 서빙"""
    from fastapi.responses import FileResponse
    file_path = OUTPUT_DIR / filepath
    if not file_path.exists() or not file_path.is_file():
        return {"error": "File not found"}
    return FileResponse(file_path, media_type="image/png")


# === Output Folder API ===
@app.get("/api/output-folders")
async def list_output_folders():
    """output 폴더 내 하위 폴더 목록 반환"""
    folders = []
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.iterdir()):
            if f.is_dir():
                # 폴더 내 이미지 수 카운트
                image_count = len([p for p in f.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']])
                folders.append({"name": f.name, "image_count": image_count})
    return {"success": True, "folders": folders}


@app.post("/api/output-folders")
async def create_output_folder(req: dict):
    """output 폴더 내에 새 하위 폴더 생성"""
    folder_name = req.get("name", "").strip()
    if not folder_name:
        return {"success": False, "error": "폴더명이 필요합니다"}

    # 안전한 폴더명 확인
    if "/" in folder_name or "\\" in folder_name or ".." in folder_name:
        return {"success": False, "error": "잘못된 폴더명입니다"}

    folder_path = OUTPUT_DIR / folder_name
    if folder_path.exists():
        return {"success": False, "error": "이미 존재하는 폴더입니다"}

    try:
        folder_path.mkdir(parents=True, exist_ok=True)
        return {"success": True, "name": folder_name}
    except Exception as e:
        return {"success": False, "error": str(e)}


# === Preset API ===
class PresetSlot(BaseModel):
    name: str = ""
    content: str = ""

class PresetData(BaseModel):
    name: str
    slots: List[PresetSlot]
    prefix: str = "Name_"

@app.get("/api/presets")
async def list_presets():
    """프리셋 목록 반환"""
    presets = []
    if PRESETS_DIR.exists():
        for f in sorted(PRESETS_DIR.iterdir()):
            if f.suffix == ".json":
                try:
                    data = json.loads(f.read_text(encoding='utf-8'))
                    presets.append({"name": data.get("name", f.stem), "filename": f.name})
                except:
                    pass
    return {"presets": presets}

@app.get("/api/presets/{filename}")
async def get_preset(filename: str):
    """특정 프리셋 로드"""
    filepath = PRESETS_DIR / filename
    if not filepath.exists() or not filepath.suffix == ".json":
        raise HTTPException(status_code=404, detail="Preset not found")
    
    data = json.loads(filepath.read_text(encoding='utf-8'))
    return data

@app.post("/api/presets")
async def create_preset(preset: PresetData):
    """새 프리셋 저장"""
    # 파일명 생성 (특수문자 제거)
    safe_name = "".join(c for c in preset.name if c.isalnum() or c in " _-").strip()
    if not safe_name:
        safe_name = "preset"
    
    filename = f"{safe_name}.json"
    filepath = PRESETS_DIR / filename
    
    # 중복 시 숫자 추가
    counter = 1
    while filepath.exists():
        filename = f"{safe_name}_{counter}.json"
        filepath = PRESETS_DIR / filename
        counter += 1
    
    data = {
        "name": preset.name,
        "prefix": preset.prefix,
        "slots": [{"name": s.name, "content": s.content} for s in preset.slots]
    }
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    
    return {"filename": filename, "name": preset.name}

@app.put("/api/presets/{filename}")
async def update_preset(filename: str, preset: PresetData):
    """프리셋 업데이트 (이름 변경 또는 내용 덮어쓰기)"""
    filepath = PRESETS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Preset not found")
    
    data = {
        "name": preset.name,
        "prefix": preset.prefix,
        "slots": [{"name": s.name, "content": s.content} for s in preset.slots]
    }
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    
    return {"filename": filename, "name": preset.name}

@app.delete("/api/presets/{filename}")
async def delete_preset(filename: str):
    """프리셋 삭제"""
    filepath = PRESETS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Preset not found")
    
    filepath.unlink()
    return {"deleted": filename}


# === Prompt Presets API ===
class PromptPresetData(BaseModel):
    name: str
    content: str

@app.get("/api/prompts/{category}")
async def list_prompt_presets(category: str):
    """프롬프트 프리셋 목록 (category: base, negative, character)"""
    if category not in ['base', 'negative', 'character']:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    folder = PROMPTS_DIR / category
    presets = []
    if folder.exists():
        for f in sorted(folder.iterdir()):
            if f.suffix == ".txt":
                presets.append({"name": f.stem, "filename": f.name})
    return {"presets": presets}

@app.get("/api/prompts/{category}/{filename}")
async def get_prompt_preset(category: str, filename: str):
    """프롬프트 프리셋 내용 가져오기"""
    if category not in ['base', 'negative', 'character']:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    filepath = PROMPTS_DIR / category / filename
    if not filepath.exists() or filepath.suffix != ".txt":
        raise HTTPException(status_code=404, detail="Prompt preset not found")
    
    content = filepath.read_text(encoding='utf-8')
    return {"name": filepath.stem, "content": content}

@app.post("/api/prompts/{category}")
async def create_prompt_preset(category: str, data: PromptPresetData):
    """프롬프트 프리셋 저장"""
    if category not in ['base', 'negative', 'character']:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    # 파일명 생성
    safe_name = "".join(c for c in data.name if c.isalnum() or c in " _-가-힣").strip()
    if not safe_name:
        safe_name = "prompt"
    
    filename = f"{safe_name}.txt"
    filepath = PROMPTS_DIR / category / filename
    
    # 중복 시 숫자 추가
    counter = 1
    while filepath.exists():
        filename = f"{safe_name}_{counter}.txt"
        filepath = PROMPTS_DIR / category / filename
        counter += 1
    
    filepath.write_text(data.content, encoding='utf-8')
    return {"filename": filename, "name": safe_name}

@app.delete("/api/prompts/{category}/{filename}")
async def delete_prompt_preset(category: str, filename: str):
    """프롬프트 프리셋 삭제"""
    if category not in ['base', 'negative', 'character']:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    filepath = PROMPTS_DIR / category / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Prompt preset not found")
    
    filepath.unlink()
    return {"deleted": filename}


# ============================================================
# Local Environment Installation System
# ============================================================
import subprocess
import zipfile
import tarfile
import shutil
import platform
import urllib.request

# 설치 상태 추적 (런타임)
install_status = {
    "installing": False,
    "progress": 0,
    "message": "",
    "error": None
}

def _get_site_packages_dir():
    """site-packages 경로 반환"""
    if platform.system() == "Windows":
        return PYTHON_ENV_DIR / "Lib" / "site-packages"
    else:
        return PYTHON_ENV_DIR / "python" / "lib" / f"python{PYTHON_VERSION[:4]}" / "site-packages"

# 설치 상태 파일 (영구 저장 - 매번 파일시스템 체크 방지)
INSTALL_STATUS_FILE = PYTHON_ENV_DIR / "install_status.json"

def _check_python_installed():
    """Python 설치 여부 실제 확인"""
    if platform.system() == "Windows":
        return (PYTHON_ENV_DIR / "python.exe").exists()
    else:
        return (PYTHON_ENV_DIR / "python" / "bin" / "python3").exists()

def _check_torch_version():
    """torch 버전 확인 (None/cpu/cuda)"""
    torch_dir = _get_site_packages_dir() / "torch"
    if not torch_dir.exists():
        return None
    
    # CUDA 버전 확인: torch/lib에 cudnn 또는 cublas 파일이 있으면 CUDA
    torch_lib = torch_dir / "lib"
    if torch_lib.exists():
        # 전체 순회 대신 glob으로 빠르게 확인
        cuda_patterns = ["*cudnn*", "*cublas*", "*nvcuda*"]
        for pattern in cuda_patterns:
            if list(torch_lib.glob(pattern)):
                return "cuda"
    return "cpu"

def _check_ultralytics_installed():
    """ultralytics 설치 여부 실제 확인"""
    return (_get_site_packages_dir() / "ultralytics").exists()

def _check_diffusers_installed():
    """diffusers 설치 여부 실제 확인 (로컬 생성용)"""
    return (_get_site_packages_dir() / "diffusers").exists()

def _detect_install_status():
    """실제 파일 시스템을 스캔하여 설치 상태 감지 (느림)"""
    return {
        "python": _check_python_installed(),
        "torch": _check_torch_version(),
        "censor": _check_ultralytics_installed(),
        "local": _check_diffusers_installed()
    }

def load_install_status():
    """설치 상태 파일 로드 (없으면 실제 감지 후 저장)"""
    default_status = {
        "python": False,
        "torch": None,
        "censor": False,
        "local": False
    }
    
    # 파일이 있으면 로드
    if INSTALL_STATUS_FILE.exists():
        try:
            status = json.loads(INSTALL_STATUS_FILE.read_text(encoding='utf-8'))
            return {**default_status, **status}
        except:
            pass
    
    # 파일이 없으면 실제 감지 (배포판 첫 실행)
    # Python이 있으면 배포판이므로 실제 상태 감지 후 저장
    if _check_python_installed():
        status = _detect_install_status()
        save_install_status(status)
        return status
    
    return default_status

def save_install_status(status: dict):
    """설치 상태 파일 저장"""
    PYTHON_ENV_DIR.mkdir(parents=True, exist_ok=True)
    INSTALL_STATUS_FILE.write_text(json.dumps(status, indent=2), encoding='utf-8')

def get_install_status():
    """설치 상태 반환 (캐시된 파일에서 로드)"""
    return load_install_status()

def get_env_status():
    """환경 설치 상태 반환 (API용)"""
    status = get_install_status()
    
    return {
        "installed": status["python"],
        "torch_version": status["torch"],
        "censor_ready": status["censor"],
        "local_ready": status["local"],
        "installing": install_status["installing"],
        "progress": install_status["progress"],
        "message": install_status["message"],
        "error": install_status["error"]
    }

# Python 버전 및 URL (python-build-standalone)
PYTHON_VERSION = "3.11.9"
PYTHON_BUILD_DATE = "20240415"

def get_python_download_url():
    """플랫폼에 맞는 Python URL 반환 (빌드와 동일)"""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        # Windows: python.org embedded Python (빌드와 동일)
        if machine in ["amd64", "x86_64"]:
            return f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
        else:
            return f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-win32.zip"
    elif system == "darwin":
        # macOS: python-build-standalone
        if machine == "arm64":
            arch = "aarch64"
        else:
            arch = "x86_64"
        filename = f"cpython-{PYTHON_VERSION}+{PYTHON_BUILD_DATE}-{arch}-apple-darwin-install_only.tar.gz"
        return f"https://github.com/indygreg/python-build-standalone/releases/download/{PYTHON_BUILD_DATE}/{filename}"
    else:  # linux
        # Linux: python-build-standalone
        if machine == "aarch64":
            arch = "aarch64"
        else:
            arch = "x86_64"
        filename = f"cpython-{PYTHON_VERSION}+{PYTHON_BUILD_DATE}-{arch}-unknown-linux-gnu-install_only.tar.gz"
        return f"https://github.com/indygreg/python-build-standalone/releases/download/{PYTHON_BUILD_DATE}/{filename}"


def get_uv_download_url():
    """플랫폼에 맞는 uv URL 반환"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        if machine in ["amd64", "x86_64"]:
            return "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
        else:
            return "https://github.com/astral-sh/uv/releases/latest/download/uv-i686-pc-windows-msvc.zip"
    elif system == "darwin":
        if machine == "arm64":
            return "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz"
        else:
            return "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-apple-darwin.tar.gz"
    else:
        if machine == "aarch64":
            return "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-unknown-linux-gnu.tar.gz"
        else:
            return "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"


def is_local_env_installed():
    """로컬 환경 설치 여부 확인 - torch와 diffusers 둘 다 필요"""
    if platform.system() == "Windows":
        # Windows embedded Python: python/python.exe
        python_exe = PYTHON_ENV_DIR / "python.exe"
        site_packages = PYTHON_ENV_DIR / "Lib" / "site-packages"
    else:
        # macOS/Linux: python-build-standalone
        python_exe = PYTHON_ENV_DIR / "python" / "bin" / "python3"
        site_packages = PYTHON_ENV_DIR / "python" / "lib" / f"python{PYTHON_VERSION[:4]}" / "site-packages"

    torch_check = site_packages / "torch"
    diffusers_check = site_packages / "diffusers"

    python_exists = python_exe.exists()
    torch_exists = torch_check.exists()
    diffusers_exists = diffusers_check.exists()

    print(f"[Local Env Check] Python: {python_exe} exists={python_exists}")
    print(f"[Local Env Check] Torch: {torch_check} exists={torch_exists}")
    print(f"[Local Env Check] Diffusers: {diffusers_check} exists={diffusers_exists}")

    # torch와 diffusers 둘 다 있어야 설치된 것으로 인정
    return python_exists and torch_exists and diffusers_exists


def download_file(url: str, dest: Path, progress_callback=None):
    """파일 다운로드 with 진행률"""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB
            
            with open(dest, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size:
                        progress_callback(downloaded, total_size)
        return True
    except Exception as e:
        print(f"[Download Error] {url}: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path):
    """압축 해제"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(dest_dir)
    elif archive_path.name.endswith('.tar.gz') or archive_path.suffix == '.tar':
        # Windows에서 tarfile 인코딩 문제 방지 - tar 명령어 사용
        if platform.system() == "Windows":
            import subprocess
            # Windows 10 이상에 tar 기본 포함
            subprocess.run(
                ["tar", "-xzf", str(archive_path), "-C", str(dest_dir)],
                check=True,
                capture_output=True,
                encoding='utf-8',
                errors='replace'
            )
        else:
            mode = 'r:gz' if archive_path.name.endswith('.tar.gz') else 'r'
            with tarfile.open(archive_path, mode) as tf:
                tf.extractall(dest_dir)


def _setup_python_and_uv():
    """Python과 uv 설치 (공통 부분)"""
    global install_status
    
    PYTHON_ENV_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = PYTHON_ENV_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Python과 uv 경로
    if platform.system() == "Windows":
        # Windows embedded Python: python/python.exe (빌드와 동일)
        python_exe = PYTHON_ENV_DIR / "python.exe"
        uv_exe = PYTHON_ENV_DIR / "uv.exe"
    else:
        # macOS/Linux: python-build-standalone은 python/bin/python3
        python_exe = PYTHON_ENV_DIR / "python" / "bin" / "python3"
        uv_exe = PYTHON_ENV_DIR / "uv"
    
    # Python이 이미 있으면 다운로드 스킵
    if not python_exe.exists():
        install_status["message"] = "Downloading Python..."
        install_status["progress"] = 5

        python_url = get_python_download_url()
        # Windows는 ZIP, 다른 플랫폼은 tar.gz
        if platform.system() == "Windows":
            python_archive = temp_dir / "python.zip"
        else:
            python_archive = temp_dir / "python.tar.gz"

        def python_progress(downloaded, total):
            install_status["progress"] = 5 + int((downloaded / total) * 15)

        if not download_file(python_url, python_archive, python_progress):
            raise Exception("Failed to download Python")

        install_status["message"] = "Extracting Python..."
        install_status["progress"] = 20

        if platform.system() == "Windows":
            # Windows embedded Python: 직접 PYTHON_ENV_DIR (python/)에 압축 해제
            extract_archive(python_archive, PYTHON_ENV_DIR)

            # python311._pth 파일 수정 (import site 활성화)
            pth_file = PYTHON_ENV_DIR / f"python{PYTHON_VERSION.replace('.', '')[:3]}._pth"
            if pth_file.exists():
                content = pth_file.read_text(encoding='utf-8', errors='replace')
                content = content.replace("#import site", "import site")
                pth_file.write_text(content, encoding='utf-8')

            # get-pip.py 다운로드 및 설치
            install_status["message"] = "Installing pip..."
            get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
            get_pip_file = temp_dir / "get-pip.py"
            if download_file(get_pip_url, get_pip_file):
                subprocess.run(
                    [str(python_exe), str(get_pip_file), "--no-warn-script-location"],
                    check=True,
                    encoding='utf-8',
                    errors='replace'
                )
        else:
            # macOS/Linux: python-build-standalone
            extract_archive(python_archive, PYTHON_ENV_DIR)
    
    install_status["progress"] = 25
    
    # uv가 이미 있으면 다운로드 스킵
    if not uv_exe.exists():
        install_status["message"] = "Downloading uv..."
        install_status["progress"] = 25
        
        uv_url = get_uv_download_url()
        uv_archive = temp_dir / ("uv.zip" if platform.system() == "Windows" else "uv.tar.gz")
        
        def uv_progress(downloaded, total):
            install_status["progress"] = 25 + int((downloaded / total) * 10)
        
        if not download_file(uv_url, uv_archive, uv_progress):
            raise Exception("Failed to download uv")
        
        install_status["message"] = "Extracting uv..."
        uv_dir = temp_dir / "uv_extracted"
        extract_archive(uv_archive, uv_dir)
        
        # uv 실행파일 찾기 및 복사
        if platform.system() == "Windows":
            uv_found = None
            for p in uv_dir.rglob("uv.exe"):
                uv_found = p
                break
            if uv_found:
                shutil.copy(uv_found, PYTHON_ENV_DIR / "uv.exe")
        else:
            uv_found = None
            for p in uv_dir.rglob("uv"):
                if p.is_file():
                    uv_found = p
                    break
            if uv_found:
                dest = PYTHON_ENV_DIR / "uv"
                shutil.copy(uv_found, dest)
                dest.chmod(0o755)
    
    install_status["progress"] = 35
    return python_exe, uv_exe, temp_dir


def _run_uv_install(uv_exe, python_exe, packages, index_url=None, progress_base=40, progress_end=70, reinstall_packages=None):
    """uv로 패키지 설치

    Args:
        reinstall_packages: 재설치할 특정 패키지 목록 (예: ["torch", "torchvision"])
    """
    global install_status

    env = os.environ.copy()
    env["UV_PYTHON"] = str(python_exe)

    cmd = [str(uv_exe), "pip", "install", "--python", str(python_exe)]
    if index_url:
        cmd.extend(["--index-url", index_url])
    # 특정 패키지만 재설치 (의존성은 건드리지 않음)
    if reinstall_packages:
        for pkg in reinstall_packages:
            cmd.extend(["--reinstall-package", pkg])
    cmd.extend(packages)
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',  # 디코딩 불가능한 문자는 ? 로 대체
        env=env
    )
    
    install_keywords = ["Downloading", "Installing", "Prepared", "Built", "Resolved"]
    line_count = 0
    for line in proc.stdout:
        line_count += 1
        print(f"[uv] {line.strip()}")
        if any(kw in line for kw in install_keywords):
            progress = min(progress_base + (line_count // 3), progress_end)
            install_status["progress"] = progress
    
    proc.wait()
    return proc.returncode


def _install_base_environment_sync():
    """기본 환경 설치 (첫 실행 시 자동) - torch CPU + ultralytics"""
    global install_status
    
    try:
        python_exe, uv_exe, temp_dir = _setup_python_and_uv()
        
        # torch CPU 설치 (35% -> 60%)
        install_status["message"] = "Installing PyTorch (CPU)..."
        install_status["progress"] = 35
        
        ret = _run_uv_install(
            uv_exe, python_exe,
            ["torch==2.5.1", "torchvision==0.20.1"],
            index_url="https://download.pytorch.org/whl/cpu",
            progress_base=35,
            progress_end=60
        )
        if ret != 0:
            raise Exception(f"PyTorch CPU installation failed with code {ret}")
        
        # ultralytics + opencv 설치 (60% -> 80%)
        install_status["message"] = "Installing ultralytics..."
        install_status["progress"] = 60
        
        ret = _run_uv_install(
            uv_exe, python_exe,
            ["ultralytics", "opencv-python"],
            progress_base=60,
            progress_end=80
        )
        if ret != 0:
            raise Exception(f"ultralytics installation failed with code {ret}")
        
        # backend dependencies (80% -> 95%)
        install_status["message"] = "Installing backend dependencies..."
        install_status["progress"] = 80
        
        ret = _run_uv_install(
            uv_exe, python_exe,
            ["fastapi", "uvicorn[standard]", "httpx", "aiofiles", "python-multipart", "pillow", "piexif", "websockets"],
            progress_base=80,
            progress_end=95
        )
        if ret != 0:
            raise Exception(f"Backend dependencies installation failed with code {ret}")
        
        # 정리 및 상태 저장
        install_status["message"] = "Cleaning up..."
        install_status["progress"] = 95
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # 설치 완료 - 상태 저장
        save_install_status({
            "python": True,
            "torch": "cpu",
            "censor": True,
            "local": False
        })

        install_status["progress"] = 100
        install_status["message"] = "Installation complete!"
        install_status["installing"] = False

        # 기본 환경 설치 완료 후 상태 리셋 (로컬 환경과 구분)
        # 잠시 후 리셋하여 프론트엔드가 progress=100을 "로컬 설치 완료"로 오인하지 않도록 함
        import time
        time.sleep(0.5)
        install_status["progress"] = 0
        install_status["message"] = ""

        return True
        
    except Exception as e:
        install_status["error"] = str(e)
        install_status["installing"] = False
        print(f"[Install Error] {e}")
        return False


def _install_local_environment_sync():
    """로컬 생성 환경 설치 - torch CUDA + diffusers"""
    global install_status

    try:
        # Python과 uv 경로 (_setup_python_and_uv와 동일하게)
        if platform.system() == "Windows":
            # Windows embedded Python: python/python.exe (직접)
            python_exe = PYTHON_ENV_DIR / "python.exe"
            uv_exe = PYTHON_ENV_DIR / "uv.exe"
        else:
            # macOS/Linux: python-build-standalone
            python_exe = PYTHON_ENV_DIR / "python" / "bin" / "python3"
            uv_exe = PYTHON_ENV_DIR / "uv"

        # 기본 환경이 없으면 먼저 설치
        if not python_exe.exists() or not uv_exe.exists():
            python_exe, uv_exe, temp_dir = _setup_python_and_uv()
        
        # torch CUDA로 업그레이드 (0% -> 50%)
        # reinstall_packages로 torch/torchvision만 재설치 (의존성은 건드리지 않음)
        install_status["message"] = "Upgrading PyTorch to CUDA version..."
        install_status["progress"] = 10

        ret = _run_uv_install(
            uv_exe, python_exe,
            ["torch==2.5.1", "torchvision==0.20.1"],
            index_url="https://download.pytorch.org/whl/cu121",
            progress_base=10,
            progress_end=50,
            reinstall_packages=["torch", "torchvision"]  # CPU -> CUDA 강제 교체
        )
        if ret != 0:
            raise Exception(f"PyTorch CUDA installation failed with code {ret}")
        
        # diffusers 등 설치 (50% -> 80%)
        install_status["message"] = "Installing diffusers..."
        install_status["progress"] = 50
        
        ret = _run_uv_install(
            uv_exe, python_exe,
            ["diffusers", "transformers", "accelerate", "safetensors", "peft", "spandrel"],
            progress_base=50,
            progress_end=80
        )
        if ret != 0:
            raise Exception(f"diffusers installation failed with code {ret}")

        # ultralytics가 없으면 설치 (80% -> 90%)
        current_status = get_install_status()
        if not current_status.get("censor"):
            install_status["message"] = "Installing ultralytics..."
            install_status["progress"] = 80
            
            ret = _run_uv_install(
                uv_exe, python_exe,
                ["ultralytics", "opencv-python"],
                progress_base=80,
                progress_end=90
            )
            if ret != 0:
                raise Exception(f"ultralytics installation failed with code {ret}")
        
        # backend dependencies (이미 있어도 확인용)
        install_status["message"] = "Verifying dependencies..."
        install_status["progress"] = 90
        
        ret = _run_uv_install(
            uv_exe, python_exe,
            ["fastapi", "uvicorn[standard]", "httpx", "aiofiles", "python-multipart", "pillow", "piexif", "websockets"],
            progress_base=90,
            progress_end=95
        )
        
        # 설치 완료 - 상태 저장
        save_install_status({
            "python": True,
            "torch": "cuda",
            "censor": True,
            "local": True
        })
        
        install_status["progress"] = 100
        install_status["message"] = "Installation complete!"
        install_status["installing"] = False
        
        return True
        
    except Exception as e:
        install_status["error"] = str(e)
        install_status["installing"] = False
        print(f"[Install Error] {e}")
        return False


async def install_local_environment():
    """로컬 생성 환경 설치 - 스레드에서 실행하여 이벤트 루프 블로킹 방지"""
    global install_status
    install_status = {"installing": True, "progress": 0, "message": "Starting...", "error": None}
    
    # 동기 함수를 스레드 풀에서 실행
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _install_local_environment_sync)


@app.get("/api/local/status")
async def get_local_status():
    """로컬 환경 상태 확인"""
    # 설치 중일 때는 installed를 false로 (torch 폴더가 생겨도 아직 완료 아님)
    is_installed = False if install_status["installing"] else is_local_env_installed()
    return {
        "installed": is_installed,
        "installing": install_status["installing"],
        "progress": install_status["progress"],
        "message": install_status["message"],
        "error": install_status["error"],
        "python_env_dir": str(PYTHON_ENV_DIR),
        "censor_ready": True  # ultralytics는 초기 설치에 포함됨
    }


@app.post("/api/local/install")
async def start_local_install(background_tasks: BackgroundTasks):
    """로컬 환경 설치 시작"""
    global install_status

    if install_status["installing"]:
        raise HTTPException(status_code=400, detail="Installation already in progress")

    if is_local_env_installed():
        raise HTTPException(status_code=400, detail="Local environment already installed")

    # torch가 이미 로드되어 있으면 재시작 필요
    # (실행 중 torch 파일 교체 불가 - 파일 잠금)
    torch_dir = _get_site_packages_dir() / "torch"
    if torch_dir.exists():
        # pending 플래그 저장 후 재시작 요청
        status = get_install_status()
        status["pending_cuda_install"] = True
        save_install_status(status)
        return {
            "status": "restart_required",
            "message": "torch가 이미 로드되어 있어 재시작이 필요합니다. 재시작 후 자동으로 CUDA 버전이 설치됩니다."
        }

    # 먼저 상태를 installing으로 설정 (race condition 방지)
    install_status = {"installing": True, "progress": 0, "message": "Starting...", "error": None}

    # 백그라운드에서 설치 실행
    asyncio.create_task(install_local_environment())
    
    return {"status": "started", "message": "Installation started"}


@app.delete("/api/local/uninstall")
async def uninstall_local():
    """로컬 환경 삭제 - torch/diffusers 등 로컬 생성 패키지만 삭제 (embedded python 유지)"""
    if install_status["installing"]:
        raise HTTPException(status_code=400, detail="Installation in progress")

    # 삭제할 패키지 목록 (로컬 생성에 필요한 패키지들)
    packages_to_remove = [
        "torch", "torchvision", "torchaudio",
        "diffusers", "transformers", "accelerate",
        "safetensors", "peft", "huggingface_hub", "spandrel",
        "nvidia_cublas", "nvidia_cuda", "nvidia_cudnn", "nvidia_cufft",
        "nvidia_curand", "nvidia_cusolver", "nvidia_cusparse", "nvidia_nccl", "nvidia_nvtx",
        "triton"
    ]

    # site-packages 경로 찾기
    if platform.system() == "Windows":
        site_packages = PYTHON_ENV_DIR / "Lib" / "site-packages"
    else:
        # macOS/Linux
        site_packages = PYTHON_ENV_DIR / "lib" / f"python{PYTHON_VERSION[:4]}" / "site-packages"

    if not site_packages.exists():
        return {"status": "uninstalled", "message": "이미 제거되었거나 설치된 적 없음"}

    deleted_count = 0
    errors = []

    try:
        # site-packages 내 패키지 폴더/파일 삭제
        for item in site_packages.iterdir():
            item_name = item.name.lower()
            should_delete = False

            # 패키지 이름 매칭
            for pkg in packages_to_remove:
                pkg_lower = pkg.lower()
                if (item_name == pkg_lower or
                    item_name.startswith(pkg_lower + "-") or
                    item_name.startswith(pkg_lower + "_") or
                    (item_name.endswith(".dist-info") and pkg_lower in item_name) or
                    (item_name.endswith(".egg-info") and pkg_lower in item_name)):
                    should_delete = True
                    break

            if should_delete:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    deleted_count += 1
                except Exception as e:
                    errors.append(f"{item.name}: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"삭제 중 오류: {e}")

    # install_status.json 업데이트 (torch/diffusers 제거 반영)
    # 기본 환경(ultralytics 등)은 유지하므로 censor는 True로 유지
    save_install_status({
        "python": True,
        "torch": "cpu",  # CUDA torch 제거됨, 기본 환경 재설치 시 CPU로
        "censor": True,
        "local": False   # diffusers 제거됨
    })

    if errors:
        # 일부만 삭제된 경우
        return {
            "status": "partial",
            "message": f"{deleted_count}개 삭제됨, {len(errors)}개 실패",
            "errors": errors[:5]  # 처음 5개만
        }

    return {"status": "uninstalled", "deleted": deleted_count}


@app.post("/api/restart")
async def restart_server():
    """서버 재시작 (embedded Python으로 전환)"""
    import subprocess
    import sys
    
    if platform.system() == "Windows":
        # Windows: 새 CMD 창에서 bat 파일 실행
        bat_file = APP_DIR / "PeroPix.bat"
        if bat_file.exists():
            # start cmd /c: 새 창에서 실행
            subprocess.Popen(
                f'start cmd /c "{bat_file}"',
                shell=True,
                cwd=str(APP_DIR)
            )
        else:
            # bat 파일이 없으면 직접 Python 실행
            embedded_python = PYTHON_ENV_DIR / "python" / "python.exe"
            python_to_use = str(embedded_python) if embedded_python.exists() else sys.executable
            subprocess.Popen(
                f'start cmd /k "{python_to_use}" backend.py',
                shell=True,
                cwd=str(APP_DIR)
            )
    else:
        # Unix: 새 터미널에서 실행
        command_file = APP_DIR / "PeroPix.command"
        if command_file.exists():
            subprocess.Popen(
                ["open", str(command_file)],
                cwd=str(APP_DIR)
            )
        else:
            embedded_python = PYTHON_ENV_DIR / "python" / "bin" / "python3"
            python_to_use = str(embedded_python) if embedded_python.exists() else sys.executable
            subprocess.Popen(
                [python_to_use, "backend.py"],
                start_new_session=True,
                cwd=str(APP_DIR)
            )
    
    # 현재 프로세스 종료 예약 (응답 반환 후)
    async def delayed_shutdown():
        await asyncio.sleep(0.5)
        os._exit(0)
    
    asyncio.create_task(delayed_shutdown())
    
    return {"status": "restarting"}


# ============================================================
# Auto Censor (YOLO-based)
# ============================================================

# YOLO 모델 (ultralytics는 초기 설치에 포함됨, lazy import)
YOLO = None

def _get_yolo():
    """YOLO 클래스 lazy import"""
    global YOLO
    if YOLO is None:
        from ultralytics import YOLO as _YOLO
        YOLO = _YOLO
    return YOLO

# 검열 모델 캐시
censor_model_cache = {
    "model": None,
    "model_path": None,
    "classes": []
}


def get_censor_model(model_name: str = None):
    """검열 모델 로드 (캐시됨)"""
    # 모델 이름이 없으면 첫 번째 모델 사용
    if not model_name:
        models = list(CENSOR_MODELS_DIR.glob("*.pt"))
        if not models:
            raise RuntimeError(f"검열 모델이 없습니다: {CENSOR_MODELS_DIR}")
        model_name = models[0].name
    
    model_path = CENSOR_MODELS_DIR / model_name
    if not model_path.exists():
        raise RuntimeError(f"모델 파일 없음: {model_path}")
    
    # 캐시된 모델 반환
    if censor_model_cache["model_path"] == str(model_path):
        return censor_model_cache["model"], censor_model_cache["classes"]
    
    # 새 모델 로드
    print(f"[Censor] Loading model: {model_name}")
    model = _get_yolo()(str(model_path))
    classes = list(model.names.values()) if hasattr(model, 'names') else []
    
    censor_model_cache["model"] = model
    censor_model_cache["model_path"] = str(model_path)
    censor_model_cache["classes"] = classes
    
    print(f"[Censor] Model loaded, classes: {classes}")
    return model, classes


def detect_nsfw_regions(image_path: str, model_name: str = None, 
                        target_labels: list = None, label_conf: dict = None,
                        default_conf: float = 0.25, return_all: bool = False):
    """이미지에서 NSFW 영역 감지 (bbox only)"""
    import cv2
    
    model, classes = get_censor_model(model_name)
    
    # 기본값 설정
    if target_labels is None:
        target_labels = classes  # 모든 클래스 감지
    if label_conf is None:
        label_conf = {}
    
    # 최소 confidence로 모델 실행 (return_all이면 매우 낮은 threshold 사용)
    if return_all:
        min_conf = 0.01
    else:
        min_conf = min([label_conf.get(label, default_conf) for label in target_labels] + [default_conf])
    results = model(image_path, conf=min_conf, verbose=False)
    
    detections = []
    for result in results:
        if result.boxes is not None:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                label = result.names[int(cls)]
                
                # 대상 라벨 필터링
                if label not in target_labels:
                    continue
                
                # 라벨별 confidence 체크
                required_conf = label_conf.get(label, default_conf)
                conf_val = float(conf)
                passes_threshold = conf_val >= required_conf
                
                # return_all이 아니면 threshold 미달 건너뛰기
                if not return_all and not passes_threshold:
                    continue
                
                detections.append({
                    "label": label,
                    "confidence": round(conf_val, 3),
                    "box": [int(x) for x in box.tolist()],  # [x1, y1, x2, y2]
                    "passes_threshold": passes_threshold
                })
    
    return detections


def apply_censor_boxes(image_path: str, boxes: list, method: str = "black",
                       color: str = None, output_path: str = None,
                       expand_pixels: int = 0, feather: int = 0):
    """이미지에 검열 박스 적용 (회전, 확장, 그라데이션 지원)

    Args:
        image_path: 이미지 경로
        boxes: 검열 박스 목록
        method: 검열 방식 (black, white, blur, mosaic, color)
        color: 커스텀 색상 (hex)
        output_path: 저장 경로 (None이면 저장 안 함)
        expand_pixels: 박스 확장 픽셀 (0 이상)
        feather: 그라데이션 테두리 픽셀 (0 이상)
    """
    import cv2
    import numpy as np
    import math

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지 로드 실패: {image_path}")

    h, w = image.shape[:2]

    def get_rotated_corners(x1, y1, x2, y2, rotation, expand=0):
        """회전된 사각형의 네 코너 좌표 계산 (확장 포함)"""
        # 확장 적용 (회전 전에 적용)
        if expand > 0:
            x1 -= expand
            y1 -= expand
            x2 += expand
            y2 += expand

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # 로컬 코너 (중심 기준)
        corners_local = [
            (x1 - cx, y1 - cy),  # NW
            (x2 - cx, y1 - cy),  # NE
            (x2 - cx, y2 - cy),  # SE
            (x1 - cx, y2 - cy),  # SW
        ]
        # 회전 적용
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        corners_world = []
        for lx, ly in corners_local:
            wx = cx + lx * cos_r - ly * sin_r
            wy = cy + lx * sin_r + ly * cos_r
            corners_world.append((int(round(wx)), int(round(wy))))
        return corners_world

    def apply_feathered_censor(image, pts, censor_color, feather_px):
        """그라데이션 테두리가 적용된 검열"""
        # 바운딩 박스 계산 (feather 영역 포함)
        bx, by, bw, bh = cv2.boundingRect(pts)
        margin = feather_px + 2
        bx = max(0, bx - margin)
        by = max(0, by - margin)
        bx2 = min(w, bx + bw + margin * 2)
        by2 = min(h, by + bh + margin * 2)

        if bx2 <= bx or by2 <= by:
            return image

        # ROI 추출
        roi = image[by:by2, bx:bx2].copy()
        roi_h, roi_w = roi.shape[:2]

        # 마스크 생성 (ROI 좌표계)
        pts_local = pts - np.array([bx, by])
        core_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillPoly(core_mask, [pts_local], 255, lineType=cv2.LINE_AA)

        # 그라데이션 영역: 마스크 확장
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_px * 2 + 1, feather_px * 2 + 1))
        expanded_mask = cv2.dilate(core_mask, kernel, iterations=1)
        feather_zone = (expanded_mask > 0) & (core_mask == 0)

        # 거리 변환으로 그라데이션 알파 계산
        inv_core = 255 - core_mask
        dist_outside = cv2.distanceTransform(inv_core, cv2.DIST_L2, 5)

        # 알파 맵 생성
        alpha = np.zeros((roi_h, roi_w), dtype=np.float32)
        alpha[core_mask > 0] = 1.0  # 코어 영역은 100%
        # feather zone은 거리에 따라 감쇠
        alpha[feather_zone] = 1.0 - np.clip(dist_outside[feather_zone] / feather_px, 0, 1)

        # 검열 레이어 생성
        censor_layer = np.full_like(roi, censor_color, dtype=np.uint8)

        # 알파 블렌딩
        alpha_3ch = cv2.merge([alpha, alpha, alpha])
        blended = (censor_layer.astype(np.float32) * alpha_3ch +
                   roi.astype(np.float32) * (1 - alpha_3ch)).astype(np.uint8)

        image[by:by2, bx:bx2] = blended
        return image

    def apply_feathered_effect(image, pts, effect_type, feather_px):
        """그라데이션 테두리가 적용된 블러/모자이크"""
        bx, by, bw, bh = cv2.boundingRect(pts)
        margin = feather_px + 2
        bx = max(0, bx - margin)
        by = max(0, by - margin)
        bx2 = min(w, bx + bw + margin * 2)
        by2 = min(h, by + bh + margin * 2)

        if bx2 <= bx or by2 <= by:
            return image

        roi = image[by:by2, bx:bx2].copy()
        roi_h, roi_w = roi.shape[:2]

        # 마스크 생성
        pts_local = pts - np.array([bx, by])
        core_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillPoly(core_mask, [pts_local], 255, lineType=cv2.LINE_AA)

        # 그라데이션 확장
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_px * 2 + 1, feather_px * 2 + 1))
        expanded_mask = cv2.dilate(core_mask, kernel, iterations=1)
        feather_zone = (expanded_mask > 0) & (core_mask == 0)

        # 알파 계산
        inv_core = 255 - core_mask
        dist_outside = cv2.distanceTransform(inv_core, cv2.DIST_L2, 5)
        alpha = np.zeros((roi_h, roi_w), dtype=np.float32)
        alpha[core_mask > 0] = 1.0
        alpha[feather_zone] = 1.0 - np.clip(dist_outside[feather_zone] / feather_px, 0, 1)

        # 효과 적용
        if effect_type == "blur":
            effected = cv2.GaussianBlur(roi, (99, 99), 30)
        else:  # mosaic
            if roi_h > 0 and roi_w > 0:
                small = cv2.resize(roi, (max(1, roi_w // 10), max(1, roi_h // 10)), interpolation=cv2.INTER_LINEAR)
                effected = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            else:
                effected = roi

        # 알파 블렌딩
        alpha_3ch = cv2.merge([alpha, alpha, alpha])
        blended = (effected.astype(np.float32) * alpha_3ch +
                   roi.astype(np.float32) * (1 - alpha_3ch)).astype(np.uint8)

        image[by:by2, bx:bx2] = blended
        return image

    for i, box_info in enumerate(boxes):
        # 박스 데이터 검증
        box = box_info.get("box")
        if box is None:
            print(f"[WARNING] Box {i}: 'box' 키 없음, box_info={box_info}")
            continue
        if not isinstance(box, (list, tuple)):
            print(f"[WARNING] Box {i}: box가 리스트/튜플이 아님, type={type(box)}, value={box}")
            continue
        if len(box) != 4:
            print(f"[WARNING] Box {i}: box 길이가 4가 아님, len={len(box)}, box={box}")
            continue

        try:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        except (ValueError, TypeError, IndexError) as e:
            print(f"[WARNING] Box {i}: 좌표 변환 실패, box={box}, error={e}")
            continue

        box_method = box_info.get("method", method)
        box_color = box_info.get("color", color)
        rotation = box_info.get("rotation", 0)

        # 회전된 코너 계산 (확장 포함)
        corners = get_rotated_corners(x1, y1, x2, y2, rotation, expand_pixels)
        pts = np.array(corners, dtype=np.int32)

        # 검열 색상 결정
        if box_method == "black":
            censor_color = (0, 0, 0)
        elif box_method == "white":
            censor_color = (255, 255, 255)
        elif box_method == "color" and box_color and box_color.startswith("#"):
            hex_color = box_color[1:]
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            censor_color = (b, g, r)
        else:
            censor_color = None

        # 그라데이션 적용 여부
        use_feather = feather > 0

        if box_method in ("black", "white", "color") and censor_color:
            if use_feather:
                image = apply_feathered_censor(image, pts, censor_color, feather)
            else:
                cv2.fillPoly(image, [pts], censor_color, lineType=cv2.LINE_AA)

        elif box_method in ("blur", "mosaic"):
            if use_feather:
                image = apply_feathered_effect(image, pts, box_method, feather)
            else:
                # 기존 로직
                bx, by, bw, bh = cv2.boundingRect(pts)
                bx = max(0, bx)
                by = max(0, by)
                bx2 = min(w, bx + bw)
                by2 = min(h, by + bh)

                if bx2 > bx and by2 > by:
                    roi = image[by:by2, bx:bx2].copy()

                    mask = np.zeros((by2 - by, bx2 - bx), dtype=np.uint8)
                    pts_local = pts - np.array([bx, by])
                    cv2.fillPoly(mask, [pts_local], 255, lineType=cv2.LINE_AA)

                    if box_method == "blur":
                        blurred = cv2.GaussianBlur(roi, (99, 99), 30)
                    else:  # mosaic
                        roi_h, roi_w = roi.shape[:2]
                        if roi_h > 0 and roi_w > 0:
                            small = cv2.resize(roi, (max(1, roi_w // 10), max(1, roi_h // 10)), interpolation=cv2.INTER_LINEAR)
                            blurred = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                        else:
                            blurred = roi

                    alpha = mask.astype(np.float32) / 255.0
                    alpha_3ch = cv2.merge([alpha, alpha, alpha])
                    roi_result = (blurred.astype(np.float32) * alpha_3ch +
                                 roi.astype(np.float32) * (1 - alpha_3ch)).astype(np.uint8)
                    image[by:by2, bx:bx2] = roi_result

    # 저장
    if output_path:
        cv2.imwrite(output_path, image)

    # base64로 반환
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


# === Censor API Endpoints ===

@app.get("/api/censor/folders")
async def list_censor_folders(type: str = "uncensored"):
    """검열 폴더 하위 폴더 목록"""
    base_dir = UNCENSORED_DIR if type == "uncensored" else CENSORED_DIR
    folders = []
    
    if base_dir.exists():
        for item in sorted(base_dir.iterdir()):
            if item.is_dir():
                # 폴더 내 이미지 수 카운트
                image_count = 0
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
                    image_count += len(list(item.glob(ext)))
                folders.append({
                    "name": item.name,
                    "image_count": image_count
                })
    
    return {"success": True, "folders": folders, "type": type}


@app.post("/api/censor/folders")
async def create_censor_folder(request: dict):
    """검열 폴더 생성"""
    folder_type = request.get("type", "uncensored")
    folder_name = request.get("name", "").strip()
    
    if not folder_name:
        return {"success": False, "error": "폴더 이름이 필요합니다"}
    
    # 안전한 폴더명인지 확인
    if "/" in folder_name or "\\" in folder_name or ".." in folder_name:
        return {"success": False, "error": "유효하지 않은 폴더 이름입니다"}
    
    base_dir = UNCENSORED_DIR if folder_type == "uncensored" else CENSORED_DIR
    folder_path = base_dir / folder_name
    
    if folder_path.exists():
        return {"success": False, "error": "폴더가 이미 존재합니다"}
    
    try:
        folder_path.mkdir(parents=True)
        return {"success": True, "name": folder_name}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/censor/models")
async def list_censor_models():
    """사용 가능한 검열 모델 목록"""
    models = []
    if CENSOR_MODELS_DIR.exists():
        for f in sorted(CENSOR_MODELS_DIR.glob("*.pt")):
            models.append(f.name)
    return {"success": True, "models": models, "yolo_available": True}


@app.get("/api/censor/model-info")
async def get_censor_model_info(model: str = None):
    """모델의 클래스 정보 조회"""
    try:
        _, classes = get_censor_model(model)
        return {
            "success": True,
            "model": model or censor_model_cache["model_path"],
            "classes": classes
        }
    except Exception as e:
        import traceback
        print(f"[Censor] Model info error: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


class CensorScanRequest(BaseModel):
    image_path: str = None  # outputs 폴더 기준 상대 경로
    absolute_path: str = None  # 절대 경로 (드롭된 외부 이미지용)
    image_base64: str = None  # 또는 base64 이미지
    model: str = None
    target_labels: List[str] = None
    label_conf: dict = None
    default_conf: float = 0.25
    return_all: bool = False  # True면 threshold 미달도 반환


@app.post("/api/censor/scan")
async def scan_image_for_censor(req: CensorScanRequest):
    """이미지 스캔 (감지만, 검열 미적용)"""
    import tempfile
    temp_file = None
    
    try:
        # 이미지 경로 결정 (우선순위: absolute_path > image_path > image_base64)
        if req.absolute_path:
            # 절대 경로 (외부 이미지)
            image_path = Path(req.absolute_path)
            if not image_path.exists():
                return {"success": False, "error": f"이미지 없음: {req.absolute_path}"}
        elif req.image_path:
            # censoring/uncensored 폴더 기준 상대 경로
            image_path = UNCENSORED_DIR / req.image_path
            if not image_path.exists():
                return {"success": False, "error": f"이미지 없음: {req.image_path}"}
        elif req.image_base64:
            # base64를 임시 파일로 저장
            image_data = base64.b64decode(req.image_base64)
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_file.write(image_data)
            temp_file.close()
            image_path = Path(temp_file.name)
        else:
            return {"success": False, "error": "image_path 또는 image_base64 필요"}
        
        # 감지 실행
        detections = detect_nsfw_regions(
            str(image_path),
            model_name=req.model,
            target_labels=req.target_labels,
            label_conf=req.label_conf,
            default_conf=req.default_conf,
            return_all=req.return_all
        )
        
        return {"success": True, "detections": detections}
    
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if temp_file and Path(temp_file.name).exists():
            Path(temp_file.name).unlink()


class CensorApplyRequest(BaseModel):
    image_path: str = None
    absolute_path: str = None  # 절대 경로 (드롭된 외부 이미지용)
    image_base64: str = None
    boxes: List[dict]  # [{"box": [x1,y1,x2,y2], "method": "black", "color": "#000000"}]
    method: str = "black"  # 기본 검열 방식
    color: str = None  # 커스텀 색상
    expand_pixels: int = 0  # 박스 확장 픽셀
    feather: int = 0  # 그라데이션 테두리 픽셀
    source: str = "uncensored"  # 이미지 소스 폴더 (uncensored 또는 censored)


@app.post("/api/censor/apply")
async def apply_censor(req: CensorApplyRequest):
    """검열 적용 (프리뷰용, 저장 안 함)"""
    import tempfile
    temp_file = None

    try:
        # 이미지 경로 결정 (우선순위: absolute_path > image_path > image_base64)
        if req.absolute_path:
            image_path = Path(req.absolute_path)
            if not image_path.exists():
                return {"success": False, "error": f"이미지 없음: {req.absolute_path}"}
        elif req.image_path:
            # source에 따라 적절한 폴더 사용
            base_dir = CENSORED_DIR if req.source == "censored" else UNCENSORED_DIR
            image_path = base_dir / req.image_path
            if not image_path.exists():
                return {"success": False, "error": f"이미지 없음: {req.image_path}"}
        elif req.image_base64:
            image_data = base64.b64decode(req.image_base64)
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_file.write(image_data)
            temp_file.close()
            image_path = Path(temp_file.name)
        else:
            return {"success": False, "error": "image_path 또는 image_base64 필요"}

        # 검열 적용
        result_base64 = apply_censor_boxes(
            str(image_path),
            req.boxes,
            method=req.method,
            color=req.color,
            expand_pixels=req.expand_pixels,
            feather=req.feather
        )

        return {"success": True, "image": result_base64}
    
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if temp_file and Path(temp_file.name).exists():
            Path(temp_file.name).unlink()


class CensorSaveRequest(BaseModel):
    image_path: str = None
    absolute_path: str = None  # 절대 경로 (드롭된 외부 이미지용)
    image_base64: str = None
    boxes: List[dict]
    method: str = "black"
    color: str = None
    output_folder: str = ""  # censored 하위 폴더
    filename: str = None  # 저장 파일명 (없으면 원본명 사용)
    source: str = "uncensored"  # uncensored 또는 censored (absolute_path 사용 시 무시됨)
    expand_pixels: int = 0  # 박스 확장 픽셀
    feather: int = 0  # 그라데이션 테두리 픽셀


@app.post("/api/censor/save")
async def save_censored_image(req: CensorSaveRequest):
    """검열 이미지 저장"""
    import tempfile
    temp_file = None
    
    # 디버깅: 받은 박스 데이터 로깅
    print(f"[DEBUG] save_censored_image - image_path: {req.image_path}, absolute_path: {req.absolute_path}")
    print(f"[DEBUG] save_censored_image - boxes count: {len(req.boxes)}")
    for i, box in enumerate(req.boxes):
        print(f"[DEBUG] Box {i}: {box}")
    
    try:
        # 이미지 경로 결정 (우선순위: absolute_path > image_path > image_base64)
        if req.absolute_path:
            # 절대 경로 (외부 이미지)
            image_path = Path(req.absolute_path)
            if not image_path.exists():
                return {"success": False, "error": f"이미지 없음: {req.absolute_path}"}
            original_filename = image_path.name
        elif req.image_path:
            source_dir = CENSORED_DIR if req.source == "censored" else UNCENSORED_DIR
            image_path = source_dir / req.image_path
            if not image_path.exists():
                return {"success": False, "error": f"이미지 없음: {req.image_path}"}
            original_filename = Path(req.image_path).name
        elif req.image_base64:
            image_data = base64.b64decode(req.image_base64)
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_file.write(image_data)
            temp_file.close()
            image_path = Path(temp_file.name)
            original_filename = "censored.png"
        else:
            return {"success": False, "error": "image_path, absolute_path 또는 image_base64 필요"}
        
        # 저장 경로 결정 (censoring/censored)
        output_folder = CENSORED_DIR / req.output_folder if req.output_folder else CENSORED_DIR
        output_folder.mkdir(parents=True, exist_ok=True)
        
        filename = req.filename or original_filename
        output_path = output_folder / filename
        
        # 동일 파일명 있으면 덮어쓰기 (넘버링 없음)
        
        # 검열 적용 및 저장
        apply_censor_boxes(
            str(image_path),
            req.boxes,
            method=req.method,
            color=req.color,
            output_path=str(output_path),
            expand_pixels=req.expand_pixels,
            feather=req.feather
        )

        return {"success": True, "filename": output_path.name, "path": str(output_path.relative_to(APP_DIR))}
    
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if temp_file and Path(temp_file.name).exists():
            Path(temp_file.name).unlink()


@app.get("/api/censor/images")
async def list_images_for_censor(folder: str = ""):
    """검열할 이미지 목록 (censoring/uncensored 폴더)"""
    images = []
    target_folder = UNCENSORED_DIR / folder if folder else UNCENSORED_DIR
    
    if not target_folder.exists():
        return {"success": True, "images": [], "folder": folder}
    
    # 이미지 파일 검색
    all_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
        all_files.extend(target_folder.glob(ext))
    
    # 수정 시간순 정렬
    all_files = sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    for filepath in all_files[:200]:  # 최대 200개
        try:
            # 썸네일 생성 (작은 크기 + JPEG로 빠르게)
            img = Image.open(filepath)
            thumb = img.copy()
            thumb.thumbnail((80, 80), Image.BILINEAR)  # 더 작게, 빠른 리샘플링
            if thumb.mode == 'RGBA':
                thumb = thumb.convert('RGB')
            buffer = io.BytesIO()
            thumb.save(buffer, format="JPEG", quality=60)  # JPEG로 압축
            thumb_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            images.append({
                "filename": filepath.name,
                "path": str(filepath.relative_to(UNCENSORED_DIR)),
                "thumbnail": thumb_base64,
                "width": img.width,
                "height": img.height
            })
        except Exception as e:
            print(f"[Censor] Error loading {filepath}: {e}")
            continue
    
    return {"success": True, "images": images, "folder": folder}


@app.get("/api/censor/censored")
async def list_censored_images(folder: str = ""):
    """검열된 이미지 목록"""
    images = []
    target_folder = CENSORED_DIR / folder if folder else CENSORED_DIR
    
    if not target_folder.exists():
        return {"success": True, "images": [], "folder": folder}
    
    # 이미지 파일 검색
    all_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
        all_files.extend(target_folder.glob(ext))
    
    # 수정 시간순 정렬
    all_files = sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    for filepath in all_files[:200]:
        try:
            # 썸네일 생성 (작은 크기 + JPEG로 빠르게)
            img = Image.open(filepath)
            thumb = img.copy()
            thumb.thumbnail((80, 80), Image.BILINEAR)
            if thumb.mode == 'RGBA':
                thumb = thumb.convert('RGB')
            buffer = io.BytesIO()
            thumb.save(buffer, format="JPEG", quality=60)
            thumb_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            images.append({
                "filename": filepath.name,
                "path": str(filepath.relative_to(CENSORED_DIR)),
                "thumbnail": thumb_base64,
                "width": img.width,
                "height": img.height
            })
        except Exception as e:
            continue
    
    return {"success": True, "images": images, "folder": folder}


@app.get("/api/censor/image")
async def get_censor_image(path: str, source: str = "uncensored"):
    """검열용 이미지 조회 (전체 크기) - base64 버전 (하위 호환)"""
    if source == "uncensored" or source == "outputs":
        filepath = UNCENSORED_DIR / path
    elif source == "censored":
        filepath = CENSORED_DIR / path
    else:
        return {"success": False, "error": "Invalid source"}
    
    if not filepath.exists():
        return {"success": False, "error": "Image not found"}
    
    try:
        img = Image.open(filepath)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "image": image_base64,
            "width": img.width,
            "height": img.height,
            "filename": filepath.name
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/censor/file/{source}/{filepath:path}")
async def get_censor_file(source: str, filepath: str):
    """검열용 이미지 파일 직접 서빙 (FileResponse) - Canvas CORS용"""
    if source == "uncensored":
        file_path = UNCENSORED_DIR / filepath
    elif source == "censored":
        file_path = CENSORED_DIR / filepath
    else:
        return {"success": False, "error": "Invalid source"}
    
    if not file_path.exists() or not file_path.is_file():
        return {"success": False, "error": "File not found"}
    
    # 확장자에 따른 미디어 타입 결정
    suffix = file_path.suffix.lower()
    media_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp'
    }
    media_type = media_types.get(suffix, 'image/png')
    
    return FileResponse(
        file_path,
        media_type=media_type,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=3600"  # 1시간 브라우저 캐시
        }
    )


@app.post("/api/censor/batch")
async def batch_censor(request: dict):
    """폴더 내 모든 이미지 일괄 검열"""
    source_folder = request.get("source_folder", "")
    output_folder = request.get("output_folder", "")
    model = request.get("model")
    target_labels = request.get("target_labels")
    label_conf = request.get("label_conf", {})
    default_conf = request.get("default_conf", 0.25)
    method = request.get("method", "black")
    color = request.get("color")
    expand_pixels = request.get("expand_pixels", 0)
    feather = request.get("feather", 0)

    source_path = UNCENSORED_DIR / source_folder if source_folder else UNCENSORED_DIR
    output_path = CENSORED_DIR / output_folder if output_folder else CENSORED_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        return {"success": False, "error": f"폴더 없음: {source_folder}"}

    # 이미지 파일 목록
    all_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
        all_files.extend(source_path.glob(ext))

    results = []
    for filepath in all_files:
        try:
            # 감지
            detections = detect_nsfw_regions(
                str(filepath),
                model_name=model,
                target_labels=target_labels,
                label_conf=label_conf,
                default_conf=default_conf
            )

            # 박스 형식 변환
            boxes = [{"box": d["box"], "method": method, "color": color} for d in detections]

            if boxes:
                # 검열 적용 및 저장
                out_filepath = output_path / filepath.name
                apply_censor_boxes(
                    str(filepath), boxes, method=method, color=color,
                    output_path=str(out_filepath),
                    expand_pixels=expand_pixels, feather=feather
                )
                results.append({
                    "filename": filepath.name,
                    "censored": len(boxes),
                    "saved": True
                })
            else:
                results.append({
                    "filename": filepath.name,
                    "censored": 0,
                    "saved": False
                })
        except Exception as e:
            results.append({
                "filename": filepath.name,
                "error": str(e)
            })

    total = len(results)
    censored = sum(1 for r in results if r.get("censored", 0) > 0)

    return {
        "success": True,
        "total": total,
        "censored": censored,
        "results": results
    }


# ============================================================
# Image Set Save (Slot Mode)
# ============================================================

def select_folder_dialog(title: str = "폴더 선택") -> Optional[str]:
    """시스템 폴더 선택 다이얼로그 (Windows/macOS/Linux) - tkinter 사용"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        # Windows에서 다이얼로그가 뒤로 가는 문제 방지
        root.attributes('-topmost', True)
        root.update()
        
        folder_path = filedialog.askdirectory(
            title=title,
            parent=root
        )
        
        root.destroy()
        return folder_path if folder_path else None
    except Exception as e:
        print(f"[FolderDialog] tkinter error: {e}")
        return None


class SaveImageSetRequest(BaseModel):
    image_paths: List[str]  # outputs 폴더 기준 상대 경로 목록


@app.post("/api/save-image-set")
async def save_image_set(req: SaveImageSetRequest):
    """현재 슬롯에 표시된 이미지들을 선택한 폴더로 복사"""
    if not req.image_paths:
        return {"success": False, "error": "저장할 이미지가 없습니다"}
    
    # 폴더 선택 다이얼로그
    target_folder = select_folder_dialog("이미지 세트 저장 폴더 선택")
    
    if not target_folder:
        return {"success": False, "cancelled": True, "error": "폴더 선택이 취소되었습니다"}
    
    target_path = Path(target_folder)
    if not target_path.exists():
        return {"success": False, "error": f"폴더가 존재하지 않습니다: {target_folder}"}
    
    saved_count = 0
    errors = []
    
    for image_path in req.image_paths:
        try:
            # outputs 폴더 기준 상대 경로
            source = OUTPUT_DIR / image_path
            if not source.exists():
                errors.append(f"파일 없음: {image_path}")
                continue
            
            # 대상 경로
            dest = target_path / source.name
            
            # 파일명 충돌 시 처리
            if dest.exists():
                stem = source.stem
                suffix = source.suffix
                counter = 1
                while dest.exists():
                    dest = target_path / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            # 파일 복사 (메타데이터 포함)
            shutil.copy2(str(source), str(dest))
            saved_count += 1
            
        except Exception as e:
            errors.append(f"{image_path}: {str(e)}")
    
    return {
        "success": saved_count > 0,
        "saved_count": saved_count,
        "total_count": len(req.image_paths),
        "target_folder": target_folder,
        "errors": errors if errors else None
    }


def run_first_install_if_needed():
    """첫 실행 시 기본 환경 설치 (CMD에서 동기식으로 진행)"""
    status = get_install_status()
    
    if status["python"]:
        # 이미 설치됨
        print(f"[Environment] Ready: torch={status['torch']}, censor={status['censor']}, local={status['local']}")
        return True
    
    # 첫 실행 - 설치 시작
    print()
    print("=" * 60)
    print("  First run detected! Installing base environment...")
    print("  This will install Python + PyTorch (CPU) + Ultralytics")
    print("  Please wait... (약 3-5분 소요)")
    print("=" * 60)
    print()
    
    global install_status
    install_status = {"installing": True, "progress": 0, "message": "Starting...", "error": None}
    
    # 동기식으로 설치 진행 (CMD에서 진행 상황 출력됨)
    success = _install_base_environment_sync()
    
    if success:
        print()
        print("=" * 60)
        print("  Installation complete! Starting PeroPix...")
        print("=" * 60)
        print()
        return True
    else:
        print()
        print("=" * 60)
        print(f"  Installation failed: {install_status.get('error', 'Unknown error')}")
        print("  Please check your internet connection and try again.")
        print("=" * 60)
        print()
        return False


if __name__ == "__main__":
    import uvicorn

    # Startup Banner
    banner = """
    ╔═════════════════════════════════════════════════╗
    ║      PeroPix - Slot Base Image Generator        ║
    ║            http://127.0.0.1:8765                ║
    ╚═════════════════════════════════════════════════╝
    """
    print(banner)

    # 첫 실행 감지 및 동기 설치 (서버 시작 전)
    status = get_install_status()
    # Python은 있지만 torch/ultralytics가 없으면 설치 필요 (배포판 첫 실행)
    needs_install = not status["python"] or not status["torch"] or not status["censor"]

    if needs_install:
        print("\n" + "=" * 50)
        print("  기본 환경 설치를 시작합니다")
        print("  (torch CPU + ultralytics, 약 3-5분 소요)")
        print("=" * 50 + "\n")

        success = _install_base_environment_sync()

        if success:
            print("\n" + "=" * 50)
            print("  ✅ 기본 환경 설치 완료!")
            print("  서버를 시작합니다...")
            print("=" * 50 + "\n")
            # 설치 후 status 갱신
            status = get_install_status()
        else:
            print("\n" + "=" * 50)
            print("  ❌ 설치 실패. 로그를 확인해주세요.")
            print("=" * 50 + "\n")
            # 실패해도 서버는 시작 (NAI 모드는 사용 가능)
    elif status.get("pending_cuda_install"):
        # 재시작 후 CUDA torch 설치 (서버 시작 전, torch 로드 전)
        print("\n" + "=" * 50)
        print("  🔥 CUDA PyTorch 설치를 시작합니다")
        print("  (torch CUDA + diffusers, 약 5-10분 소요)")
        print("=" * 50 + "\n")

        success = _install_local_environment_sync()

        if success:
            print("\n" + "=" * 50)
            print("  ✅ CUDA 환경 설치 완료!")
            print("=" * 50 + "\n")
            # pending 플래그 제거
            status = get_install_status()
            status.pop("pending_cuda_install", None)
            save_install_status(status)
        else:
            print("\n" + "=" * 50)
            print("  ❌ CUDA 설치 실패. 로그를 확인해주세요.")
            print("=" * 50 + "\n")
            # pending 플래그 제거 (무한 루프 방지)
            status = get_install_status()
            status.pop("pending_cuda_install", None)
            save_install_status(status)
        status = get_install_status()
    else:
        print(f"[Environment] torch={status['torch']}, censor={status['censor']}, local={status['local']}")

    # YOLO 모델 미리 로드 (서버 시작 전, API 블로킹 방지)
    if status.get("censor"):
        try:
            print("[Preload] Loading YOLO model...")
            get_censor_model()
            print("[Preload] YOLO model ready")
        except Exception as e:
            print(f"[Preload] YOLO model failed: {e}")

    # Windows: 폴더 열기 foreground 권한 획득을 위한 더미 호출
    import platform
    if platform.system() == "Windows":
        try:
            import win32com.client
            win32com.client.Dispatch("Shell.Application")
        except:
            pass

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
