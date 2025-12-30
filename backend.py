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
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

# 경로 설정 (앱 디렉토리 기준)
if getattr(sys, 'frozen', False):
    # PyInstaller로 빌드된 경우
    APP_DIR = Path(sys.executable).parent
else:
    APP_DIR = Path(__file__).parent

MODELS_DIR = APP_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LORA_DIR = MODELS_DIR / "loras"
UPSCALE_DIR = MODELS_DIR / "upscale_models"
OUTPUT_DIR = APP_DIR / "outputs"
PRESETS_DIR = APP_DIR / "presets"
PROMPTS_DIR = APP_DIR / "prompts"
CONFIG_FILE = APP_DIR / "config.json"
PYTHON_ENV_DIR = APP_DIR / "python_env"  # 로컬 생성용 Python 환경

# 카테고리별 이미지 순번 - 매번 실제 폴더 스캔
def get_next_image_number(category: str, save_dir: Path = None) -> int:
    """해당 카테고리의 다음 순번 반환 - 실제 폴더에서 스캔"""
    if save_dir is None:
        save_dir = OUTPUT_DIR

    max_num = 0
    pattern = re.compile(rf'^{re.escape(category)}_(\d{{7}})\.png$')

    if save_dir.exists():
        for f in save_dir.iterdir():
            try:
                if f.suffix == '.png':
                    match = pattern.match(f.name)
                    if match:
                        num = int(match.group(1))
                        if num > max_num:
                            max_num = num
            except Exception:
                pass

    return max_num + 1

# 디렉토리 생성
for d in [CHECKPOINTS_DIR, LORA_DIR, UPSCALE_DIR, OUTPUT_DIR, PRESETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Prompts 하위 폴더 생성
for subdir in ['base', 'negative', 'character']:
    (PROMPTS_DIR / subdir).mkdir(parents=True, exist_ok=True)

# Config 관리
def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {"nai_token": "", "checkpoints_dir": str(CHECKPOINTS_DIR), "lora_dir": str(LORA_DIR)}

def save_config(config):
    CONFIG_FILE.write_text(json.dumps(config, indent=2))

CONFIG = load_config()

# ============================================================
# Imports (lazy loading for faster startup)
# ============================================================
torch = None
np = None
Image = None

def lazy_imports():
    global torch, np, Image
    if torch is None:
        import torch as _torch
        import numpy as _np
        from PIL import Image as _Image
        torch = _torch
        np = _np
        Image = _Image

# ============================================================
# Model Cache
# ============================================================
class ModelCache:
    def __init__(self):
        self.pipe = None
        self.model_path = None
        self.loaded_loras = {}
        self._device = None
        self._dtype = None
    
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
            self._dtype = torch.float16 if self.device == "cuda" else torch.float32
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
        
        print(f"[Load] Loading LoRA: {lora_name} (scale={scale})")
        self.pipe.load_lora_weights(str(lora_path), adapter_name=lora_name)
        self.loaded_loras[lora_name] = scale
    
    def set_lora_scales(self, lora_configs: List[dict]):
        if not lora_configs:
            if self.loaded_loras:
                self.pipe.unload_lora_weights()
                self.loaded_loras = {}
            return
        
        adapter_names = []
        adapter_weights = []
        
        for config in lora_configs:
            name = config["name"]
            scale = config.get("scale", 1.0)
            
            if name not in self.loaded_loras:
                self.load_lora(name, scale)
            
            adapter_names.append(name)
            adapter_weights.append(scale)
        
        if adapter_names:
            self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
    
    def clear(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self.model_path = None
        self.loaded_loras = {}
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
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

    # NAI Vibe Transfer (최대 16개)
    vibe_transfer: List[dict] = []  # [{"image": base64, "info_extracted": 1.0, "strength": 0.6}, ...]

    # NAI Character Reference (V4.5 only)
    character_reference: Optional[dict] = None  # {"image": base64, "fidelity": 0.5, "style_aware": True}
    
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
    model: str = ""
    loras: List[dict] = []
    output_folder: str = ""  # 비어있으면 outputs에 직접 저장, 있으면 outputs/폴더명에 저장
    
    # Upscale (Local only)
    enable_upscale: bool = False
    upscale_model: str = ""
    downscale_ratio: float = 0.7
    upscale_steps: int = 15
    upscale_cfg: float = 5.0
    upscale_denoise: float = 0.5
    size_alignment: str = "none"


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

async def call_nai_api(req: GenerateRequest):
    lazy_imports()
    import httpx
    
    token = CONFIG.get("nai_token", "")
    if not token:
        raise HTTPException(status_code=500, detail="NAI token not set. Go to Settings.")
    
    uc_preset_map = {"Heavy": 0, "Light": 1, "Human Focus": 2, "None": 3}
    uc_preset_value = uc_preset_map.get(req.uc_preset, 0)
    
    sm = req.smea in ["SMEA", "SMEA+DYN"]
    sm_dyn = req.smea == "SMEA+DYN"
    
    seed = req.seed if req.seed >= 0 else np.random.randint(0, 2**31 - 1)
    
    # NAI 값이면 그대로, KSampler 값이면 변환
    nai_sampler = NAI_SAMPLER_MAP.get(req.sampler, req.sampler)
    nai_scheduler = NAI_SCHEDULER_MAP.get(req.scheduler, req.scheduler)
    
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
        "cfg_rescale": 0.0,
        "noise_schedule": nai_scheduler,
        "legacy_v3_extend": False,
        "uncond_scale": 1.0,
        "negative_prompt": req.negative_prompt,
        "prompt": req.prompt,
        "reference_image_multiple": [v["image"] for v in req.vibe_transfer] if req.vibe_transfer else [],
        "reference_information_extracted_multiple": [v.get("info_extracted", 1.0) for v in req.vibe_transfer] if req.vibe_transfer else [],
        "reference_strength_multiple": [v.get("strength", 0.6) for v in req.vibe_transfer] if req.vibe_transfer else [],
        "extra_noise_seed": int(seed),
        "use_coords": False,
        "characterPrompts": [{"prompt": cp, "uc": "", "center": {"x": 0.5, "y": 0.5}, "enabled": True} for cp in req.character_prompts] if req.character_prompts else [],
        "v4_prompt": {
            "use_coords": False,
            "use_order": True,
            "caption": {
                "base_caption": req.prompt, 
                "char_captions": [{"char_caption": cp, "centers": [{"x": 0.5, "y": 0.5}]} for cp in req.character_prompts] if req.character_prompts else []
            }
        },
        "v4_negative_prompt": {
            "legacy_uc": False,
            "caption": {
                "base_caption": req.negative_prompt,
                "char_captions": [{"char_caption": "", "centers": [{"x": 0.5, "y": 0.5}]} for _ in req.character_prompts] if req.character_prompts else []
            }
        }
    }

    # Character Reference (V4.5 only)
    if req.character_reference and req.character_reference.get("image"):
        fidelity = req.character_reference.get("fidelity", 0.5)
        style_aware = req.character_reference.get("style_aware", True)
        caption_type = "character&style" if style_aware else "character"

        params["director_reference_images"] = [req.character_reference["image"]]
        params["director_reference_descriptions"] = [{
            "use_coords": False,
            "use_order": False,
            "legacy_uc": False,
            "caption": {
                "base_caption": caption_type,
                "char_captions": []
            }
        }]
        params["director_reference_strength_values"] = [1.0]
        params["director_reference_secondary_strength_values"] = [1.0 - fidelity]
        params["director_reference_information_extracted"] = [1.0]

    payload = {
        "input": req.prompt,
        "model": req.nai_model,
        "action": "generate",
        "parameters": params
    }
    
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
            raise HTTPException(
                status_code=response.status_code,
                detail=f"NAI API error: {response.text}"
            )
        
        import zipfile
        zip_buffer = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_buffer, 'r') as zf:
            image_name = zf.namelist()[0]
            image_data = zf.read(image_name)
            return Image.open(io.BytesIO(image_data)), int(seed)


# ============================================================
# Local Diffusers
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
    scheduler_name = scheduler_map.get(req.sampler, "EulerAncestralDiscreteScheduler")
    scheduler_class = getattr(schedulers, scheduler_name, None)
    if scheduler_class:
        try:
            pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
        except Exception as e:
            print(f"[Warning] Failed to set scheduler {scheduler_name}: {e}")
    
    print(f"[Generate] prompt={req.prompt[:50]}..., size={req.width}x{req.height}, steps={req.steps}")
    
    # 1st pass - 기본 생성
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
    print("=" * 50)
    print("NAI + Local Generator Backend")
    print(f"App dir: {APP_DIR}")
    print(f"Checkpoints: {CONFIG.get('checkpoints_dir', CHECKPOINTS_DIR)}")
    print(f"NAI Token: {'Set' if CONFIG.get('nai_token') else 'Not set'}")
    print("=" * 50)
    
    # 큐 처리 백그라운드 태스크 시작
    queue_task = asyncio.create_task(process_queue())
    
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

# 큐 시스템
from collections import deque
import uuid

class GenerationQueue:
    def __init__(self):
        self.queue = deque()  # 대기 큐
        self.current_job = None  # 현재 실행 중인 작업
        self.current_job_id = None
        self.cancel_current = False  # 현재 작업 취소 플래그
        self.is_processing = False
        self.clients = []  # SSE 클라이언트들
    
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
            "queued_jobs": [{"id": j["id"], "prompts": len(j["request"].prompt_list or [''])} for j in self.queue]
        }
    
    async def broadcast(self, data):
        """모든 클라이언트에 메시지 전송"""
        for client in self.clients[:]:
            try:
                await client.put(data)
            except:
                self.clients.remove(client)

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
                
                await process_job(job)
                
                gen_queue.is_processing = False
                gen_queue.current_job = None
                gen_queue.current_job_id = None
                gen_queue.cancel_current = False
        
        await asyncio.sleep(0.1)


async def process_job(job):
    """단일 작업 처리"""
    lazy_imports()
    req = job["request"]
    job_id = job["id"]
    
    # PromptItem 리스트 처리
    prompts = req.prompt_list if req.prompt_list else [PromptItem(name="", content="")]
    total_images = len(prompts)
    image_idx = 0
    
    # 시드 설정
    current_seed = req.seed if req.seed >= 0 else np.random.randint(0, 2**31 - 1)
    
    # 시작 알림
    await gen_queue.broadcast({
        "type": "job_start",
        "job_id": job_id,
        "total": total_images,
        "queue_length": len(gen_queue.queue)
    })
    
    for prompt_idx, prompt_item in enumerate(prompts):
        # 취소 체크
        if gen_queue.cancel_current:
            cancelled_images = total_images - image_idx
            await gen_queue.broadcast({
                "type": "job_cancelled",
                "job_id": job_id,
                "index": image_idx,
                "total": total_images,
                "cancelled_images": cancelled_images
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
            current_seed = np.random.randint(0, 2**31 - 1)
        
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
            model=req.model,
            loras=req.loras,
            # Upscale params
            enable_upscale=req.enable_upscale,
            upscale_model=req.upscale_model,
            downscale_ratio=req.downscale_ratio,
            upscale_steps=req.upscale_steps,
            upscale_cfg=req.upscale_cfg,
            upscale_denoise=req.upscale_denoise,
            size_alignment=req.size_alignment,
        )
        
        try:
            if req.provider == "nai":
                image, actual_seed = await call_nai_api(single_req)
            else:
                image, actual_seed = call_local_diffusers(single_req)
            
            # 파일명용 태그 결정: name > 첫 태그
            file_tag = ""
            if prompt_name:
                file_tag = "".join(c for c in prompt_name if c.isalnum() or c in "_-")[:20]
            elif extra_prompt:
                tag = extra_prompt.split(",")[0].split()[0].strip()
                file_tag = "".join(c for c in tag if c.isalnum() or c in "_-")[:20]
            
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

            file_num = get_next_image_number(category, save_dir)
            filename = f"{category}_{file_num:07d}.png"
            image.save(save_dir / filename)
            
            await gen_queue.broadcast({
                "type": "image",
                "job_id": job_id,
                "index": image_idx - 1,
                "total": total_images,
                "prompt": extra_prompt,
                "prompt_idx": slot_index,
                "seed": actual_seed,
                "image": image_to_base64(image),
                "filename": filename,
                "queue_length": len(gen_queue.queue)
            })
            
        except Exception as e:
            import traceback
            print(f"[Error] Generation failed: {e}")
            traceback.print_exc()
            image_idx += 1
            await gen_queue.broadcast({
                "type": "error",
                "job_id": job_id,
                "index": image_idx - 1,
                "total": total_images,
                "error": str(e)
            })
    
    await gen_queue.broadcast({
        "type": "job_done",
        "job_id": job_id,
        "queue_length": len(gen_queue.queue)
    })



def image_to_base64(img) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config")
async def get_config():
    return {
        "nai_token_set": bool(CONFIG.get("nai_token")),
        "checkpoints_dir": CONFIG.get("checkpoints_dir", str(CHECKPOINTS_DIR)),
        "lora_dir": CONFIG.get("lora_dir", str(LORA_DIR)),
    }


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    global CONFIG
    if update.nai_token is not None:
        CONFIG["nai_token"] = update.nai_token
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
                         vibe_count: int = 0, has_char_ref: bool = False) -> int:
    """NAI 이미지 생성 Anlas 소모량 계산"""
    # Opus 구독자는 일정 조건에서 무료
    # 기본 해상도(1024x1024 이하)에서 28 steps 이하면 무료

    pixels = width * height
    base_pixels = 1024 * 1024  # 기준 해상도

    # Opus 무료 조건 체크
    if is_opus and pixels <= base_pixels and steps <= 28 and vibe_count <= 0 and not has_char_ref:
        return 0

    # 기본 비용 계산 (해상도 기반)
    if pixels <= 640 * 640:
        base_cost = 4
    elif pixels <= 832 * 1216:  # Portrait/Landscape
        base_cost = 8
    elif pixels <= 1024 * 1024:
        base_cost = 10
    elif pixels <= 1216 * 832:
        base_cost = 8
    elif pixels <= 1024 * 1536:
        base_cost = 16
    elif pixels <= 1536 * 1024:
        base_cost = 16
    else:
        # 큰 해상도
        base_cost = int(pixels / base_pixels * 20)

    # Steps 보정 (28 초과시)
    if steps > 28:
        base_cost = int(base_cost * (steps / 28))

    # Vibe Transfer 추가 비용 (4개 초과시 개당 2 Anlas)
    if vibe_count > 4:
        base_cost += (vibe_count - 4) * 2

    return base_cost


@app.post("/api/nai/calculate-cost")
async def calculate_cost(request: dict):
    """Anlas 소모량 계산"""
    width = request.get("width", 832)
    height = request.get("height", 1216)
    steps = request.get("steps", 28)
    is_opus = request.get("is_opus", False)
    vibe_count = request.get("vibe_count", 0)
    has_char_ref = request.get("has_char_ref", False)
    count = request.get("count", 1)  # 생성 횟수

    cost_per_image = calculate_anlas_cost(width, height, steps, is_opus, vibe_count, has_char_ref)
    total_cost = cost_per_image * count

    return {
        "cost_per_image": cost_per_image,
        "total_cost": total_cost,
        "count": count,
        "is_free": cost_per_image == 0
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
            image, seed = call_local_diffusers(req)
        
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
    return {
        "success": True,
        "job_id": job_id,
        "queue_length": len(gen_queue.queue),
        "message": f"Job {job_id} added to queue"
    }


@app.get("/api/stream")
async def stream():
    """SSE 스트림 - 모든 이벤트 수신"""
    async def event_stream():
        queue = asyncio.Queue()
        gen_queue.clients.append(queue)
        
        try:
            # 초기 상태 전송
            yield f"data: {json.dumps({'type': 'status', **gen_queue.get_status()})}\n\n"
            
            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in gen_queue.clients:
                gen_queue.clients.remove(queue)
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    )


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
    await gen_queue.broadcast({
        "type": "queue_cleared",
        "cleared_jobs": cleared_jobs,
        "cleared_images": cleared_images,
        "queue_length": 0
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

# 설치 상태 추적
install_status = {
    "installing": False,
    "progress": 0,
    "message": "",
    "error": None
}

# Python 버전 및 URL (python-build-standalone)
PYTHON_VERSION = "3.12.8"
PYTHON_BUILD_DATE = "20241219"

def get_python_download_url():
    """플랫폼에 맞는 Python standalone URL 반환"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        if machine in ["amd64", "x86_64"]:
            arch = "x86_64"
        else:
            arch = "i686"
        filename = f"cpython-{PYTHON_VERSION}+{PYTHON_BUILD_DATE}-{arch}-pc-windows-msvc-install_only_stripped.tar.gz"
    elif system == "darwin":
        if machine == "arm64":
            arch = "aarch64"
        else:
            arch = "x86_64"
        filename = f"cpython-{PYTHON_VERSION}+{PYTHON_BUILD_DATE}-{arch}-apple-darwin-install_only_stripped.tar.gz"
    else:  # linux
        if machine == "aarch64":
            arch = "aarch64"
        else:
            arch = "x86_64"
        filename = f"cpython-{PYTHON_VERSION}+{PYTHON_BUILD_DATE}-{arch}-unknown-linux-gnu-install_only_stripped.tar.gz"
    
    return f"https://github.com/astral-sh/python-build-standalone/releases/download/{PYTHON_BUILD_DATE}/{filename}"


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
    """로컬 환경 설치 여부 확인"""
    python_exe = PYTHON_ENV_DIR / "python" / "python.exe" if platform.system() == "Windows" else PYTHON_ENV_DIR / "python" / "bin" / "python3"
    torch_check = PYTHON_ENV_DIR / "python" / "Lib" / "site-packages" / "torch" if platform.system() == "Windows" else PYTHON_ENV_DIR / "python" / "lib" / f"python{PYTHON_VERSION[:4]}" / "site-packages" / "torch"
    
    return python_exe.exists() and torch_check.exists()


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
    elif archive_path.name.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tf:
            tf.extractall(dest_dir)
    elif archive_path.suffix == '.tar':
        with tarfile.open(archive_path, 'r') as tf:
            tf.extractall(dest_dir)


async def install_local_environment():
    """로컬 생성 환경 설치 - PyTorch + diffusers"""
    global install_status
    
    install_status = {"installing": True, "progress": 0, "message": "Starting...", "error": None}
    
    try:
        PYTHON_ENV_DIR.mkdir(parents=True, exist_ok=True)
        temp_dir = PYTHON_ENV_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Python과 uv 경로
        if platform.system() == "Windows":
            python_exe = PYTHON_ENV_DIR / "python" / "python.exe"
            uv_exe = PYTHON_ENV_DIR / "uv.exe"
        else:
            python_exe = PYTHON_ENV_DIR / "python" / "bin" / "python3"
            uv_exe = PYTHON_ENV_DIR / "uv"
        
        # Python이 이미 있으면 다운로드 스킵 (배포판에 포함된 경우)
        if not python_exe.exists():
            # 1. Python standalone 다운로드 (10%)
            install_status["message"] = "Downloading Python..."
            install_status["progress"] = 5
            
            python_url = get_python_download_url()
            python_archive = temp_dir / "python.tar.gz"
            
            def python_progress(downloaded, total):
                install_status["progress"] = 5 + int((downloaded / total) * 15)
            
            if not download_file(python_url, python_archive, python_progress):
                raise Exception("Failed to download Python")
            
            # 2. Python 압축 해제 (25%)
            install_status["message"] = "Extracting Python..."
            install_status["progress"] = 20
            
            extract_archive(python_archive, PYTHON_ENV_DIR)
        
        install_status["progress"] = 25
        
        # uv가 이미 있으면 다운로드 스킵
        if not uv_exe.exists():
            # 3. uv 다운로드 (35%)
            install_status["message"] = "Downloading uv..."
            install_status["progress"] = 25
            
            uv_url = get_uv_download_url()
            uv_archive = temp_dir / ("uv.zip" if platform.system() == "Windows" else "uv.tar.gz")
            
            def uv_progress(downloaded, total):
                install_status["progress"] = 25 + int((downloaded / total) * 10)
            
            if not download_file(uv_url, uv_archive, uv_progress):
                raise Exception("Failed to download uv")
            
            # 4. uv 압축 해제 (40%)
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
        
        install_status["progress"] = 40
        
        # 5. PyTorch + diffusers 설치 (40% -> 95%)
        install_status["message"] = "Installing PyTorch (this may take a few minutes)..."
        
        env = os.environ.copy()
        env["UV_PYTHON"] = str(python_exe)
        
        def run_uv_install(packages, index_url=None, progress_base=40, progress_end=70):
            """uv로 패키지 설치"""
            cmd = [str(uv_exe), "pip", "install", "--python", str(python_exe)]
            if index_url:
                cmd.extend(["--index-url", index_url])
            cmd.extend(packages)
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
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
        
        # Step 1: PyTorch with CUDA (40% -> 70%)
        install_status["message"] = "Installing PyTorch with CUDA..."
        install_status["progress"] = 40
        
        ret = run_uv_install(
            ["torch", "torchvision"],
            index_url="https://download.pytorch.org/whl/cu121",
            progress_base=40,
            progress_end=65
        )
        if ret != 0:
            raise Exception(f"PyTorch installation failed with code {ret}")
        
        # Step 2: diffusers and others (65% -> 85%)
        install_status["message"] = "Installing diffusers..."
        install_status["progress"] = 65
        
        ret = run_uv_install(
            ["diffusers", "transformers", "accelerate", "safetensors", "peft", "spandrel"],
            index_url=None,  # 기본 PyPI
            progress_base=65,
            progress_end=85
        )
        if ret != 0:
            raise Exception(f"diffusers installation failed with code {ret}")
        
        # Step 3: FastAPI and backend dependencies (85% -> 95%)
        install_status["message"] = "Installing backend dependencies..."
        install_status["progress"] = 85
        
        ret = run_uv_install(
            ["fastapi", "uvicorn", "httpx", "aiofiles", "python-multipart"],
            index_url=None,
            progress_base=85,
            progress_end=95
        )
        if ret != 0:
            raise Exception(f"Backend dependencies installation failed with code {ret}")
        
        install_status["progress"] = 95
        
        # 6. 정리 (100%)
        install_status["message"] = "Cleaning up..."
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        install_status["progress"] = 100
        install_status["message"] = "Installation complete!"
        install_status["installing"] = False
        
        return True
        
    except Exception as e:
        install_status["error"] = str(e)
        install_status["installing"] = False
        print(f"[Install Error] {e}")
        return False


@app.get("/api/local/status")
async def get_local_status():
    """로컬 환경 상태 확인"""
    return {
        "installed": is_local_env_installed(),
        "installing": install_status["installing"],
        "progress": install_status["progress"],
        "message": install_status["message"],
        "error": install_status["error"],
        "python_env_dir": str(PYTHON_ENV_DIR)
    }


@app.post("/api/local/install")
async def start_local_install(background_tasks: BackgroundTasks):
    """로컬 환경 설치 시작"""
    if install_status["installing"]:
        raise HTTPException(status_code=400, detail="Installation already in progress")
    
    if is_local_env_installed():
        raise HTTPException(status_code=400, detail="Local environment already installed")
    
    # 백그라운드에서 설치 실행
    asyncio.create_task(install_local_environment())
    
    return {"status": "started", "message": "Installation started"}


@app.delete("/api/local/uninstall")
async def uninstall_local():
    """로컬 환경 삭제"""
    if install_status["installing"]:
        raise HTTPException(status_code=400, detail="Installation in progress")
    
    if PYTHON_ENV_DIR.exists():
        shutil.rmtree(PYTHON_ENV_DIR, ignore_errors=True)
    
    return {"status": "uninstalled"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
