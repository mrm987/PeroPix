"""
PeroPix Local Engine - ComfyUI 기반 로컬 이미지 생성 엔진

Diffusers를 완전히 대체하여 ComfyUI와 동일한 결과물을 생성합니다.
"""

from .generate import SDXLGenerator, create_generator, generate
from .model_sampling import ModelSamplingDiscrete, EPS
from .samplers import sample_euler, sample_euler_ancestral, SAMPLERS
from .schedulers import (
    normal_scheduler,
    simple_scheduler,
    sgm_uniform,
    get_sigmas_karras,
    get_sigmas,
    SCHEDULERS,
)
from .model_loader import (
    load_sdxl_model,
    get_model_cache,
    SDXLModelLoader,
    ModelCache,
)
from .lora import LoRALoader, apply_lora

__all__ = [
    # 생성기
    "SDXLGenerator",
    "create_generator",
    "generate",
    # 모델 샘플링
    "ModelSamplingDiscrete",
    "EPS",
    # 샘플러
    "sample_euler",
    "sample_euler_ancestral",
    "SAMPLERS",
    # 스케줄러
    "normal_scheduler",
    "simple_scheduler",
    "sgm_uniform",
    "get_sigmas_karras",
    "get_sigmas",
    "SCHEDULERS",
    # 모델 로더
    "load_sdxl_model",
    "get_model_cache",
    "SDXLModelLoader",
    "ModelCache",
    # LoRA
    "LoRALoader",
    "apply_lora",
]
