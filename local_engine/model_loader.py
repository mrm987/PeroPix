"""
모델 로더

safetensors 체크포인트에서 UNet, VAE, CLIP을 로드하고
state_dict를 적절한 형식으로 분리
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from safetensors.torch import load_file

from .ldm.models.autoencoder import AutoencoderKL, SDXL_VAE_CONFIG
from .ldm.modules.diffusionmodules.openaimodel import UNetModel, SDXL_UNET_CONFIG
from .ldm.modules.attention import XFORMERS_AVAILABLE
from .clip.sdxl_clip import SDXLClipModel


# PyTorch 백엔드 최적화 설정 (ComfyUI 방식)
def _configure_pytorch_backends():
    """PyTorch CUDA 백엔드 최적화 설정"""
    if not torch.cuda.is_available():
        return

    # 결정론적 알고리즘 사용 (재현 가능한 결과를 위해)
    # 주의: 약간의 성능 저하가 있을 수 있음
    import os
    if os.environ.get("PEROPIX_DETERMINISTIC", "0") == "1":
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info("Deterministic mode enabled (PEROPIX_DETERMINISTIC=1)")
    else:
        # cuDNN 자동 튜닝 활성화 (동일 입력 크기에서 최적 알고리즘 캐싱)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            logging.info("cuDNN benchmark mode enabled")

    # SDPA (Scaled Dot Product Attention) 백엔드 활성화
    try:
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logging.info("SDPA backends enabled: math, flash, mem_efficient")
    except AttributeError:
        # PyTorch 버전이 낮으면 해당 함수가 없을 수 있음
        pass

    # Float32 matmul precision - 'highest'로 설정 (ComfyUI 기본값)
    # 'medium'/'high'는 TF32 사용하여 정밀도 손실 발생 가능
    try:
        torch.set_float32_matmul_precision('highest')
        logging.info("Float32 matmul precision: highest (quality priority)")
    except Exception as e:
        logging.warning(f"Could not set float32 matmul precision: {e}")

    # FP16 matmul accumulation - PyTorch 2.1+에서 지원
    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_fp16_accumulation'):
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
        logging.info("FP16 matmul accumulation: disabled")

    # FP16/BF16 reduction for SDPA - PyTorch 2.5+에서 지원
    # ComfyUI는 이것을 True로 설정하지만, 품질 우선이면 False가 나을 수 있음
    if hasattr(torch.backends.cuda, 'allow_fp16_bf16_reduction_math_sdp'):
        try:
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
            logging.info("FP16/BF16 SDPA reduction: disabled (quality priority)")
        except Exception:
            pass


# 모듈 로드 시 백엔드 설정 적용
_configure_pytorch_backends()


def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """safetensors 파일 로드"""
    return load_file(path)


def transformers_convert(sd: Dict[str, torch.Tensor], prefix_from: str, prefix_to: str, number: int) -> Dict[str, torch.Tensor]:
    """
    OpenAI CLIP 형식 → HuggingFace Transformers 형식 변환

    ComfyUI comfy/utils.py의 transformers_convert() 포팅
    """
    # 특수 키 변환 (embeddings, final_layer_norm)
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from*x:shape_from*(x + 1)]

    return sd


def convert_clip_l_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    CLIP-L 키 변환

    SDXL 체크포인트의 CLIP-L은 이미 HuggingFace 형식 (text_model.*)
    """
    # 이미 text_model.* prefix가 있으면 그대로 반환
    if any(k.startswith("text_model.") for k in sd.keys()):
        return sd

    # text_model. prefix 추가
    new_sd = {}
    for k, v in sd.items():
        new_sd["text_model." + k] = v

    return new_sd


def convert_clip_g_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    CLIP-G 키를 HuggingFace 형식으로 변환

    SDXL 체크포인트의 CLIP-G는 OpenAI 형식:
    - transformer.resblocks.* → text_model.encoder.layers.*
    - positional_embedding → text_model.embeddings.position_embedding.weight
    - token_embedding.weight → text_model.embeddings.token_embedding.weight
    - ln_final.* → text_model.final_layer_norm.*
    """
    # 이미 HuggingFace 형식이면 그대로 반환
    if any(k.startswith("text_model.") for k in sd.keys()):
        return sd

    # OpenAI 형식 → HuggingFace 형식 변환
    # prefix_from="" 이므로 transformer.resblocks.*, positional_embedding 등을 찾음
    sd = transformers_convert(sd, "", "text_model.", 32)

    # text_projection 변환 (있는 경우)
    if "text_projection" in sd:
        # OpenAI는 [embed_dim, embed_dim], transpose 필요
        sd["text_projection.weight"] = sd.pop("text_projection").T.contiguous()

    return sd


def extract_unet_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    체크포인트에서 UNet state_dict 추출

    SDXL 체크포인트에서 'model.diffusion_model.' prefix 제거
    """
    unet_sd = {}
    prefix = "model.diffusion_model."

    for key, value in sd.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            unet_sd[new_key] = value

    return unet_sd


def extract_vae_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    체크포인트에서 VAE state_dict 추출

    'first_stage_model.' prefix 제거
    """
    vae_sd = {}
    prefix = "first_stage_model."

    for key, value in sd.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            vae_sd[new_key] = value

    return vae_sd


def extract_clip_state_dicts(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    체크포인트에서 CLIP state_dict 추출

    SDXL은 두 개의 CLIP 사용:
    - CLIP-L: 'conditioner.embedders.0.transformer.text_model.' (이미 HuggingFace 형식)
    - CLIP-G: 'conditioner.embedders.1.model.' (OpenAI 형식)

    Returns:
        (clip_l_sd, clip_g_sd)
    """
    clip_l_sd = {}
    clip_g_sd = {}

    # CLIP-L 추출 (이미 HuggingFace 형식 - text_model.* prefix 유지)
    # conditioner.embedders.0.transformer.text_model.embeddings.* → text_model.embeddings.*
    clip_l_prefix = "conditioner.embedders.0.transformer."
    for key, value in sd.items():
        if key.startswith(clip_l_prefix):
            new_key = key[len(clip_l_prefix):]
            clip_l_sd[new_key] = value

    # CLIP-G 추출 (OpenAI 형식 → HuggingFace 형식 변환 필요)
    # conditioner.embedders.1.model.transformer.resblocks.* → text_model.encoder.layers.*
    clip_g_prefix = "conditioner.embedders.1.model."
    clip_g_raw = {}
    for key, value in sd.items():
        if key.startswith(clip_g_prefix):
            new_key = key[len(clip_g_prefix):]
            clip_g_raw[new_key] = value

    # OpenAI 형식 → HuggingFace 형식 변환
    clip_g_sd = convert_clip_g_keys(clip_g_raw)

    logging.debug(f"CLIP-L sample keys: {list(clip_l_sd.keys())[:5]}")
    logging.debug(f"CLIP-G sample keys: {list(clip_g_sd.keys())[:5]}")

    return clip_l_sd, clip_g_sd


class SDXLModelLoader:
    """
    SDXL 모델 로더

    safetensors 체크포인트에서 UNet, VAE, CLIP을 로드
    """

    def __init__(self, dtype: torch.dtype = torch.float16, device: str = "cuda"):
        self.dtype = dtype
        self.device = device
        self._cache = {}

    def load_checkpoint(self, path: str) -> Dict[str, torch.Tensor]:
        """체크포인트 로드 (캐싱 제거 - VRAM 절약)"""
        path = str(Path(path).resolve())

        # 캐시 비활성화: SDXL 체크포인트는 ~6.5GB이므로 캐시하면 메모리 부족
        # 모델 로드 후에는 체크포인트가 필요없음
        logging.info(f"Loading checkpoint: {path}")
        sd = load_safetensors(path)
        return sd

    def load_unet(self, path: str, config: Optional[Dict] = None) -> UNetModel:
        """UNet 모델 로드"""
        sd = self.load_checkpoint(path)
        unet_sd = extract_unet_state_dict(sd)

        logging.info(f"UNet extracted keys: {len(unet_sd)}")
        if unet_sd:
            logging.info(f"UNet sample keys: {list(unet_sd.keys())[:5]}")

        if config is None:
            config = SDXL_UNET_CONFIG.copy()

        unet = UNetModel(**config, dtype=self.dtype, device="cpu")
        missing, unexpected = unet.load_state_dict(unet_sd, strict=False)

        logging.info(f"UNet missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if missing:
            logging.info(f"UNet missing sample: {missing[:5]}")
        if unexpected:
            logging.info(f"UNet unexpected sample: {unexpected[:5]}")

        unet = unet.to(device=self.device, dtype=self.dtype)
        unet.eval()

        # Attention 백엔드 정보 출력
        if XFORMERS_AVAILABLE:
            logging.info("Attention backend: xformers (memory_efficient_attention)")
        else:
            logging.info("Attention backend: PyTorch SDPA (scaled_dot_product_attention)")

        # 워밍업: CUDA 커널 컴파일을 위한 더미 forward pass (최소한만)
        if self.device == "cuda":
            logging.info("UNet warmup...")
            with torch.no_grad():
                # 기본 해상도만 워밍업 (1024x1024 -> 128x128, context 77)
                dummy_x = torch.zeros(2, 4, 128, 128, device=self.device, dtype=self.dtype)
                dummy_t = torch.zeros(2, device=self.device, dtype=self.dtype)
                dummy_c = torch.zeros(2, 77, 2048, device=self.device, dtype=self.dtype)
                dummy_y = torch.zeros(2, 2816, device=self.device, dtype=self.dtype)
                _ = unet(dummy_x, timesteps=dummy_t, context=dummy_c, y=dummy_y)
                # 워밍업 텐서 즉시 해제
                del dummy_x, dummy_t, dummy_c, dummy_y
                torch.cuda.empty_cache()
            logging.info("UNet warmup done")

        return unet

    def load_vae(self, path: str, config: Optional[Dict] = None) -> AutoencoderKL:
        """VAE 모델 로드"""
        sd = self.load_checkpoint(path)
        vae_sd = extract_vae_state_dict(sd)

        logging.info(f"VAE extracted keys: {len(vae_sd)}")
        if vae_sd:
            logging.info(f"VAE sample keys: {list(vae_sd.keys())[:5]}")

        if config is None:
            config = SDXL_VAE_CONFIG.copy()

        vae = AutoencoderKL(**config)
        missing, unexpected = vae.load_state_dict(vae_sd, strict=False)

        logging.info(f"VAE missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if missing:
            logging.info(f"VAE missing sample: {missing[:5]}")

        vae = vae.to(device=self.device, dtype=self.dtype)
        vae.eval()

        # 워밍업: CUDA 커널 컴파일을 위한 더미 forward pass (최소한만)
        if self.device == "cuda":
            logging.info("VAE warmup...")
            with torch.no_grad():
                # 기본 해상도만 워밍업 (1024x1024 -> 128x128)
                dummy_latent = torch.zeros(1, 4, 128, 128, device=self.device, dtype=self.dtype)
                _ = vae.decode(dummy_latent)
                del dummy_latent
                torch.cuda.empty_cache()
            logging.info("VAE warmup done")

        return vae

    def load_clip(self, path: str) -> SDXLClipModel:
        """CLIP 모델 로드"""
        sd = self.load_checkpoint(path)
        clip_l_sd, clip_g_sd = extract_clip_state_dicts(sd)

        logging.info(f"CLIP-L extracted keys: {len(clip_l_sd)}")
        logging.info(f"CLIP-G extracted keys: {len(clip_g_sd)}")

        # extract_clip_state_dicts에서 이미 변환되었으므로 추가 변환 불필요
        # CLIP-L은 이미 HuggingFace 형식, CLIP-G는 extract에서 convert_clip_g_keys 적용됨
        if clip_l_sd:
            logging.info(f"CLIP-L sample keys: {list(clip_l_sd.keys())[:3]}")
        if clip_g_sd:
            logging.info(f"CLIP-G sample keys: {list(clip_g_sd.keys())[:3]}")

        clip = SDXLClipModel(device="cpu", dtype=torch.float32)  # fp32로 CLIP 유지

        if clip_l_sd:
            missing, unexpected = clip.load_clip_l(clip_l_sd)
            if missing:
                logging.warning(f"CLIP-L missing keys: {len(missing)}")
                logging.info(f"CLIP-L missing sample: {missing[:5]}")
            if unexpected:
                logging.info(f"CLIP-L unexpected keys: {len(unexpected)}")

        if clip_g_sd:
            missing, unexpected = clip.load_clip_g(clip_g_sd)
            if missing:
                logging.warning(f"CLIP-G missing keys: {len(missing)}")
                logging.info(f"CLIP-G missing sample: {missing[:5]}")
            if unexpected:
                logging.info(f"CLIP-G unexpected keys: {len(unexpected)}")

        clip = clip.to(device=self.device)
        return clip

    def load_all(self, path: str) -> Tuple[UNetModel, AutoencoderKL, SDXLClipModel]:
        """UNet, VAE, CLIP 모두 로드 - 체크포인트 1회만 로드"""
        # 체크포인트를 1번만 로드하여 메모리 절약
        sd = self.load_checkpoint(path)

        # UNet 로드 (ComfyUI 방식: bfloat16 사용 가능하면 bfloat16)
        # bfloat16은 float16보다 수치 범위가 넓어 안정적
        unet_sd = extract_unet_state_dict(sd)
        logging.info(f"UNet extracted keys: {len(unet_sd)}")
        # bfloat16 지원 여부에 따라 dtype 결정
        import os
        force_fp16 = os.environ.get("PEROPIX_FORCE_FP16", "0") == "1"
        if force_fp16:
            unet_dtype = torch.float16
            logging.info("UNet dtype: float16 (forced by PEROPIX_FORCE_FP16=1)")
        elif torch.cuda.is_bf16_supported():
            unet_dtype = torch.bfloat16
            logging.info("UNet dtype: bfloat16 (CUDA bf16 supported)")
        else:
            unet_dtype = self.dtype
            logging.info(f"UNet dtype: {unet_dtype}")
        unet = UNetModel(**SDXL_UNET_CONFIG.copy(), dtype=unet_dtype, device="cpu")
        unet.load_state_dict(unet_sd, strict=False)
        del unet_sd  # 메모리 해제
        unet = unet.to(device=self.device, dtype=unet_dtype)
        unet.eval()

        # VAE 로드 (ComfyUI 방식: bfloat16 사용)
        vae_sd = extract_vae_state_dict(sd)
        logging.info(f"VAE extracted keys: {len(vae_sd)}")
        vae = AutoencoderKL(**SDXL_VAE_CONFIG.copy())
        vae.load_state_dict(vae_sd, strict=False)
        del vae_sd  # 메모리 해제
        # VAE는 bfloat16 사용 (ComfyUI 기본값, 색상 정확도 향상)
        vae_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else self.dtype
        vae = vae.to(device=self.device, dtype=vae_dtype)
        vae.eval()
        logging.info(f"VAE dtype: {vae_dtype}")

        # CLIP 로드 (ComfyUI 방식: fp32로 유지하여 정밀도 보장)
        clip_l_sd, clip_g_sd = extract_clip_state_dicts(sd)
        logging.info(f"CLIP-L extracted keys: {len(clip_l_sd)}, CLIP-G: {len(clip_g_sd)}")
        clip = SDXLClipModel(device="cpu", dtype=torch.float32)  # fp32로 CLIP 유지
        if clip_l_sd:
            clip.load_clip_l(clip_l_sd)
        if clip_g_sd:
            clip.load_clip_g(clip_g_sd)
        del clip_l_sd, clip_g_sd  # 메모리 해제
        clip = clip.to(device=self.device)
        logging.info("CLIP dtype: float32 (for precision)")

        # 체크포인트 메모리 해제 (중요!)
        del sd
        torch.cuda.empty_cache()

        # 워밍업 (최소화: 1회만)
        if self.device == "cuda":
            logging.info("Model warmup (minimal)...")
            with torch.no_grad():
                # 가장 일반적인 설정으로 1회만 워밍업
                # UNet dtype과 일치해야 함
                dummy_x = torch.zeros(2, 4, 128, 128, device=self.device, dtype=unet_dtype)
                dummy_t = torch.zeros(2, device=self.device, dtype=unet_dtype)
                dummy_c = torch.zeros(2, 77, 2048, device=self.device, dtype=unet_dtype)
                dummy_y = torch.zeros(2, 2816, device=self.device, dtype=unet_dtype)
                _ = unet(dummy_x, timesteps=dummy_t, context=dummy_c, y=dummy_y)
                del dummy_x, dummy_t, dummy_c, dummy_y
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            logging.info("Model warmup done")

        return unet, vae, clip

    def clear_cache(self):
        """캐시 클리어"""
        self._cache.clear()
        torch.cuda.empty_cache()


class ModelCache:
    """
    모델 캐시 관리

    로드된 모델을 캐시하여 재로딩 방지
    """

    def __init__(self, dtype: torch.dtype = torch.float16, device: str = "cuda"):
        self.loader = SDXLModelLoader(dtype=dtype, device=device)
        self._models: Dict[str, Any] = {}
        self._current_path: Optional[str] = None

    def get_models(self, path: str) -> Tuple[UNetModel, AutoencoderKL, SDXLClipModel]:
        """
        모델 가져오기 (캐시된 경우 재사용)

        Args:
            path: 체크포인트 경로

        Returns:
            (unet, vae, clip)
        """
        path = str(Path(path).resolve())

        if self._current_path == path and path in self._models:
            return self._models[path]

        # 이전 모델 언로드
        self.unload()

        # 새 모델 로드
        unet, vae, clip = self.loader.load_all(path)
        self._models[path] = (unet, vae, clip)
        self._current_path = path

        return unet, vae, clip

    def unload(self):
        """현재 모델 언로드 - VRAM 완전 해제"""
        # 모델들을 CPU로 이동 후 삭제 (VRAM 해제)
        for path, models in self._models.items():
            unet, vae, clip = models
            if unet is not None:
                unet.to("cpu")
                del unet
            if vae is not None:
                vae.to("cpu")
                del vae
            if clip is not None:
                clip.to("cpu")
                del clip

        self._models.clear()
        self._current_path = None
        self.loader.clear_cache()

        # 강제 가비지 컬렉션 후 VRAM 해제
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# 전역 모델 캐시
_model_cache: Optional[ModelCache] = None


def get_model_cache(dtype: torch.dtype = torch.float16, device: str = "cuda") -> ModelCache:
    """전역 모델 캐시 가져오기"""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache(dtype=dtype, device=device)
    return _model_cache


def load_sdxl_model(path: str, dtype: torch.dtype = torch.float16, device: str = "cuda") -> Tuple[UNetModel, AutoencoderKL, SDXLClipModel]:
    """
    SDXL 모델 로드 (편의 함수)

    Args:
        path: 체크포인트 경로
        dtype: 모델 dtype
        device: 디바이스

    Returns:
        (unet, vae, clip)
    """
    cache = get_model_cache(dtype=dtype, device=device)
    return cache.get_models(path)
