"""
SDXL 이미지 생성

ComfyUI 방식의 샘플링으로 txt2img, img2img, inpaint 지원
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
import logging

from .model_sampling import ModelSamplingDiscrete, EPS
from .samplers import sample_euler, sample_euler_ancestral, SAMPLERS
from .schedulers import get_sigmas, SCHEDULERS
from .model_loader import load_sdxl_model, get_model_cache
from .lora import LoRALoader


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL 이미지를 [0, 1] 범위 텐서로 변환"""
    image = image.convert("RGB")
    np_image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """[0, 1] 범위 텐서를 PIL 이미지로 변환"""
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    tensor = tensor.clamp(0, 1)
    np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_image)


def encode_image(vae, image: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    """이미지를 latent로 인코딩"""
    # [0, 1] -> [-1, 1]
    image = image * 2.0 - 1.0
    image = image.to(device=device, dtype=dtype)

    with torch.no_grad():
        latent = vae.encode(image)

    # VAE 스케일 팩터 적용 (SDXL: 0.13025)
    latent = latent * 0.13025
    return latent


def decode_latent(vae, latent: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """latent를 이미지로 디코딩"""
    # VAE 스케일 팩터 역적용
    latent = latent / 0.13025

    with torch.no_grad():
        image = vae.decode(latent.to(dtype))

    # [-1, 1] -> [0, 1]
    image = (image + 1.0) / 2.0
    return image


class SDXLGenerator:
    """
    SDXL 이미지 생성기

    ComfyUI 방식의 샘플링으로 동일한 결과물 생성
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path

        # 모델 로드
        self.unet, self.vae, self.clip = load_sdxl_model(
            model_path, dtype=dtype, device=device
        )

        # 모델 샘플링 설정 (GPU에서 실행)
        self.model_sampling = ModelSamplingDiscrete().to(device)
        self.eps = EPS()

        # LoRA 로더
        self.lora_loader = LoRALoader(device=device, dtype=dtype)

    def apply_lora(self, lora_path: str, scale: float = 1.0):
        """LoRA 적용"""
        self.lora_loader.apply_lora_to_model(self.unet, lora_path, scale)

    def remove_loras(self):
        """모든 LoRA 제거"""
        self.lora_loader.remove_lora(self.unet)

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = ""
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        프롬프트 인코딩

        Returns:
            (cond, cond_pooled, uncond, uncond_pooled)
        """
        # Positive
        cond, cond_pooled = self.clip.encode_text(prompt)
        cond = cond.to(device=self.device, dtype=self.dtype)
        cond_pooled = cond_pooled.to(device=self.device, dtype=self.dtype)

        # Negative
        if negative_prompt:
            uncond, uncond_pooled = self.clip.encode_text(negative_prompt)
        else:
            uncond, uncond_pooled = self.clip.encode_text("")
        uncond = uncond.to(device=self.device, dtype=self.dtype)
        uncond_pooled = uncond_pooled.to(device=self.device, dtype=self.dtype)

        return cond, cond_pooled, uncond, uncond_pooled

    def prepare_y(
        self,
        pooled: torch.Tensor,
        width: int,
        height: int,
        crop_w: int = 0,
        crop_h: int = 0,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None
    ) -> torch.Tensor:
        """
        SDXL y (class embeddings) 준비

        pooled: [B, 1280] CLIP-G pooled output
        + size embeddings (original size, crop coords, target size)

        ComfyUI의 SDXL.encode_adm() 방식:
        각 size 값에 sinusoidal timestep embedding (256d) 적용
        """
        from .ldm.modules.diffusionmodules.util import timestep_embedding

        if target_width is None:
            target_width = width
        if target_height is None:
            target_height = height

        batch_size = pooled.shape[0]

        # Size embeddings: 각 값에 256차원 sinusoidal embedding 적용
        # ComfyUI: self.embedder = Timestep(256)
        size_values = [height, width, crop_h, crop_w, target_height, target_width]
        size_embs = []
        for val in size_values:
            # timestep_embedding expects [B] tensor, returns [B, dim]
            t = torch.tensor([val], dtype=torch.float32, device=self.device)
            emb = timestep_embedding(t, 256)  # [1, 256]
            size_embs.append(emb)

        # Concatenate all size embeddings: [1, 256*6] = [1, 1536]
        size_emb = torch.cat(size_embs, dim=-1)  # [1, 1536]
        size_emb = size_emb.repeat(batch_size, 1)  # [B, 1536]
        size_emb = size_emb.to(dtype=self.dtype)

        # SDXL y: [B, 2816] = pooled (1280) + size_emb (1536)
        y = torch.cat([pooled, size_emb], dim=-1)

        return y

    def get_noise(
        self,
        batch_size: int,
        height: int,
        width: int,
        seed: int
    ) -> torch.Tensor:
        """
        초기 노이즈 생성

        ComfyUI와 동일한 방식:
        1. torch.manual_seed(seed)
        2. torch.randn(..., dtype=latent_image.dtype) - 즉 float32
        3. CPU에서 생성, device로 이동

        중요: dtype은 float32로 유지 (ComfyUI 방식)
        """
        latent_height = height // 8
        latent_width = width // 8

        # ComfyUI 방식: float32로 생성
        generator = torch.manual_seed(seed)
        noise = torch.randn(
            (batch_size, 4, latent_height, latent_width),
            dtype=torch.float32,  # ComfyUI: latent_image.dtype = float32
            generator=generator,
            device="cpu"
        ).to(device=self.device)  # device만 이동, dtype은 float32 유지

        return noise

    def model_wrapper(
        self,
        cond: torch.Tensor,
        cond_pooled: torch.Tensor,
        uncond: torch.Tensor,
        uncond_pooled: torch.Tensor,
        cfg_scale: float,
        width: int,
        height: int
    ):
        """
        CFG가 적용된 모델 래퍼

        샘플러에서 호출되는 함수

        중요: UNet은 eps (noise prediction)을 출력합니다.
        샘플러는 denoised 이미지를 기대하므로 변환이 필요합니다.
        denoised = x - eps * sigma
        """
        # y 준비
        y_cond = self.prepare_y(cond_pooled, width, height)
        y_uncond = self.prepare_y(uncond_pooled, width, height)

        # cond와 uncond의 시퀀스 길이를 맞춤 (긴 프롬프트 지원)
        # 짧은 쪽을 0으로 패딩
        cond_len = cond.shape[1]
        uncond_len = uncond.shape[1]
        if cond_len != uncond_len:
            max_len = max(cond_len, uncond_len)
            if cond_len < max_len:
                pad = torch.zeros(
                    (cond.shape[0], max_len - cond_len, cond.shape[2]),
                    dtype=cond.dtype, device=cond.device
                )
                cond = torch.cat([cond, pad], dim=1)
            if uncond_len < max_len:
                pad = torch.zeros(
                    (uncond.shape[0], max_len - uncond_len, uncond.shape[2]),
                    dtype=uncond.dtype, device=uncond.device
                )
                uncond = torch.cat([uncond, pad], dim=1)

        def wrapper(x, sigma, **kwargs):
            # sigma -> timestep (ComfyUI model_base.py:182 참조)
            # timestep = self.model_sampling.timestep(t).float()
            timestep = self.model_sampling.timestep(sigma).float()
            timestep = timestep.expand(x.shape[0])

            # 입력 스케일링: x_scaled = x / sqrt(sigma^2 + sigma_data^2)
            # ComfyUI의 EPS.calculate_input과 동일
            x_scaled = self.eps.calculate_input(sigma, x)

            # 배치 처리 (cond + uncond)
            x_in = torch.cat([x_scaled, x_scaled], dim=0)
            t_in = torch.cat([timestep, timestep], dim=0)
            c_in = torch.cat([cond, uncond], dim=0)
            y_in = torch.cat([y_cond, y_uncond], dim=0)

            # UNet forward - eps (noise prediction) 출력
            with torch.no_grad():
                eps_out = self.unet(x_in, timesteps=t_in, context=c_in, y=y_in)

            # eps -> denoised 변환 (각각 별도로)
            # ComfyUI는 apply_model에서 calculate_denoised까지 처리하여 denoised를 반환
            eps_cond, eps_uncond = eps_out.chunk(2)
            denoised_cond = self.eps.calculate_denoised(sigma, eps_cond.float(), x)
            denoised_uncond = self.eps.calculate_denoised(sigma, eps_uncond.float(), x)

            # CFG 적용 (denoised 공간에서) - ComfyUI 방식
            # cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale
            cfg_result = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)

            return cfg_result

        return wrapper

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        sampler: str = "euler",
        scheduler: str = "normal",
        base_image: Optional[Image.Image] = None,
        mask: Optional[Image.Image] = None,
        denoise: float = 1.0,
        callback: Optional[callable] = None
    ) -> Tuple[Image.Image, int]:
        """
        이미지 생성

        Args:
            prompt: 프롬프트
            negative_prompt: 네거티브 프롬프트
            width: 이미지 너비
            height: 이미지 높이
            steps: 샘플링 스텝
            cfg_scale: CFG 스케일
            seed: 랜덤 시드 (-1이면 랜덤)
            sampler: 샘플러 이름 (euler, euler_ancestral)
            scheduler: 스케줄러 이름 (normal, karras, simple, sgm_uniform, exponential)
            base_image: img2img/inpaint 베이스 이미지
            mask: 인페인트 마스크 (검정=유지, 흰색=인페인트)
            denoise: 디노이즈 강도 (0.0-1.0)
            callback: 진행 콜백

        Returns:
            (생성된 이미지, 사용된 시드)
        """
        # 시드 처리
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()

        # 프롬프트 인코딩
        cond, cond_pooled, uncond, uncond_pooled = self.encode_prompt(prompt, negative_prompt)

        # 디버그: CLIP 출력 확인
        logging.info(f"[Debug CLIP] cond shape: {cond.shape}, cond[0,0,:5]: {cond[0,0,:5].tolist()}")
        logging.info(f"[Debug CLIP] cond_pooled shape: {cond_pooled.shape}, cond_pooled[0,:5]: {cond_pooled[0,:5].tolist()}")
        logging.info(f"[Debug CLIP] uncond shape: {uncond.shape}, uncond[0,0,:5]: {uncond[0,0,:5].tolist()}")

        # 시그마 스케줄 생성
        sigmas = get_sigmas(self.model_sampling, scheduler, steps, device=self.device)

        # max_denoise 판단 (ComfyUI 방식: samplers.py:718-721)
        # math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma
        import math
        sigma_max = float(self.model_sampling.sigma_max)
        sigma_start = float(sigmas[0])
        is_max_denoise = math.isclose(sigma_max, sigma_start, rel_tol=1e-05) or sigma_start > sigma_max

        # 디버그: sigmas 출력
        logging.info(f"[Debug] sigma_max: {sigma_max:.6f}, sigma_start: {sigma_start:.6f}, is_max_denoise: {is_max_denoise}")
        logging.info(f"[Debug] sigmas (first 5): {[f'{s:.4f}' for s in sigmas[:5].tolist()]}")
        logging.info(f"[Debug] sigmas (last 5): {[f'{s:.4f}' for s in sigmas[-5:].tolist()]}")

        # img2img/inpaint 처리
        if base_image is not None:
            # 이미지를 latent로 인코딩
            image_tensor = pil_to_tensor(base_image.resize((width, height)))
            latent = encode_image(self.vae, image_tensor, self.device, self.dtype)

            # denoise에 따라 시그마 조정
            denoise_steps = int(steps * denoise)
            if denoise_steps < steps:
                sigmas = sigmas[-(denoise_steps + 1):]
                # denoise < 1.0이면 max_denoise가 아님
                is_max_denoise = False

            # 초기 latent = noise_scaling(sigma[0], noise, latent)
            noise = self.get_noise(1, height, width, seed)
            x = self.eps.noise_scaling(sigmas[0], noise, latent, max_denoise=is_max_denoise)
        else:
            # txt2img: ComfyUI 방식의 noise_scaling 사용
            # latent_image는 0 텐서
            noise = self.get_noise(1, height, width, seed)
            latent_image = torch.zeros_like(noise)
            x = self.eps.noise_scaling(sigmas[0], noise, latent_image, max_denoise=is_max_denoise)

        # 디버그: noise와 x 초기값 출력
        logging.info(f"[Debug] noise dtype: {noise.dtype}, device: {noise.device}, shape: {noise.shape}")
        logging.info(f"[Debug] noise[0,0,0,:5]: {noise[0,0,0,:5].tolist()}")
        logging.info(f"[Debug] x dtype: {x.dtype}, device: {x.device}, shape: {x.shape}")
        logging.info(f"[Debug] x[0,0,0,:5]: {x[0,0,0,:5].tolist()}")

        # 인페인트 마스크 처리
        latent_mask = None
        original_latent = None
        if mask is not None and base_image is not None:
            # 마스크를 latent 크기로 리사이즈
            mask_resized = mask.resize((width // 8, height // 8), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_resized) / 255.0)
            latent_mask = mask_tensor.unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=self.dtype)

            # 원본 latent 저장
            image_tensor = pil_to_tensor(base_image.resize((width, height)))
            original_latent = encode_image(self.vae, image_tensor, self.device, self.dtype)

        # 모델 래퍼 생성
        model_fn = self.model_wrapper(
            cond, cond_pooled, uncond, uncond_pooled,
            cfg_scale, width, height
        )

        # 샘플러 선택
        if sampler == "euler_ancestral":
            sampler_fn = sample_euler_ancestral
        else:
            sampler_fn = sample_euler

        # 콜백 래퍼
        def step_callback(info):
            if callback:
                step = info.get("i", 0)
                total = len(sigmas) - 1
                callback(step, total)

        # 샘플링
        import time as _time
        _t0 = _time.perf_counter()

        samples = sampler_fn(
            model_fn, x, sigmas,
            extra_args={"seed": seed},
            callback=step_callback if callback else None,
            disable=True  # tqdm 비활성화 - 콘솔 출력 오버헤드 방지
        )

        if self.device == "cuda":
            torch.cuda.synchronize()
        _t1 = _time.perf_counter()
        logging.info(f"[Timing] Sampling: {(_t1-_t0)*1000:.1f}ms")

        # 디버그: 샘플링 결과 확인
        logging.info(f"[Debug] samples has NaN: {torch.isnan(samples).any()}")
        logging.info(f"[Debug] samples min/max: {samples.min():.4f} / {samples.max():.4f}")

        # 인페인트: 마스크된 영역만 업데이트
        if latent_mask is not None and original_latent is not None:
            samples = original_latent * (1 - latent_mask) + samples * latent_mask

        # VAE 디코딩
        _t0 = _time.perf_counter()
        image = decode_latent(self.vae, samples, self.dtype)
        if self.device == "cuda":
            torch.cuda.synchronize()
        _t1 = _time.perf_counter()
        logging.info(f"[Timing] VAE decode: {(_t1-_t0)*1000:.1f}ms")

        # 디버그: VAE 디코딩 결과 확인
        logging.info(f"[Debug] decoded has NaN: {torch.isnan(image).any()}")
        logging.info(f"[Debug] decoded min/max: {image.min():.4f} / {image.max():.4f}")

        pil_image = tensor_to_pil(image)

        return pil_image, seed


# 편의 함수
def create_generator(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> SDXLGenerator:
    """SDXL 생성기 생성"""
    return SDXLGenerator(model_path, device, dtype)


def generate(
    model_path: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg_scale: float = 7.0,
    seed: int = -1,
    sampler: str = "euler",
    scheduler: str = "normal",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    **kwargs
) -> Tuple[Image.Image, int]:
    """
    이미지 생성 편의 함수

    매번 모델을 로드하므로 배치 생성에는 SDXLGenerator 사용 권장
    """
    generator = SDXLGenerator(model_path, device, dtype)
    return generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        sampler=sampler,
        scheduler=scheduler,
        **kwargs
    )
