"""
ComfyUI 샘플러 포팅

원본: ComfyUI/comfy/k_diffusion/sampling.py
"""

import torch
from tqdm import tqdm
from .utils import append_dims


def to_d(x, sigma, denoised):
    """
    denoiser 출력을 Karras ODE 미분으로 변환

    ComfyUI 원본 그대로
    """
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """
    ancestral 샘플링에서 noise level (sigma_down)과
    추가할 노이즈 양 (sigma_up)을 계산합니다.

    ComfyUI 원본 그대로
    """
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x, seed=None):
    """
    기본 노이즈 샘플러

    ComfyUI 원본 그대로
    """
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None

    return lambda sigma, sigma_next: torch.randn(
        x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator
    )


@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                 s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """
    Karras et al. (2022)의 Algorithm 2 (Euler steps) 구현

    ComfyUI 원본 그대로
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in tqdm(range(len(sigmas) - 1), disable=disable):
        if s_churn > 0:
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt

    return x


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None,
                           eta=1., s_noise=1., noise_sampler=None):
    """
    Ancestral sampling with Euler method steps

    ComfyUI 원본 그대로
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in tqdm(range(len(sigmas) - 1), disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigma_down == 0:
            x = denoised
        else:
            d = to_d(x, sigmas[i], denoised)
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

    return x


# 샘플러 레지스트리
SAMPLERS = {
    "euler": sample_euler,
    "euler_ancestral": sample_euler_ancestral,
}


def get_sampler(name):
    """이름으로 샘플러 함수 반환"""
    if name not in SAMPLERS:
        raise ValueError(f"Unknown sampler: {name}. Available: {list(SAMPLERS.keys())}")
    return SAMPLERS[name]
