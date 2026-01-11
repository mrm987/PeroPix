"""
ComfyUI 스케줄러 포팅

원본: ComfyUI/comfy/samplers.py, ComfyUI/comfy/k_diffusion/sampling.py
"""

import math
import torch
import numpy as np
from .utils import append_zero


def normal_scheduler(model_sampling, steps, sgm=False, floor=False):
    """
    Normal 스케줄러

    ComfyUI 핵심 로직:
    1. timestep 공간에서 선형 보간
    2. 각 timestep을 sigma로 변환

    이것이 Diffusers와의 핵심 차이점입니다.
    Diffusers는 sigma 공간에서 직접 선형 보간합니다.
    """
    s = model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)

    append_zero = True
    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        # ComfyUI 방식: sigma(end)가 0에 가까우면 steps+1하고 append_zero=False
        if math.isclose(float(s.sigma(end)), 0, abs_tol=0.00001):
            steps += 1
            append_zero = False
        timesteps = torch.linspace(start, end, steps)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(float(s.sigma(ts)))

    if append_zero:
        sigs += [0.0]

    return torch.FloatTensor(sigs)


def simple_scheduler(model_sampling, steps):
    """
    Simple 스케줄러

    전체 시그마 배열에서 균등하게 샘플링합니다.
    ComfyUI 원본 그대로
    """
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs.append(float(s.sigmas[-(1 + int(x * ss))]))
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def sgm_uniform(model_sampling, steps):
    """SGM Uniform = normal_scheduler with sgm=True"""
    return normal_scheduler(model_sampling, steps, sgm=True)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """
    Karras et al. (2022)의 노이즈 스케줄 구성

    ComfyUI 원본 그대로
    """
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Exponential 노이즈 스케줄"""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def ddim_scheduler(model_sampling, steps):
    """
    DDIM Uniform 스케줄러

    ComfyUI 원본: 균등하게 간격을 둔 시그마 값들을 역순으로 정렬
    """
    s = model_sampling
    sigs = []
    x = 1
    if math.isclose(float(s.sigmas[x]), 0, abs_tol=0.00001):
        steps += 1
        sigs = []
    else:
        sigs = [0.0]

    ss = max(len(s.sigmas) // steps, 1)
    while x < len(s.sigmas):
        sigs.append(float(s.sigmas[x]))
        x += ss
    sigs = sigs[::-1]
    return torch.FloatTensor(sigs)


def beta_scheduler(model_sampling, steps, alpha=0.6, beta=0.6):
    """
    Beta 스케줄러

    ComfyUI 원본: 베타 분포의 역누적분포함수를 이용해 시간 단계를 샘플링
    scipy 없이 numpy로 근사 구현
    """
    try:
        from scipy.stats import beta as beta_dist
        has_scipy = True
    except ImportError:
        has_scipy = False

    total_timesteps = len(model_sampling.sigmas) - 1
    ts = 1 - np.linspace(0, 1, steps, endpoint=False)

    if has_scipy:
        ts = np.rint(beta_dist.ppf(ts, alpha, beta) * total_timesteps)
    else:
        # scipy 없으면 simple 스케줄러로 폴백
        ts = np.rint(ts * total_timesteps)

    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs.append(float(model_sampling.sigmas[int(t)]))
        last_t = t
    sigs.append(0.0)
    return torch.FloatTensor(sigs)


# 스케줄러 레지스트리
SCHEDULERS = {
    "normal": lambda ms, steps: normal_scheduler(ms, steps, sgm=False),
    "sgm_uniform": lambda ms, steps: normal_scheduler(ms, steps, sgm=True),
    "simple": simple_scheduler,
    "ddim_uniform": ddim_scheduler,
    "beta": beta_scheduler,
    "karras": None,  # 별도 처리 필요 (sigma_min/max 필요)
    "exponential": None,  # 별도 처리 필요
}


def get_sigmas(model_sampling, scheduler_name, steps, device='cpu'):
    """
    스케줄러 이름과 스텝 수로 시그마 배열 생성

    Args:
        model_sampling: ModelSamplingDiscrete 인스턴스
        scheduler_name: 스케줄러 이름
        steps: 샘플링 스텝 수
        device: 디바이스

    Returns:
        torch.Tensor: 시그마 배열 (길이: steps + 1, 마지막은 0)
    """
    if scheduler_name == "karras":
        return get_sigmas_karras(
            steps,
            float(model_sampling.sigma_min),
            float(model_sampling.sigma_max),
            device=device
        )
    elif scheduler_name == "exponential":
        return get_sigmas_exponential(
            steps,
            float(model_sampling.sigma_min),
            float(model_sampling.sigma_max),
            device=device
        )

    scheduler_fn = SCHEDULERS.get(scheduler_name)
    if scheduler_fn is None:
        raise ValueError(f"Unknown scheduler: {scheduler_name}. Available: {list(SCHEDULERS.keys())}")

    return scheduler_fn(model_sampling, steps).to(device)
