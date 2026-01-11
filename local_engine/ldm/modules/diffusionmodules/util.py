"""
ComfyUI ldm util 포팅

원본: ComfyUI/comfy/ldm/modules/diffusionmodules/util.py
SDXL에 필요한 핵심 함수만 포팅
"""

import math
import torch
import torch.nn as nn
from einops import repeat


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """베타 스케줄 생성"""
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clamp(betas, min=0, max=0.999)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Sinusoidal timestep embeddings 생성

    Args:
        timesteps: [N] 텐서, 배치당 하나의 인덱스
        dim: 출력 차원
        max_period: 임베딩의 최소 주파수 제어

    Returns:
        [N x dim] positional embeddings 텐서
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """모듈의 파라미터를 0으로 초기화"""
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """모듈의 파라미터를 스케일링"""
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def checkpoint(func, inputs, params, flag):
    """
    Gradient checkpointing

    중간 활성화를 캐싱하지 않고 함수를 평가하여
    메모리를 절약하되 backward에서 추가 계산이 필요합니다.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled()
        }
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def avg_pool_nd(dims, *args, **kwargs):
    """1D, 2D, 또는 3D average pooling 모듈 생성"""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class AlphaBlender(nn.Module):
    """
    Temporal/Spatial 블렌딩을 위한 알파 블렌더

    Video 모델에서 사용되지만 SDXL 이미지 생성에서는 사용되지 않음
    """
    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "fixed",
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.tensor([alpha]))
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            self.register_parameter(
                "mix_factor", nn.Parameter(torch.tensor([alpha]))
            )
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator=None):
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            if image_only_indicator is None:
                alpha = torch.ones(1, 1, device=self.mix_factor.device)
            else:
                alpha = torch.where(
                    image_only_indicator.bool(),
                    torch.ones(1, 1, device=self.mix_factor.device),
                    repeat(torch.sigmoid(self.mix_factor), "1 -> b 1", b=image_only_indicator.shape[0]),
                )
        else:
            raise NotImplementedError()
        return alpha

    def forward(
        self,
        x_spatial,
        x_temporal,
        image_only_indicator=None,
    ):
        alpha = self.get_alpha(image_only_indicator)
        x = alpha * x_spatial + (1.0 - alpha) * x_temporal
        return x
