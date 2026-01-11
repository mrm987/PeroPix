"""
ComfyUI 유틸리티 함수 포팅

원본: ComfyUI/comfy/k_diffusion/utils.py
"""

import torch


def append_dims(x, target_dims):
    """
    텐서 끝에 차원을 추가하여 target_dims 차원이 되게 합니다.

    ComfyUI 원본 그대로
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    expanded = x[(...,) + (None,) * dims_to_append]
    # MPS 디바이스에서는 clone이 필요
    return expanded.detach().clone() if expanded.device.type == 'mps' else expanded


def append_zero(x):
    """텐서 끝에 0을 추가합니다."""
    return torch.cat([x, x.new_zeros([1])])
