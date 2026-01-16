"""
ComfyUI attention 포팅

원본: ComfyUI/comfy/ldm/modules/attention.py
SDXL에 필요한 핵심 attention 모듈만 포팅
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional

from ...ops import disable_weight_init as ops


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


# SDPA 백엔드 설정 (ComfyUI ops.py 방식)
SDPA_ENABLED = False
SDPA_BACKEND_PRIORITY = None
_sdpa_kernel_context = None

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import inspect
    SDPA_ENABLED = True

    # Windows + CUDA에서 백엔드 우선순위 설정 (ComfyUI 방식: CUDNN > FLASH > EFFICIENT > MATH)
    if torch.cuda.is_available():
        if "set_priority" in inspect.signature(sdpa_kernel).parameters:
            # ComfyUI와 동일한 순서: CUDNN을 최우선으로
            SDPA_BACKEND_PRIORITY = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
            print(f"[Attention] SDPA with priority: CUDNN > FLASH > EFFICIENT > MATH")
        else:
            print(f"[Attention] SDPA enabled (torch version too old for priority)")
    else:
        print(f"[Attention] SDPA enabled (CPU mode)")
except ImportError:
    print("[Attention] SDPA not available, using basic attention")
    pass


# xformers 지원 체크
XFORMERS_AVAILABLE = False
try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
except ImportError:
    pass


# FeedForward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None, device=None, operations=ops):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out * 2, dtype=dtype, device=device)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., dtype=None, device=None, operations=ops):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            operations.Linear(dim, inner_dim, dtype=dtype, device=device),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim, dtype=dtype, device=device, operations=operations)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            operations.Linear(inner_dim, dim_out, dtype=dtype, device=device)
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels, dtype=None, device=None):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)


def attention_xformers(q, k, v, heads, mask=None, skip_reshape=False):
    """xformers memory_efficient_attention 사용 - 가장 빠른 옵션"""
    # dtype 통일
    if k.dtype != q.dtype:
        k = k.to(q.dtype)
    if v.dtype != q.dtype:
        v = v.to(q.dtype)

    if skip_reshape:
        # Already in (B, heads, seq, dim_head) format, convert to (B, seq, heads, dim_head)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    else:
        b, seq_len, inner_dim = q.shape
        dim_head = inner_dim // heads
        q = q.view(b, seq_len, heads, dim_head).contiguous()
        k = k.view(b, -1, heads, dim_head).contiguous()
        v = v.view(b, -1, heads, dim_head).contiguous()

    # xformers expects (B, seq, heads, dim_head) format
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
    # Reshape back to (B, seq, inner_dim)
    b, seq_len, heads, dim_head = out.shape
    out = out.reshape(b, seq_len, heads * dim_head)
    return out


def attention_pytorch(q, k, v, heads, mask=None, skip_reshape=False):
    """PyTorch scaled_dot_product_attention 사용 - SDPA 백엔드 우선순위 적용"""
    # dtype 통일 (q에 맞춤)
    if k.dtype != q.dtype:
        k = k.to(q.dtype)
    if v.dtype != q.dtype:
        v = v.to(q.dtype)

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    # Flash Attention은 contiguous 필요
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    # SDPA 사용 - 백엔드 우선순위 적용 (ComfyUI 방식)
    if SDPA_BACKEND_PRIORITY is not None:
        with sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


# 최적 attention 함수 선택
# 기본값: xformers 사용 가능하면 xformers, 아니면 SDPA
# 환경변수로 강제 지정 가능: PEROPIX_ATTENTION=sdpa 또는 PEROPIX_ATTENTION=xformers
import os
_FORCE_ATTENTION = os.environ.get("PEROPIX_ATTENTION", "").lower()

if _FORCE_ATTENTION == "sdpa":
    optimized_attention = attention_pytorch
    print(f"[Attention] Forced to use PyTorch SDPA (PEROPIX_ATTENTION=sdpa)")
elif _FORCE_ATTENTION == "xformers" and XFORMERS_AVAILABLE:
    optimized_attention = attention_xformers
    print(f"[Attention] Forced to use xformers (PEROPIX_ATTENTION=xformers)")
elif XFORMERS_AVAILABLE:
    optimized_attention = attention_xformers
    print(f"[Attention] Using xformers (memory_efficient_attention)")
else:
    optimized_attention = attention_pytorch
    print(f"[Attention] Using PyTorch SDPA (scaled_dot_product_attention)")


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., attn_precision=None, dtype=None, device=None, operations=ops):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.attn_precision = attn_precision

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = operations.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(operations.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout))

    def forward(self, x, context=None, value=None, mask=None, transformer_options={}):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        out = optimized_attention(q, k, v, self.heads, mask=mask)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, ff_in=False, inner_dim=None,
                 disable_self_attn=False, disable_temporal_crossattention=False, switch_temporal_ca_to_sa=False, attn_precision=None, dtype=None, device=None, operations=ops):
        super().__init__()

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        self.is_res = inner_dim == dim
        self.attn_precision = attn_precision

        if self.ff_in:
            self.norm_in = operations.LayerNorm(dim, dtype=dtype, device=device)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff, dtype=dtype, device=device, operations=operations)

        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None, attn_precision=self.attn_precision, dtype=dtype, device=device, operations=operations)
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff, dtype=dtype, device=device, operations=operations)

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            context_dim_attn2 = None
            if not switch_temporal_ca_to_sa:
                context_dim_attn2 = context_dim

            self.attn2 = CrossAttention(query_dim=inner_dim, context_dim=context_dim_attn2,
                                heads=n_heads, dim_head=d_head, dropout=dropout, attn_precision=self.attn_precision, dtype=dtype, device=device, operations=operations)
            self.norm2 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)

        self.norm1 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm3 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.n_heads = n_heads
        self.d_head = d_head
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

    def forward(self, x, context=None, transformer_options={}):
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head
        extra_options["attn_precision"] = self.attn_precision

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        n = self.norm1(x)
        if self.disable_self_attn:
            context_attn1 = context
        else:
            context_attn1 = None
        value_attn1 = None

        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            if context_attn1 is None:
                context_attn1 = n
            value_attn1 = context_attn1
            for p in patch:
                n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        if block_attn1 in attn1_replace_patch:
            if context_attn1 is None:
                context_attn1 = n
                value_attn1 = n
            n = self.attn1.to_q(n)
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)
        else:
            n = self.attn1(n, context=context_attn1, value=value_attn1, transformer_options=transformer_options)

        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x = n + x
        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)

        if self.attn2 is not None:
            n = self.norm2(x)
            if self.switch_temporal_ca_to_sa:
                context_attn2 = n
            else:
                context_attn2 = context
            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for p in patch:
                    n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block

            if block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                n = self.attn2.to_q(n)
                context_attn2 = self.attn2.to_k(context_attn2)
                value_attn2 = self.attn2.to_v(value_attn2)
                n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
                n = self.attn2.to_out(n)
            else:
                n = self.attn2(n, context=context_attn2, value=value_attn2, transformer_options=transformer_options)

        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x = n + x
        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x = x_skip + x

        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, attn_precision=None, dtype=None, device=None, operations=ops):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = operations.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)
        if not use_linear:
            self.proj_in = operations.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0, dtype=dtype, device=device)
        else:
            self.proj_in = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, attn_precision=attn_precision, dtype=dtype, device=device, operations=operations)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = operations.Conv2d(inner_dim, in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0, dtype=dtype, device=device)
        else:
            self.proj_out = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}):
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        transformer_options["activations_shape"] = list(x.shape)
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.movedim(1, 3).flatten(1, 2).contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = x.reshape(x.shape[0], h, w, x.shape[-1]).movedim(3, 1).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
