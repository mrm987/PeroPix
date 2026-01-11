"""
ComfyUI AutoencoderKL 포팅

원본: ComfyUI/comfy/ldm/models/autoencoder.py
SDXL VAE 모델
"""

import logging
import math
import torch
import torch.nn as nn
from typing import Tuple, Union

from ..modules.distributions.distributions import DiagonalGaussianDistribution
from ..modules.diffusionmodules.model import Encoder, Decoder
from ...ops import disable_weight_init as ops


class DiagonalGaussianRegularizer(nn.Module):
    def __init__(self, sample: bool = False):
        super().__init__()
        self.sample = sample

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        return z, None


class EmptyRegularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return z, None


class AutoencoderKL(nn.Module):
    """
    SDXL VAE (AutoencoderKL)

    ComfyUI 호환 구현으로, 직접 인스턴스화 가능
    """
    def __init__(
        self,
        embed_dim: int = 4,
        ddconfig: dict = None,
        sample: bool = False,
        **kwargs
    ):
        super().__init__()

        # 기본 SDXL VAE 설정
        if ddconfig is None:
            ddconfig = {
                "double_z": True,
                "z_channels": 4,
                "resolution": 256,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 128,
                "ch_mult": [1, 2, 4, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0,
            }

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.regularization = DiagonalGaussianRegularizer(sample=sample)

        # quant_conv / post_quant_conv
        conv_op = ops.Conv2d
        self.quant_conv = conv_op(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],
            (1 + ddconfig["double_z"]) * embed_dim,
            1,
        )
        self.post_quant_conv = conv_op(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        이미지를 latent로 인코딩

        Args:
            x: [B, 3, H, W] 이미지 텐서 (범위: [-1, 1])

        Returns:
            [B, 4, H//8, W//8] latent 텐서
        """
        z = self.encoder(x)
        z = self.quant_conv(z)
        z, _ = self.regularization(z)
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Latent를 이미지로 디코딩

        Args:
            z: [B, 4, H//8, W//8] latent 텐서

        Returns:
            [B, 3, H, W] 이미지 텐서 (범위: [-1, 1])
        """
        dec = self.post_quant_conv(z)
        dec = self.decoder(dec, **kwargs)
        return dec

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        dec = self.decode(z, **kwargs)
        return z, dec

    def get_last_layer(self):
        return self.decoder.conv_out.weight


# SDXL VAE 설정
SDXL_VAE_CONFIG = {
    "embed_dim": 4,
    "ddconfig": {
        "double_z": True,
        "z_channels": 4,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
    }
}


def create_sdxl_vae() -> AutoencoderKL:
    """SDXL VAE 인스턴스 생성"""
    return AutoencoderKL(**SDXL_VAE_CONFIG)
