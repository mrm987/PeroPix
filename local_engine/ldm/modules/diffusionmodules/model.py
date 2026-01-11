"""
ComfyUI VAE 모델 포팅

원본: ComfyUI/comfy/ldm/modules/diffusionmodules/model.py
VAE Encoder/Decoder 핵심 모듈
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from ....ops import disable_weight_init as ops


def nonlinearity(x):
    """swish activation"""
    return F.silu(x)


def Normalize(in_channels, num_groups=32):
    return ops.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, conv_op=ops.Conv2d, scale_factor=2.0):
        super().__init__()
        self.with_conv = with_conv
        self.scale_factor = scale_factor

        if self.with_conv:
            self.conv = conv_op(in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)

    def forward(self, x):
        scale_factor = self.scale_factor
        if isinstance(scale_factor, (int, float)):
            scale_factor = (scale_factor,) * (x.ndim - 2)

        x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride=2, conv_op=ops.Conv2d):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = conv_op(in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=stride,
                                padding=0)

    def forward(self, x):
        if self.with_conv:
            if x.ndim == 4:
                pad = (0, 1, 0, 1)
                mode = "constant"
                x = F.pad(x, pad, mode=mode, value=0)
            elif x.ndim == 5:
                pad = (1, 1, 1, 1, 2, 0)
                mode = "replicate"
                x = F.pad(x, pad, mode=mode)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0, temb_channels=512, conv_op=ops.Conv2d, norm_op=Normalize):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.swish = nn.SiLU(inplace=True)
        self.norm1 = norm_op(in_channels)
        self.conv1 = conv_op(in_channels,
                             out_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        if temb_channels > 0:
            self.temb_proj = ops.Linear(temb_channels, out_channels)
        self.norm2 = norm_op(out_channels)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.conv2 = conv_op(out_channels,
                             out_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_op(in_channels,
                                             out_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1)
            else:
                self.nin_shortcut = conv_op(in_channels,
                                            out_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.swish(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


def pytorch_attention(q, k, v):
    """PyTorch SDPA를 사용한 VAE attention"""
    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    out = out.transpose(2, 3).reshape(orig_shape)
    return out


class AttnBlock(nn.Module):
    def __init__(self, in_channels, conv_op=ops.Conv2d, norm_op=Normalize):
        super().__init__()
        self.in_channels = in_channels

        self.norm = norm_op(in_channels)
        self.q = conv_op(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = conv_op(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = conv_op(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = conv_op(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        h_ = pytorch_attention(q, k, v)

        h_ = self.proj_out(h_)
        return x + h_


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None, conv_op=ops.Conv2d):
    return AttnBlock(in_channels, conv_op=conv_op)


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 conv3d=False, time_compress=None,
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        conv_op = ops.Conv2d

        # downsampling
        self.conv_in = conv_op(in_channels,
                               self.ch,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         conv_op=conv_op))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, conv_op=conv_op))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                stride = 2
                down.downsample = Downsample(block_in, resamp_with_conv, stride=stride, conv_op=conv_op)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       conv_op=conv_op)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, conv_op=conv_op)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       conv_op=conv_op)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_op(block_in,
                                2 * z_channels if double_z else z_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)

    def forward(self, x):
        temb = None
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 conv_out_op=None,
                 resnet_op=ResnetBlock,
                 attn_op=AttnBlock,
                 conv3d=False,
                 time_compress=None,
                 **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        conv_op = ops.Conv2d
        if conv_out_op is None:
            conv_out_op = ops.Conv2d

        # compute block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logging.debug("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, torch.prod(torch.tensor(self.z_shape)).item()))

        # z to block_in
        self.conv_in = conv_op(z_channels,
                               block_in,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = resnet_op(in_channels=block_in,
                                     out_channels=block_in,
                                     temb_channels=self.temb_ch,
                                     dropout=dropout,
                                     conv_op=conv_op)
        self.mid.attn_1 = attn_op(block_in, conv_op=conv_op)
        self.mid.block_2 = resnet_op(in_channels=block_in,
                                     out_channels=block_in,
                                     temb_channels=self.temb_ch,
                                     dropout=dropout,
                                     conv_op=conv_op)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(resnet_op(in_channels=block_in,
                                       out_channels=block_out,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       conv_op=conv_op))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(attn_op(block_in, conv_op=conv_op))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                scale_factor = 2.0
                up.upsample = Upsample(block_in, resamp_with_conv, conv_op=conv_op, scale_factor=scale_factor)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_out_op(block_in,
                                    out_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

    def forward(self, z, **kwargs):
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
