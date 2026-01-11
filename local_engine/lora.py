"""
LoRA/LyCORIS 로더

ComfyUI 방식의 LoRA 적용
- SDXL UNet 키 매핑 테이블 기반
- lora_unet_xxx 형식의 LoRA 키를 ComfyUI 모델 키로 변환
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, List
from pathlib import Path
from safetensors.torch import load_file


# ============================================================
# SDXL UNet Key Mapping (ComfyUI utils.py에서 포팅)
# ============================================================

# Transformer block 내부 레이어들
TRANSFORMER_BLOCKS = {
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
}

# Attention 레이어들
UNET_MAP_ATTENTIONS = {
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
}

# ResNet 레이어 매핑 (Diffusers -> ComfyUI)
UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}

# 기본 레이어 매핑 (ComfyUI -> Diffusers)
UNET_MAP_BASIC = {
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias")
}

# CLIP 레이어 매핑
LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def unet_to_diffusers(unet_config: dict) -> Dict[str, str]:
    """
    Diffusers LoRA 키 -> ComfyUI UNet 키 매핑 테이블 생성

    ComfyUI comfy/utils.py:unet_to_diffusers 포팅

    Returns:
        {diffusers_key: comfyui_key} 매핑 딕셔너리
    """
    if "num_res_blocks" not in unet_config:
        return {}

    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)

    transformers_mid = unet_config.get("transformer_depth_middle", None)

    diffusers_unet_map = {}

    # Down blocks (input_blocks)
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            # ResNet 레이어
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)

            # Transformer 레이어
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, b)] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1

        # Downsampler
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = "input_blocks.{}.0.op.{}".format(n, k)

    # Mid block (middle_block)
    i = 0
    for b in UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = "middle_block.1.{}".format(b)

    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)

    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)

    # Up blocks (output_blocks)
    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            if i == l - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map["up_blocks.{}.upsamplers.0.conv.{}".format(x, k)] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1

    # Basic layers
    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]

    return diffusers_unet_map


# SDXL UNet config
SDXL_UNET_CONFIG = {
    "num_res_blocks": [2, 2, 2],
    "channel_mult": [1, 2, 4],
    "transformer_depth": [0, 0, 2, 2, 10, 10],
    "transformer_depth_middle": 10,
    "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
}


def build_lora_key_map(model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, str]:
    """
    LoRA 키 -> 모델 키 매핑 테이블 구축

    ComfyUI comfy/lora.py:model_lora_keys_unet + model_lora_keys_clip 포팅

    Args:
        model_state_dict: 모델의 state_dict

    Returns:
        {lora_key: model_key} 매핑 딕셔너리
    """
    key_map = {}
    sdk = set(model_state_dict.keys())

    # ============================================================
    # UNet 키 매핑
    # ============================================================

    # 1. 직접 매핑 (접두사 없는 ComfyUI 스타일: input_blocks.xxx)
    # PeroPix UNet은 diffusion_model. 접두사 없이 직접 input_blocks/middle_block/output_blocks 사용
    for k in sdk:
        if k.endswith(".weight"):
            # input_blocks, middle_block, output_blocks로 시작하는 키들
            if k.startswith("input_blocks.") or k.startswith("middle_block.") or k.startswith("output_blocks."):
                key_lora = k[:-len(".weight")].replace(".", "_")
                key_map["lora_unet_{}".format(key_lora)] = k
                # Generic 형식
                key_map["{}".format(k[:-len(".weight")])] = k
            # 기타 키들 (time_embed, label_emb, out 등)
            elif k.startswith("time_embed.") or k.startswith("label_emb.") or k.startswith("out."):
                key_lora = k[:-len(".weight")].replace(".", "_")
                key_map["lora_unet_{}".format(key_lora)] = k

    # 2. ComfyUI 네이티브 형식: diffusion_model.xxx -> lora_unet_xxx
    for k in sdk:
        if k.startswith("diffusion_model.") or k.startswith("model.diffusion_model."):
            # diffusion_model 접두사 제거
            if k.startswith("model.diffusion_model."):
                base_key = k[len("model.diffusion_model."):]
            else:
                base_key = k[len("diffusion_model."):]

            if k.endswith(".weight"):
                key_lora = base_key[:-len(".weight")].replace(".", "_")
                key_map["lora_unet_{}".format(key_lora)] = k
                # Generic 형식도 지원
                key_map["{}".format(k[:-len(".weight")])] = k

    # 3. Diffusers 형식 지원 (unet_to_diffusers 매핑 사용)
    diffusers_keys = unet_to_diffusers(SDXL_UNET_CONFIG)
    for diffusers_key, comfyui_key in diffusers_keys.items():
        # 먼저 접두사 없는 형식 시도 (PeroPix UNet)
        unet_key = comfyui_key
        if unet_key not in sdk:
            # ComfyUI 모델은 diffusion_model. 접두사 사용
            unet_key = "diffusion_model.{}".format(comfyui_key)
        if unet_key not in sdk:
            unet_key = "model.diffusion_model.{}".format(comfyui_key)

        if unet_key in sdk:
            if diffusers_key.endswith(".weight"):
                key_lora = diffusers_key[:-len(".weight")].replace(".", "_")
                key_map["lora_unet_{}".format(key_lora)] = unet_key

                # Diffusers lora 형식들
                for prefix in ["", "unet."]:
                    diffusers_lora_key = "{}{}".format(prefix, diffusers_key[:-len(".weight")].replace(".to_", ".processor.to_"))
                    if diffusers_lora_key.endswith(".to_out.0"):
                        diffusers_lora_key = diffusers_lora_key[:-2]
                    key_map[diffusers_lora_key] = unet_key

    # ============================================================
    # CLIP 키 매핑
    # ============================================================

    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    clip_l_present = False
    clip_g_present = False

    for b in range(32):
        for c, lora_c in LORA_CLIP_MAP.items():
            # CLIP-L
            k = "clip_l.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                clip_l_present = True
                key_map[text_model_lora_key.format(b, lora_c)] = k
                key_map["lora_te1_text_model_encoder_layers_{}_{}".format(b, lora_c)] = k
                key_map["text_encoder.text_model.encoder.layers.{}.{}".format(b, c)] = k

            # CLIP-G
            k = "clip_g.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                clip_g_present = True
                if clip_l_present:
                    key_map["lora_te2_text_model_encoder_layers_{}_{}".format(b, lora_c)] = k
                    key_map["text_encoder_2.text_model.encoder.layers.{}.{}".format(b, c)] = k
                else:
                    key_map["lora_te_text_model_encoder_layers_{}_{}".format(b, lora_c)] = k
                    key_map["text_encoder.text_model.encoder.layers.{}.{}".format(b, c)] = k

    # Text projection
    k = "clip_g.transformer.text_projection.weight"
    if k in sdk:
        key_map["lora_te2_text_projection"] = k

    k = "clip_l.transformer.text_projection.weight"
    if k in sdk:
        key_map["lora_te1_text_projection"] = k

    return key_map


# ============================================================
# LoRA Weight Calculation
# ============================================================

def detect_lora_type(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    LoRA 파일의 타입 감지

    Returns:
        'lokr': LokR (Kronecker product)
        'loha': LoHa (Hadamard product)
        'lora': 일반 LoRA
    """
    for key in state_dict.keys():
        if '.lokr_w1' in key or '.lokr_w2' in key:
            return 'lokr'
        if '.hada_w1_a' in key or '.hada_w2_a' in key:
            return 'loha'
    return 'lora'


def get_lora_prefixes(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """state_dict에서 고유한 LoRA prefix들을 추출"""
    prefixes = set()

    suffixes = [
        '.lora_down.weight', '.lora_up.weight',
        '.lokr_w1', '.lokr_w2', '.lokr_w1_a', '.lokr_w1_b',
        '.lokr_w2_a', '.lokr_w2_b', '.lokr_t2',
        '.hada_w1_a', '.hada_w1_b', '.hada_w2_a', '.hada_w2_b',
        '.hada_t1', '.hada_t2', '.alpha'
    ]

    for key in state_dict.keys():
        for suffix in suffixes:
            if key.endswith(suffix):
                prefixes.add(key[:-len(suffix)])
                break

    return list(prefixes)


def calculate_lora_weight(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    device: str,
    dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """
    일반 LoRA weight delta 계산: delta = (up @ down) * (alpha / rank)

    ComfyUI 방식: float32로 계산 후 최종 dtype으로 변환
    """
    lora_down_key = f"{prefix}.lora_down.weight"
    lora_up_key = f"{prefix}.lora_up.weight"
    alpha_key = f"{prefix}.alpha"

    if lora_down_key not in state_dict or lora_up_key not in state_dict:
        return None

    # ComfyUI 방식: float32로 계산 (정밀도 유지)
    calc_dtype = torch.float32
    lora_down = state_dict[lora_down_key].to(device=device, dtype=calc_dtype)
    lora_up = state_dict[lora_up_key].to(device=device, dtype=calc_dtype)

    rank = lora_down.shape[0]
    alpha = state_dict.get(alpha_key, torch.tensor(rank)).item()
    scale = alpha / rank

    # ComfyUI 방식: 2D로 펼쳐서 matmul 후 reshape
    # lora_down: (rank, in_features) 또는 (rank, in_ch, kh, kw)
    # lora_up: (out_features, rank) 또는 (out_ch, rank, 1, 1)
    if len(lora_down.shape) == 4:
        # Conv2d: 2D로 펼침
        down_flat = lora_down.flatten(1)  # (rank, in_ch*kh*kw)
        up_flat = lora_up.flatten(1)      # (out_ch, rank*1*1) = (out_ch, rank)
        delta = (up_flat @ down_flat) * scale  # (out_ch, in_ch*kh*kw)
    else:
        delta = (lora_up @ lora_down) * scale

    # 최종 dtype으로 변환
    return delta.to(dtype=dtype)


def calculate_lokr_weight(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    device: str,
    dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """
    LokR weight delta 계산: delta = kron(w1, w2) * (alpha / dim)

    ComfyUI 방식: intermediate_dtype=float32로 계산 후 변환
    """
    w1_key = f"{prefix}.lokr_w1"
    w2_key = f"{prefix}.lokr_w2"
    w1_a_key = f"{prefix}.lokr_w1_a"
    w1_b_key = f"{prefix}.lokr_w1_b"
    w2_a_key = f"{prefix}.lokr_w2_a"
    w2_b_key = f"{prefix}.lokr_w2_b"
    t2_key = f"{prefix}.lokr_t2"
    alpha_key = f"{prefix}.alpha"

    alpha = state_dict.get(alpha_key, torch.tensor(1.0)).item() if alpha_key in state_dict else 1.0
    dim = None

    # ComfyUI 방식: float32로 연산 (FP16에서 overflow 방지)
    calc_dtype = torch.float32

    # w1 계산
    if w1_key in state_dict:
        w1 = state_dict[w1_key].to(device=device, dtype=calc_dtype)
    elif w1_a_key in state_dict and w1_b_key in state_dict:
        w1_a = state_dict[w1_a_key].to(device=device, dtype=calc_dtype)
        w1_b = state_dict[w1_b_key].to(device=device, dtype=calc_dtype)
        w1 = torch.mm(w1_a, w1_b)
        dim = w1_b.shape[0]
    else:
        return None

    # w2 계산
    if w2_key in state_dict:
        w2 = state_dict[w2_key].to(device=device, dtype=calc_dtype)
    elif w2_a_key in state_dict and w2_b_key in state_dict:
        w2_a = state_dict[w2_a_key].to(device=device, dtype=calc_dtype)
        w2_b = state_dict[w2_b_key].to(device=device, dtype=calc_dtype)

        if t2_key in state_dict:
            t2 = state_dict[t2_key].to(device=device, dtype=calc_dtype)
            w2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2_b, w2_a)
        else:
            w2 = torch.mm(w2_a, w2_b)
        dim = w2_b.shape[0]
    else:
        return None

    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)

    # ComfyUI 방식: alpha / dim 스케일
    if alpha is not None and dim is not None and dim > 0:
        scale = alpha / dim
    else:
        scale = 1.0

    delta = torch.kron(w1, w2) * scale

    # 최종 dtype으로 변환
    return delta.to(dtype=dtype)


def calculate_loha_weight(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    device: str,
    dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """
    LoHa weight delta 계산: delta = (w1_a @ w1_b) * (w2_a @ w2_b) * (alpha / dim)

    ComfyUI 방식: float32로 계산 후 변환
    """
    w1_a_key = f"{prefix}.hada_w1_a"
    w1_b_key = f"{prefix}.hada_w1_b"
    w2_a_key = f"{prefix}.hada_w2_a"
    w2_b_key = f"{prefix}.hada_w2_b"
    t1_key = f"{prefix}.hada_t1"
    t2_key = f"{prefix}.hada_t2"
    alpha_key = f"{prefix}.alpha"

    if w1_a_key not in state_dict or w1_b_key not in state_dict:
        return None
    if w2_a_key not in state_dict or w2_b_key not in state_dict:
        return None

    alpha = state_dict.get(alpha_key, torch.tensor(1.0)).item() if alpha_key in state_dict else 1.0

    # ComfyUI 방식: float32로 연산
    calc_dtype = torch.float32

    w1_a = state_dict[w1_a_key].to(device=device, dtype=calc_dtype)
    w1_b = state_dict[w1_b_key].to(device=device, dtype=calc_dtype)
    w2_a = state_dict[w2_a_key].to(device=device, dtype=calc_dtype)
    w2_b = state_dict[w2_b_key].to(device=device, dtype=calc_dtype)

    dim = w1_b.shape[0]

    if t1_key in state_dict and t2_key in state_dict:
        t1 = state_dict[t1_key].to(device=device, dtype=calc_dtype)
        t2 = state_dict[t2_key].to(device=device, dtype=calc_dtype)
        m1 = torch.einsum('i j k l, j r, i p -> p r k l', t1, w1_b, w1_a)
        m2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2_b, w2_a)
    else:
        m1 = torch.mm(w1_a, w1_b)
        m2 = torch.mm(w2_a, w2_b)

    scale = alpha / dim if dim > 0 else 1.0
    delta = (m1 * m2) * scale

    # 최종 dtype으로 변환
    return delta.to(dtype=dtype)


# ============================================================
# LoRA Loader Class
# ============================================================

class LoRALoader:
    """LoRA/LyCORIS 로더 및 적용기

    VRAM 최적화: delta를 저장하지 않고, 제거 시 LoRA 파일에서 다시 계산하여 빼기
    (ComfyUI 방식과 유사)
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self._loaded_loras: Dict[str, Dict] = {}  # {name: {path, scale, applied_count}}
        self._key_map: Dict[str, str] = {}

    def load_lora(self, path: str) -> Dict[str, torch.Tensor]:
        """LoRA 파일 로드"""
        path = str(Path(path).resolve())

        if path.endswith('.safetensors'):
            return load_file(path)
        else:
            return torch.load(path, map_location='cpu')

    def build_key_map(self, model: nn.Module):
        """모델의 state_dict를 기반으로 키 매핑 테이블 구축"""
        self._key_map = build_lora_key_map(model.state_dict())
        print(f"[LoRA] Built key map with {len(self._key_map)} entries")

    def apply_lora_to_model(
        self,
        model: nn.Module,
        lora_path: str,
        scale: float = 1.0,
        lora_name: Optional[str] = None
    ) -> bool:
        """
        LoRA를 모델에 적용

        Args:
            model: 타겟 모델 (UNet 또는 CLIP)
            lora_path: LoRA 파일 경로
            scale: LoRA 강도
            lora_name: LoRA 식별 이름

        Returns:
            성공 여부
        """
        if lora_name is None:
            lora_name = Path(lora_path).stem

        # 키 맵이 없으면 구축
        if not self._key_map:
            self.build_key_map(model)

        try:
            lora_sd = self.load_lora(lora_path)
            lora_type = detect_lora_type(lora_sd)
            prefixes = get_lora_prefixes(lora_sd)

            print(f"[LoRA] Loading '{lora_name}': type={lora_type}, prefixes={len(prefixes)}")

            applied_count = 0
            skipped_count = 0
            no_mapping_count = 0
            no_param_count = 0
            no_delta_count = 0
            model_sd = model.state_dict()

            # 모델 파라미터 이름 -> 파라미터 매핑 (효율성)
            param_dict = {name: p for name, p in model.named_parameters()}

            for prefix in prefixes:
                # Delta 계산
                if lora_type == 'lokr':
                    delta = calculate_lokr_weight(lora_sd, prefix, self.device, self.dtype)
                elif lora_type == 'loha':
                    delta = calculate_loha_weight(lora_sd, prefix, self.device, self.dtype)
                else:
                    delta = calculate_lora_weight(lora_sd, prefix, self.device, self.dtype)

                if delta is None:
                    no_delta_count += 1
                    continue

                # LoRA prefix -> 모델 키 변환
                model_key = self._key_map.get(prefix)

                if model_key is None:
                    # 키 맵에 없으면 스킵
                    no_mapping_count += 1
                    skipped_count += 1
                    continue

                # 모델에서 해당 파라미터 찾기
                param = param_dict.get(model_key)

                if param is None:
                    no_param_count += 1
                    skipped_count += 1
                    continue

                # Shape 맞추기 (LokR/LoHa의 경우 reshape 필요)
                if delta.shape != param.data.shape:
                    try:
                        delta = delta.reshape(param.data.shape)
                    except RuntimeError:
                        logging.warning(f"Cannot reshape {model_key}: {delta.shape} -> {param.shape}")
                        skipped_count += 1
                        continue

                # delta 적용 (delta는 이미 alpha/rank 스케일 적용됨, 여기서 user scale 적용)
                scaled_delta = (delta * scale).to(param.device, param.dtype)
                param.data += scaled_delta
                applied_count += 1

                # delta 메모리 즉시 해제
                del scaled_delta

            self._loaded_loras[lora_name] = {
                'path': lora_path,
                'scale': scale,
                'applied_count': applied_count,
                'skipped_count': skipped_count
            }

            print(f"[LoRA] Applied '{lora_name}': {applied_count} layers applied, "
                  f"{skipped_count} skipped (no_mapping={no_mapping_count}, "
                  f"no_param={no_param_count}, no_delta={no_delta_count})")
            return applied_count > 0

        except Exception as e:
            logging.error(f"Failed to apply LoRA '{lora_name}': {e}")
            import traceback
            traceback.print_exc()
            return False

    def remove_lora(self, model: nn.Module, lora_name: Optional[str] = None):
        """
        LoRA 제거 (LoRA 파일에서 delta를 다시 계산하여 빼기)

        VRAM 최적화: delta를 저장하지 않고 필요시 재계산
        (약간의 CPU/디스크 오버헤드 대신 VRAM 절약)

        Args:
            model: 타겟 모델
            lora_name: 제거할 LoRA 이름 (None이면 전체 제거)
        """
        if not self._loaded_loras:
            return

        # 키 맵이 없으면 구축
        if not self._key_map:
            self.build_key_map(model)

        param_dict = {name: p for name, p in model.named_parameters()}

        # 제거할 LoRA 목록
        loras_to_remove = []
        if lora_name:
            if lora_name in self._loaded_loras:
                loras_to_remove.append((lora_name, self._loaded_loras[lora_name]))
        else:
            loras_to_remove = list(self._loaded_loras.items())

        # 각 LoRA에 대해 delta를 다시 계산하여 빼기
        for name, info in loras_to_remove:
            lora_path = info['path']
            scale = info['scale']

            try:
                lora_sd = self.load_lora(lora_path)
                lora_type = detect_lora_type(lora_sd)
                prefixes = get_lora_prefixes(lora_sd)

                for prefix in prefixes:
                    # Delta 계산
                    if lora_type == 'lokr':
                        delta = calculate_lokr_weight(lora_sd, prefix, self.device, self.dtype)
                    elif lora_type == 'loha':
                        delta = calculate_loha_weight(lora_sd, prefix, self.device, self.dtype)
                    else:
                        delta = calculate_lora_weight(lora_sd, prefix, self.device, self.dtype)

                    if delta is None:
                        continue

                    model_key = self._key_map.get(prefix)
                    if model_key is None:
                        continue

                    param = param_dict.get(model_key)
                    if param is None:
                        continue

                    # Shape 맞추기
                    if delta.shape != param.data.shape:
                        try:
                            delta = delta.reshape(param.data.shape)
                        except RuntimeError:
                            continue

                    # delta 빼기
                    scaled_delta = (delta * scale).to(param.device, param.dtype)
                    param.data -= scaled_delta
                    del scaled_delta, delta

                del lora_sd
                logging.info(f"Removed LoRA: {name}")

            except Exception as e:
                logging.error(f"Failed to remove LoRA '{name}': {e}")

        # 제거된 LoRA 정보 삭제
        if lora_name:
            if lora_name in self._loaded_loras:
                del self._loaded_loras[lora_name]
        else:
            self._loaded_loras.clear()

        # GPU 메모리 정리
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info(f"LoRA removal complete. Remaining: {list(self._loaded_loras.keys())}")


def apply_lora(
    model: nn.Module,
    lora_path: str,
    scale: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> LoRALoader:
    """
    LoRA 적용 편의 함수

    Args:
        model: 타겟 모델
        lora_path: LoRA 파일 경로
        scale: LoRA 강도
        device: 디바이스
        dtype: dtype

    Returns:
        LoRALoader 인스턴스 (제거 시 사용)
    """
    loader = LoRALoader(device=device, dtype=dtype)
    loader.apply_lora_to_model(model, lora_path, scale)
    return loader
