"""
LoRA/LyCORIS 로더

ComfyUI 방식의 LoRA 적용
기존 backend.py의 LyCORISLoader를 기반으로 구현
"""

import torch
import torch.nn as nn
import logging
import re
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
from safetensors.torch import load_file


def detect_lora_type(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    LoRA 파일의 타입 감지

    Returns:
        'lokr': LokR (Kronecker product)
        'loha': LoHa (Hadamard product)
        'lora': 일반 LoRA
    """
    for key in state_dict.keys():
        # LokR 감지
        if '.lokr_w1' in key or '.lokr_w2' in key:
            return 'lokr'
        # LoHa 감지
        if '.hada_w1_a' in key or '.hada_w2_a' in key:
            return 'loha'
    return 'lora'


def calculate_lora_weight(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    device: str,
    dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """
    일반 LoRA weight delta 계산

    delta = (up @ down) * (alpha / rank)
    """
    lora_down_key = f"{prefix}.lora_down.weight"
    lora_up_key = f"{prefix}.lora_up.weight"
    alpha_key = f"{prefix}.alpha"

    if lora_down_key not in state_dict or lora_up_key not in state_dict:
        return None

    lora_down = state_dict[lora_down_key].to(device=device, dtype=dtype)
    lora_up = state_dict[lora_up_key].to(device=device, dtype=dtype)

    rank = lora_down.shape[0]
    alpha = state_dict.get(alpha_key, torch.tensor(rank)).item()

    scale = alpha / rank

    # Conv2d의 경우 4D
    if len(lora_down.shape) == 4:
        delta = torch.einsum('oihw,kihw->okhw', lora_up, lora_down) * scale
    else:
        delta = (lora_up @ lora_down) * scale

    return delta


def calculate_lokr_weight(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    device: str,
    dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """
    LokR weight delta 계산

    delta = kron(w1, w2) * (alpha / dim)
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

    # w1 계산
    if w1_key in state_dict:
        w1 = state_dict[w1_key].to(device=device, dtype=dtype)
        dim = w1.shape[0]
    elif w1_a_key in state_dict and w1_b_key in state_dict:
        w1_a = state_dict[w1_a_key].to(device=device, dtype=dtype)
        w1_b = state_dict[w1_b_key].to(device=device, dtype=dtype)
        w1 = torch.mm(w1_a, w1_b)
        dim = w1_b.shape[0]
    else:
        return None

    # w2 계산
    if w2_key in state_dict:
        w2 = state_dict[w2_key].to(device=device, dtype=dtype)
    elif w2_a_key in state_dict and w2_b_key in state_dict:
        w2_a = state_dict[w2_a_key].to(device=device, dtype=dtype)
        w2_b = state_dict[w2_b_key].to(device=device, dtype=dtype)

        if t2_key in state_dict:
            t2 = state_dict[t2_key].to(device=device, dtype=dtype)
            w2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2_b, w2_a)
        else:
            w2 = torch.mm(w2_a, w2_b)
        dim = w2_b.shape[0]
    else:
        return None

    # Conv2d의 경우 4D 처리
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)

    scale = alpha / dim if dim > 0 else 1.0
    delta = torch.kron(w1, w2) * scale

    return delta


def calculate_loha_weight(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    device: str,
    dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """
    LoHa weight delta 계산

    delta = (w1_a @ w1_b) * (w2_a @ w2_b) * (alpha / dim)
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

    w1_a = state_dict[w1_a_key].to(device=device, dtype=dtype)
    w1_b = state_dict[w1_b_key].to(device=device, dtype=dtype)
    w2_a = state_dict[w2_a_key].to(device=device, dtype=dtype)
    w2_b = state_dict[w2_b_key].to(device=device, dtype=dtype)

    dim = w1_b.shape[0]

    if t1_key in state_dict and t2_key in state_dict:
        t1 = state_dict[t1_key].to(device=device, dtype=dtype)
        t2 = state_dict[t2_key].to(device=device, dtype=dtype)
        m1 = torch.einsum('i j k l, j r, i p -> p r k l', t1, w1_b, w1_a)
        m2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2_b, w2_a)
    else:
        m1 = torch.mm(w1_a, w1_b)
        m2 = torch.mm(w2_a, w2_b)

    scale = alpha / dim if dim > 0 else 1.0
    delta = (m1 * m2) * scale

    return delta


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


def lora_key_to_model_key(lora_key: str) -> str:
    """
    LoRA 키를 모델 레이어 키로 변환

    예: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q
        -> model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q
    """
    key = lora_key

    # UNet 변환
    if key.startswith('lora_unet_'):
        key = key[10:]  # 'lora_unet_' 제거

        # 언더스코어를 점으로 변환하되, 숫자 앞의 언더스코어는 유지
        # down_blocks_0 -> input_blocks.1
        # mid_block -> middle_block
        # up_blocks_0 -> output_blocks.0

        # 일단 모든 언더스코어를 점으로
        key = key.replace('_', '.')

        # 숫자 패턴 수정: blocks.0.attentions.0 -> blocks[0].attentions[0]
        # 하지만 ComfyUI 스타일은 input_blocks.1.1 형태 유지

    # CLIP 변환
    elif key.startswith('lora_te1_') or key.startswith('lora_te_'):
        prefix_len = 9 if key.startswith('lora_te1_') else 8
        key = 'clip_l.transformer.text_model.' + key[prefix_len:].replace('_', '.')

    elif key.startswith('lora_te2_'):
        key = 'clip_g.transformer.text_model.' + key[9:].replace('_', '.')

    return key


class LoRALoader:
    """LoRA/LyCORIS 로더 및 적용기"""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self._loaded_loras: Dict[str, Dict] = {}
        self._original_weights: Dict[str, torch.Tensor] = {}

    def load_lora(self, path: str) -> Dict[str, torch.Tensor]:
        """LoRA 파일 로드"""
        path = str(Path(path).resolve())

        if path.endswith('.safetensors'):
            return load_file(path)
        else:
            return torch.load(path, map_location='cpu')

    def calculate_deltas(
        self,
        lora_sd: Dict[str, torch.Tensor],
        scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        LoRA state_dict에서 weight delta 계산

        Returns:
            {model_key: delta_tensor}
        """
        lora_type = detect_lora_type(lora_sd)
        prefixes = get_lora_prefixes(lora_sd)
        deltas = {}

        for prefix in prefixes:
            if lora_type == 'lokr':
                delta = calculate_lokr_weight(lora_sd, prefix, self.device, self.dtype)
            elif lora_type == 'loha':
                delta = calculate_loha_weight(lora_sd, prefix, self.device, self.dtype)
            else:
                delta = calculate_lora_weight(lora_sd, prefix, self.device, self.dtype)

            if delta is not None:
                model_key = lora_key_to_model_key(prefix)
                deltas[model_key] = delta * scale

        return deltas

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

        try:
            lora_sd = self.load_lora(lora_path)
            deltas = self.calculate_deltas(lora_sd, scale)

            applied_count = 0

            for name, param in model.named_parameters():
                # LoRA 키와 매칭되는지 확인
                for delta_key, delta in deltas.items():
                    if self._key_matches(name, delta_key):
                        # 원본 백업
                        if name not in self._original_weights:
                            self._original_weights[name] = param.data.clone()

                        # delta 적용
                        if delta.shape == param.data.shape:
                            param.data += delta.to(param.device, param.dtype)
                            applied_count += 1
                        else:
                            logging.warning(f"Shape mismatch for {name}: {param.shape} vs {delta.shape}")

            self._loaded_loras[lora_name] = {
                'path': lora_path,
                'scale': scale,
                'applied_count': applied_count
            }

            logging.info(f"Applied LoRA '{lora_name}' with scale {scale}, {applied_count} layers")
            return True

        except Exception as e:
            logging.error(f"Failed to apply LoRA: {e}")
            return False

    def remove_lora(self, model: nn.Module, lora_name: Optional[str] = None):
        """
        LoRA 제거 (원본 가중치 복원)

        Args:
            model: 타겟 모델
            lora_name: 제거할 LoRA 이름 (None이면 전체 제거)
        """
        for name, param in model.named_parameters():
            if name in self._original_weights:
                param.data = self._original_weights[name].clone()

        if lora_name:
            if lora_name in self._loaded_loras:
                del self._loaded_loras[lora_name]
        else:
            self._loaded_loras.clear()
            self._original_weights.clear()

    def _key_matches(self, model_key: str, lora_key: str) -> bool:
        """모델 키와 LoRA 키가 매칭되는지 확인"""
        # 간단한 매칭: 키 끝부분 비교
        model_parts = model_key.split('.')
        lora_parts = lora_key.split('.')

        # 마지막 몇 개 부분만 비교
        for i in range(1, min(len(model_parts), len(lora_parts)) + 1):
            if model_parts[-i] != lora_parts[-i]:
                return False
        return True


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
