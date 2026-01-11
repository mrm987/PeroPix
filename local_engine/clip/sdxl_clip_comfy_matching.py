"""
SDXL CLIP 인코더

ComfyUI의 SDXL CLIP 구현을 단순화하여 포팅
SDXL은 CLIP-L (768d)과 CLIP-G (1280d)를 사용하여 2048d 임베딩 생성
"""

import torch
import torch.nn as nn
import logging
from transformers import CLIPTokenizer
from typing import List, Tuple, Optional, Union

from .clip_model import CLIPTextModel, CLIP_L_CONFIG, CLIP_G_CONFIG
from ..ops import manual_cast as ops


class SDXLClipL(nn.Module):
    """SDXL CLIP-L 모델 (768d, 12 layers, penultimate output)"""

    def __init__(self, device="cpu", dtype=None):
        super().__init__()
        config = CLIP_L_CONFIG.copy()
        self.transformer = CLIPTextModel(config, dtype, device, ops)
        self.special_tokens = {"start": 49406, "end": 49407, "pad": 49407}
        self.layer_idx = -2  # penultimate layer
        self.layer_norm_hidden_state = False

    def forward(self, tokens):
        """
        Args:
            tokens: [B, seq_len] 토큰 텐서

        Returns:
            (hidden_states, pooled_output)
        """
        device = next(self.transformer.parameters()).device
        tokens = tokens.to(device)

        outputs = self.transformer(
            tokens,
            intermediate_output=self.layer_idx,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
            dtype=torch.float32
        )

        # outputs: (last, intermediate, projected_pooled, pooled)
        hidden = outputs[1].float() if outputs[1] is not None else outputs[0].float()
        pooled = outputs[2].float() if outputs[2] is not None else None

        return hidden, pooled

    def load_sd(self, sd):
        return self.transformer.load_state_dict(sd, strict=False)


class SDXLClipG(nn.Module):
    """SDXL CLIP-G 모델 (1280d, 32 layers, penultimate output)"""

    def __init__(self, device="cpu", dtype=None):
        super().__init__()
        config = CLIP_G_CONFIG.copy()
        self.transformer = CLIPTextModel(config, dtype, device, ops)
        self.special_tokens = {"start": 49406, "end": 49407, "pad": 0}
        self.layer_idx = -2  # penultimate layer
        self.layer_norm_hidden_state = False

    def forward(self, tokens):
        """
        Args:
            tokens: [B, seq_len] 토큰 텐서

        Returns:
            (hidden_states, pooled_output)
        """
        device = next(self.transformer.parameters()).device
        tokens = tokens.to(device)

        outputs = self.transformer(
            tokens,
            intermediate_output=self.layer_idx,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
            dtype=torch.float32
        )

        hidden = outputs[1].float() if outputs[1] is not None else outputs[0].float()
        pooled = outputs[2].float() if outputs[2] is not None else None

        return hidden, pooled

    def load_sd(self, sd):
        return self.transformer.load_state_dict(sd, strict=False)


import re

def parse_prompt_weights(text: str) -> List[Tuple[str, float]]:
    """
    ComfyUI 방식 가중치 파싱

    지원 구문:
    - (text) -> weight * 1.1
    - (text:1.5) -> explicit weight
    - ((text)) -> weight * 1.1 * 1.1

    Returns:
        [(text_segment, weight), ...]
    """
    result = []

    # 괄호와 가중치를 처리하기 위한 스택 기반 파싱
    def parse_recursive(s: str, current_weight: float = 1.0) -> List[Tuple[str, float]]:
        segments = []
        i = 0
        buffer = ""

        while i < len(s):
            char = s[i]

            if char == '(':
                # 현재 버퍼를 저장
                if buffer.strip():
                    segments.append((buffer.strip(), current_weight))
                buffer = ""

                # 괄호 내용 찾기
                depth = 1
                start = i + 1
                i += 1
                while i < len(s) and depth > 0:
                    if s[i] == '(':
                        depth += 1
                    elif s[i] == ')':
                        depth -= 1
                    i += 1

                inner = s[start:i-1]

                # 가중치 추출 (text:1.5 형태)
                weight_match = re.match(r'^(.+):([0-9.]+)$', inner)
                if weight_match:
                    inner_text = weight_match.group(1)
                    weight_mult = float(weight_match.group(2))
                else:
                    inner_text = inner
                    weight_mult = 1.1  # 기본 강조

                # 재귀적으로 내부 파싱
                inner_segments = parse_recursive(inner_text, current_weight * weight_mult)
                segments.extend(inner_segments)

            elif char == '\\' and i + 1 < len(s):
                # 이스케이프 처리
                buffer += s[i + 1]
                i += 2
            else:
                buffer += char
                i += 1

        if buffer.strip():
            segments.append((buffer.strip(), current_weight))

        return segments

    # 닫히지 않은 괄호 처리를 위해 안전하게 파싱
    try:
        result = parse_recursive(text)
    except:
        # 파싱 실패 시 전체 텍스트를 weight 1.0으로
        result = [(text, 1.0)]

    # 빈 결과 처리
    if not result:
        result = [(text, 1.0)] if text else [("", 1.0)]

    return result


class SDXLTokenizer:
    """
    SDXL 듀얼 토크나이저

    ComfyUI 방식: 단어 경계를 존중하며 77토큰 청크로 분할
    """

    def __init__(self, tokenizer_path: Optional[str] = None):
        # HuggingFace CLIP tokenizer 사용
        if tokenizer_path:
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        self.max_length = 77
        self.start_token = 49406
        self.end_token = 49407
        self.pad_token_l = 49407
        self.pad_token_g = 0
        self.max_word_length = 8  # 이 이상의 토큰을 가진 단어는 분할 가능

    def tokenize_with_weights(self, text: str) -> List[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]]:
        """
        ComfyUI 방식: 가중치 파싱 + 단어별 토큰화 + 단어 경계 존중 청킹

        ComfyUI sd1_clip.py:573 방식을 정확히 재현

        Returns:
            [(chunk_l, chunk_g), ...] where each chunk is [(token, weight), ...]
        """
        # 가중치 파싱
        weighted_segments = parse_prompt_weights(text)

        # ComfyUI 방식: 단어별 토큰화 (sd1_clip.py:573)
        word_tokens = []  # [(tokens, weight), ...] where tokens = [token_id, ...]

        for segment_text, weight in weighted_segments:
            if not segment_text.strip():
                continue

            # ComfyUI: 공백으로 단어 분리 후 각 단어를 개별 토큰화
            # 토크나이저가 start/end 토큰을 추가하므로 add_special_tokens=False
            for word in segment_text.split():
                if not word:
                    continue
                encoded = self.tokenizer(
                    word,
                    truncation=False,
                    add_special_tokens=False
                )
                tokens = encoded["input_ids"]
                if tokens:
                    word_tokens.append((tokens, weight))

        # ComfyUI 방식 청킹: 단어 경계 존중
        chunks_l = []
        chunks_g = []

        current_chunk_l = [(self.start_token, 1.0)]
        current_chunk_g = [(self.start_token, 1.0)]

        max_content = self.max_length - 2  # start + end 제외

        for tokens, weight in word_tokens:
            is_large_word = len(tokens) >= self.max_word_length

            while len(tokens) > 0:
                current_len = len(current_chunk_l) - 1  # start 토큰 제외
                remaining = max_content - current_len

                if len(tokens) <= remaining:
                    # 현재 청크에 모두 들어감
                    for t in tokens:
                        current_chunk_l.append((t, weight))
                        current_chunk_g.append((t, weight))
                    tokens = []
                else:
                    # 청크 넘침
                    if is_large_word:
                        # 큰 단어는 분할
                        for t in tokens[:remaining]:
                            current_chunk_l.append((t, weight))
                            current_chunk_g.append((t, weight))
                        tokens = tokens[remaining:]
                    # else: 작은 단어는 다음 청크로 (분할 안함)

                    # 현재 청크 완성
                    current_chunk_l.append((self.end_token, 1.0))
                    current_chunk_g.append((self.end_token, 1.0))

                    # 패딩
                    while len(current_chunk_l) < self.max_length:
                        current_chunk_l.append((self.pad_token_l, 1.0))
                        current_chunk_g.append((self.pad_token_g, 1.0))

                    chunks_l.append(current_chunk_l)
                    chunks_g.append(current_chunk_g)

                    # 새 청크 시작
                    current_chunk_l = [(self.start_token, 1.0)]
                    current_chunk_g = [(self.start_token, 1.0)]

                    if not is_large_word:
                        # 작은 단어는 새 청크에 추가
                        for t in tokens:
                            current_chunk_l.append((t, weight))
                            current_chunk_g.append((t, weight))
                        tokens = []

        # 마지막 청크 완성
        if len(current_chunk_l) > 1:  # start 토큰만 있는 게 아니라면
            current_chunk_l.append((self.end_token, 1.0))
            current_chunk_g.append((self.end_token, 1.0))

            while len(current_chunk_l) < self.max_length:
                current_chunk_l.append((self.pad_token_l, 1.0))
                current_chunk_g.append((self.pad_token_g, 1.0))

            chunks_l.append(current_chunk_l)
            chunks_g.append(current_chunk_g)

        # 빈 입력 처리
        if not chunks_l:
            empty_l = [(self.start_token, 1.0), (self.end_token, 1.0)]
            empty_g = [(self.start_token, 1.0), (self.end_token, 1.0)]
            while len(empty_l) < self.max_length:
                empty_l.append((self.pad_token_l, 1.0))
                empty_g.append((self.pad_token_g, 1.0))
            chunks_l.append(empty_l)
            chunks_g.append(empty_g)

        return list(zip(chunks_l, chunks_g))

    def tokenize(self, text: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        텍스트를 CLIP-L과 CLIP-G용 토큰으로 변환

        ComfyUI 방식: 단어 경계를 존중하며 77토큰 청크로 분할

        Returns:
            (tokens_l_list, tokens_g_list): 각각 청크별 [1, 77] 텐서 리스트
        """
        # 가중치 파싱 + 단어 경계 청킹
        weighted_chunks = self.tokenize_with_weights(text)

        chunks_l = []
        chunks_g = []

        for chunk_l, chunk_g in weighted_chunks:
            # (token, weight) 튜플에서 토큰만 추출
            tokens_l = [t for t, w in chunk_l]
            tokens_g = [t for t, w in chunk_g]

            chunks_l.append(torch.tensor([tokens_l], dtype=torch.long))
            chunks_g.append(torch.tensor([tokens_g], dtype=torch.long))

        return chunks_l, chunks_g


class SDXLClipModel(nn.Module):
    """
    SDXL 듀얼 CLIP 모델

    CLIP-L (768d)과 CLIP-G (1280d)를 결합하여 2048d 임베딩 생성

    ComfyUI 방식: 77토큰 초과 시 여러 청크로 처리 후 concat
    """

    def __init__(self, device="cpu", dtype=None):
        super().__init__()
        self.clip_l = SDXLClipL(device=device, dtype=dtype)
        self.clip_g = SDXLClipG(device=device, dtype=dtype)
        self.tokenizer = SDXLTokenizer()

    def tokenize(self, text: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """텍스트를 CLIP-L/G용 토큰 청크 리스트로 변환"""
        return self.tokenizer.tokenize(text)

    def encode_chunks(
        self,
        chunks_l: List[torch.Tensor],
        chunks_g: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        토큰 청크들을 CLIP 임베딩으로 인코딩

        ComfyUI 방식: 모든 청크를 배치로 한번에 처리 후 시퀀스로 concat

        Args:
            chunks_l: CLIP-L 토큰 청크 리스트 (각 [1, 77])
            chunks_g: CLIP-G 토큰 청크 리스트 (각 [1, 77])

        Returns:
            (cond, pooled): cond [1, total_seq_len, 2048], pooled [1, 1280]
        """
        num_chunks = len(chunks_l)

        # 모든 청크를 배치 차원으로 합침 [N, 77]
        batch_l = torch.cat(chunks_l, dim=0)  # [N, 77]
        batch_g = torch.cat(chunks_g, dim=0)  # [N, 77]

        # 한번에 인코딩
        l_hidden, l_pooled = self.clip_l(batch_l)  # [N, 77, 768]
        g_hidden, g_pooled = self.clip_g(batch_g)  # [N, 77, 1280]

        # 첫 번째 청크의 pooled만 사용
        first_pooled = g_pooled[0:1]  # [1, 1280]

        # 시퀀스 길이 맞추기
        min_len = min(l_hidden.shape[1], g_hidden.shape[1])
        l_hidden = l_hidden[:, :min_len]
        g_hidden = g_hidden[:, :min_len]

        # 배치를 시퀀스로 변환: [N, 77, dim] -> [1, N*77, dim]
        l_concat = l_hidden.reshape(1, -1, l_hidden.shape[-1])  # [1, N*77, 768]
        g_concat = g_hidden.reshape(1, -1, g_hidden.shape[-1])  # [1, N*77, 1280]

        # 결합: [1, N*77, 768] + [1, N*77, 1280] -> [1, N*77, 2048]
        cond = torch.cat([l_concat, g_concat], dim=-1)

        return cond, first_pooled

    def encode(self, tokens_l: torch.Tensor, tokens_g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        단일 토큰 텐서를 CLIP 임베딩으로 인코딩 (하위 호환)

        Args:
            tokens_l: CLIP-L 토큰 [B, seq_len]
            tokens_g: CLIP-G 토큰 [B, seq_len]

        Returns:
            (cond, pooled): cond [B, seq_len, 2048], pooled [B, 1280]
        """
        # CLIP-L 인코딩
        l_hidden, l_pooled = self.clip_l(tokens_l)

        # CLIP-G 인코딩
        g_hidden, g_pooled = self.clip_g(tokens_g)

        # 시퀀스 길이 맞추기
        min_len = min(l_hidden.shape[1], g_hidden.shape[1])
        l_hidden = l_hidden[:, :min_len]
        g_hidden = g_hidden[:, :min_len]

        # 결합: [B, seq_len, 768] + [B, seq_len, 1280] -> [B, seq_len, 2048]
        cond = torch.cat([l_hidden, g_hidden], dim=-1)

        return cond, g_pooled

    def encode_text(self, text: str, apply_weights: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        텍스트를 직접 CLIP 임베딩으로 인코딩

        ComfyUI 방식: 77토큰 초과 시 여러 청크로 처리, 가중치 적용

        Args:
            text: 입력 텍스트
            apply_weights: 가중치 적용 여부 (default: True)

        Returns:
            (cond, pooled): SDXL conditioning 텐서들
        """
        # 가중치 포함 토큰화
        weighted_chunks = self.tokenizer.tokenize_with_weights(text)
        num_chunks = len(weighted_chunks)

        if num_chunks > 1:
            logging.info(f"[CLIP] Long prompt: {num_chunks} chunks ({num_chunks * 77} tokens max)")

        # 토큰과 가중치 분리
        chunks_l = []
        chunks_g = []
        weights_list = []

        for chunk_l, chunk_g in weighted_chunks:
            tokens_l = [t for t, w in chunk_l]
            tokens_g = [t for t, w in chunk_g]
            weights = [w for t, w in chunk_l]

            chunks_l.append(torch.tensor([tokens_l], dtype=torch.long))
            chunks_g.append(torch.tensor([tokens_g], dtype=torch.long))
            weights_list.append(torch.tensor([weights], dtype=torch.float32))

        # 인코딩
        cond, pooled = self.encode_chunks(chunks_l, chunks_g)

        # 가중치 적용 (ComfyUI 방식: weight != 1.0인 토큰에 대해 empty embedding과 lerp)
        if apply_weights:
            # 가중치를 하나의 텐서로 합침
            all_weights = torch.cat(weights_list, dim=1)  # [1, total_seq_len]
            all_weights = all_weights.to(cond.device)

            # 가중치가 1.0이 아닌 토큰이 있는지 확인
            has_weights = (all_weights != 1.0).any()

            if has_weights:
                # Empty 프롬프트 인코딩 (캐싱 가능)
                if not hasattr(self, '_empty_cond'):
                    empty_chunks_l, empty_chunks_g = self.tokenizer.tokenize("")
                    self._empty_cond, _ = self.encode_chunks(empty_chunks_l, empty_chunks_g)

                # 빈 임베딩을 현재 길이에 맞게 확장
                empty_cond = self._empty_cond
                if empty_cond.shape[1] < cond.shape[1]:
                    # 청크 수 만큼 반복
                    repeat_times = (cond.shape[1] + 76) // 77
                    empty_cond = empty_cond.repeat(1, repeat_times, 1)[:, :cond.shape[1], :]

                empty_cond = empty_cond.to(cond.device)

                # 가중치 적용: cond = empty + weight * (cond - empty)
                # 또는: cond = lerp(empty, cond, weight)
                weight_expanded = all_weights.unsqueeze(-1)  # [1, seq_len, 1]
                cond = empty_cond + weight_expanded * (cond - empty_cond)

        logging.info(f"[CLIP] Encoded cond shape: {cond.shape}")
        return cond, pooled

    def load_clip_l(self, sd: dict):
        """CLIP-L state dict 로드"""
        return self.clip_l.load_sd(sd)

    def load_clip_g(self, sd: dict):
        """CLIP-G state dict 로드"""
        return self.clip_g.load_sd(sd)

    def to(self, *args, **kwargs):
        self.clip_l = self.clip_l.to(*args, **kwargs)
        self.clip_g = self.clip_g.to(*args, **kwargs)
        return self


def create_sdxl_clip(device="cpu", dtype=None) -> SDXLClipModel:
    """SDXL CLIP 모델 생성"""
    return SDXLClipModel(device=device, dtype=dtype)
