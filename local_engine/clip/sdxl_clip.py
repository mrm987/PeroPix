"""
SDXL CLIP 인코더

ComfyUI의 SDXL CLIP 구현을 단순화하여 포팅
SDXL은 CLIP-L (768d)과 CLIP-G (1280d)를 사용하여 2048d 임베딩 생성
"""

import torch
import torch.nn as nn
import logging
import os
from pathlib import Path
from transformers import CLIPTokenizer
from typing import List, Tuple, Optional, Union, Dict, Any

from .clip_model import CLIPTextModel, CLIP_L_CONFIG, CLIP_G_CONFIG
from ..ops import manual_cast as ops, cast_to


# =============================================================================
# 임베딩 로드 유틸리티 (ComfyUI sd1_clip.py 방식)
# =============================================================================

SUPPORTED_EMBEDDING_EXTENSIONS = [".safetensors", ".pt", ".bin"]


def load_embed(embed_path: str, embedding_expected_shape: int, embedding_directory: Optional[str] = None) -> Optional[torch.Tensor]:
    """
    임베딩 파일 로드 (ComfyUI load_embed 함수 포팅)

    Args:
        embed_path: 임베딩 파일 경로 (또는 이름)
        embedding_expected_shape: 예상 임베딩 차원 (768 for CLIP-L, 1280 for CLIP-G)
        embedding_directory: 임베딩 디렉토리 (경로 검증용)

    Returns:
        임베딩 텐서 [num_vectors, embed_dim] 또는 None
    """
    # 경로 보안 검증 (디렉토리 탈출 방지)
    if embedding_directory is not None:
        try:
            embed_path = os.path.abspath(embed_path)
            embedding_directory = os.path.abspath(embedding_directory)
            # 임베딩 경로가 디렉토리 내에 있는지 확인
            if not embed_path.startswith(embedding_directory):
                logging.warning(f"[Embedding] Path escape detected: {embed_path}")
                return None
        except Exception:
            return None

    embed = None

    try:
        if embed_path.lower().endswith(".safetensors"):
            import safetensors.torch
            embed = safetensors.torch.load_file(embed_path, device="cpu")
        else:
            embed = torch.load(embed_path, weights_only=True, map_location="cpu")
    except Exception as e:
        logging.warning(f"[Embedding] Failed to load {embed_path}: {e}")
        return None

    if embed is None:
        return None

    # 텐서 추출 (다양한 형식 지원)
    out = None

    if isinstance(embed, dict):
        # 일반적인 형식: {"emb_name": tensor} 또는 {"string_to_param": {"*": tensor}}
        if "string_to_param" in embed:
            embed = embed["string_to_param"]
            if "*" in embed:
                out = embed["*"]
        elif "emb_params" in embed:
            out = embed["emb_params"]
        else:
            # 첫 번째 텐서 사용
            for key in embed:
                if isinstance(embed[key], torch.Tensor):
                    out = embed[key]
                    break
    elif isinstance(embed, torch.Tensor):
        out = embed

    if out is None:
        logging.warning(f"[Embedding] No valid tensor found in {embed_path}")
        return None

    # 텐서 형태 정리
    if out.dim() == 1:
        out = out.unsqueeze(0)  # [embed_dim] -> [1, embed_dim]

    # 차원 검증
    if out.shape[-1] != embedding_expected_shape:
        # 일부 임베딩은 여러 모델용 텐서를 포함 (SD1.5 768, SDXL 1280 등)
        # 올바른 크기의 텐서 찾기
        if out.numel() % embedding_expected_shape == 0:
            num_vectors = out.numel() // embedding_expected_shape
            out = out.reshape(num_vectors, embedding_expected_shape)
        else:
            logging.warning(
                f"[Embedding] Dimension mismatch in {embed_path}: "
                f"got {out.shape[-1]}, expected {embedding_expected_shape}"
            )
            return None

    return out.float()


class EmbeddingDatabase:
    """
    임베딩 데이터베이스 (ComfyUI 방식)

    디렉토리에서 임베딩 파일을 찾아 로드하고 캐싱
    """

    def __init__(self, embedding_directory: Optional[str] = None):
        self.embedding_directory = embedding_directory
        self.embeddings_cache: Dict[str, Dict[str, torch.Tensor]] = {}  # {name: {"clip_l": tensor, "clip_g": tensor}}
        self._scanned = False
        self._available_embeddings: Dict[str, str] = {}  # {name: path}

    def set_directory(self, directory: str):
        """임베딩 디렉토리 설정"""
        self.embedding_directory = directory
        self._scanned = False
        self._available_embeddings.clear()
        self.embeddings_cache.clear()

    def scan_directory(self):
        """디렉토리에서 사용 가능한 임베딩 스캔"""
        if self.embedding_directory is None:
            return

        self._available_embeddings.clear()
        emb_dir = Path(self.embedding_directory)

        if not emb_dir.exists():
            logging.warning(f"[Embedding] Directory not found: {self.embedding_directory}")
            self._scanned = True
            return

        for ext in SUPPORTED_EMBEDDING_EXTENSIONS:
            for file_path in emb_dir.glob(f"*{ext}"):
                name = file_path.stem  # 확장자 제외한 파일명
                self._available_embeddings[name.lower()] = str(file_path)

        if self._available_embeddings:
            logging.info(f"[Embedding] Found {len(self._available_embeddings)} embeddings")

        self._scanned = True

    def get_embedding(self, name: str, clip_l_dim: int = 768, clip_g_dim: int = 1280) -> Optional[Dict[str, torch.Tensor]]:
        """
        임베딩 가져오기 (캐시 사용)

        Args:
            name: 임베딩 이름 (확장자 없이)
            clip_l_dim: CLIP-L 임베딩 차원
            clip_g_dim: CLIP-G 임베딩 차원

        Returns:
            {"clip_l": tensor, "clip_g": tensor} 또는 None
        """
        if not self._scanned:
            self.scan_directory()

        name_lower = name.lower()

        # 캐시 확인
        if name_lower in self.embeddings_cache:
            return self.embeddings_cache[name_lower]

        # 파일 찾기
        embed_path = None
        if name_lower in self._available_embeddings:
            embed_path = self._available_embeddings[name_lower]
        else:
            # 확장자 포함된 이름으로 시도
            for ext in SUPPORTED_EMBEDDING_EXTENSIONS:
                test_name = name_lower.replace(ext, "")
                if test_name in self._available_embeddings:
                    embed_path = self._available_embeddings[test_name]
                    break

        if embed_path is None:
            return None

        # 임베딩 로드 (CLIP-L과 CLIP-G 모두)
        # SDXL 임베딩은 보통 CLIP-G (1280) 차원
        embed_g = load_embed(embed_path, clip_g_dim, self.embedding_directory)

        # SD1.5 호환 임베딩일 경우 CLIP-L 차원도 시도
        embed_l = load_embed(embed_path, clip_l_dim, self.embedding_directory)

        if embed_g is None and embed_l is None:
            logging.warning(f"[Embedding] Failed to load: {name}")
            return None

        result = {}
        if embed_l is not None:
            result["clip_l"] = embed_l
        if embed_g is not None:
            result["clip_g"] = embed_g

        # 캐시에 저장
        self.embeddings_cache[name_lower] = result
        logging.info(f"[Embedding] Loaded: {name} (L: {embed_l is not None}, G: {embed_g is not None})")

        return result

    def list_embeddings(self) -> List[str]:
        """사용 가능한 임베딩 목록 반환"""
        if not self._scanned:
            self.scan_directory()
        return list(self._available_embeddings.keys())


# 전역 임베딩 데이터베이스
_embedding_db: Optional[EmbeddingDatabase] = None


def get_embedding_database() -> EmbeddingDatabase:
    """전역 임베딩 데이터베이스 가져오기"""
    global _embedding_db
    if _embedding_db is None:
        _embedding_db = EmbeddingDatabase()
    return _embedding_db


def set_embedding_directory(directory: str):
    """임베딩 디렉토리 설정"""
    db = get_embedding_database()
    db.set_directory(directory)


# =============================================================================
# CLIP 모델 클래스
# =============================================================================


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
    임베딩 지원: embedding:name 구문으로 텍스트 임베딩 사용
    """

    # 임베딩 구문 식별자
    EMBEDDING_IDENTIFIER = "embedding:"

    def __init__(self, tokenizer_path: Optional[str] = None, embedding_directory: Optional[str] = None):
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

        # 임베딩 설정
        self.embedding_directory = embedding_directory
        self._embedding_db: Optional[EmbeddingDatabase] = None

    def get_embedding_db(self) -> EmbeddingDatabase:
        """임베딩 데이터베이스 가져오기 (지연 초기화)"""
        if self._embedding_db is None:
            self._embedding_db = EmbeddingDatabase(self.embedding_directory)
        return self._embedding_db

    def set_embedding_directory(self, directory: str):
        """임베딩 디렉토리 설정"""
        self.embedding_directory = directory
        if self._embedding_db is not None:
            self._embedding_db.set_directory(directory)

    def _try_get_embedding(self, embedding_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        임베딩 가져오기 시도 (ComfyUI _try_get_embedding 방식)

        Args:
            embedding_name: 임베딩 이름 (embedding: 접두사 제외)

        Returns:
            {"clip_l": tensor, "clip_g": tensor} 또는 None
        """
        if self.embedding_directory is None:
            return None

        db = self.get_embedding_db()
        return db.get_embedding(embedding_name.strip())

    def _parse_embedding_segment(self, text: str) -> List[Tuple[Union[str, Dict[str, torch.Tensor]], float]]:
        """
        텍스트에서 임베딩 구문 파싱 (ComfyUI 방식)

        embedding:name 형식을 인식하고 일반 텍스트와 분리

        Args:
            text: 파싱할 텍스트

        Returns:
            [(텍스트 또는 임베딩 dict, 가중치), ...]
        """
        result = []

        # embedding: 으로 분할 (공백/줄바꿈 앞에만)
        parts = re.split(r'(?:^|\s)' + re.escape(self.EMBEDDING_IDENTIFIER), text)

        for i, part in enumerate(parts):
            if i == 0:
                # 첫 부분은 임베딩 접두사 없음
                if part.strip():
                    result.append((part.strip(), 1.0))
                continue

            # embedding:name 이후 부분 처리
            # 첫 단어가 임베딩 이름, 나머지는 일반 텍스트
            words = part.split(None, 1)  # 최대 2개로 분할
            if not words:
                continue

            embed_name = words[0].rstrip(',')  # 쉼표 제거
            leftover = words[1] if len(words) > 1 else ""

            # 임베딩 로드 시도
            embed_data = self._try_get_embedding(embed_name)
            if embed_data is not None:
                result.append((embed_data, 1.0))
                logging.debug(f"[Tokenizer] Found embedding: {embed_name}")
            else:
                # 임베딩을 찾지 못하면 텍스트로 처리
                logging.warning(f"[Tokenizer] Embedding not found: {embed_name}")
                result.append((f"embedding:{embed_name}", 1.0))

            # 남은 텍스트 추가
            if leftover.strip():
                result.append((leftover.strip(), 1.0))

        return result

    def tokenize_with_weights_and_embeddings(self, text: str) -> Tuple[
        List[Tuple[List[Tuple[Union[int, torch.Tensor], float]], List[Tuple[Union[int, torch.Tensor], float]]]],
        bool
    ]:
        """
        ComfyUI 방식: 가중치 파싱 + 임베딩 처리 + 세그먼트 토큰화 + 청킹

        Returns:
            ([(chunk_l, chunk_g), ...], has_embeddings)
            각 chunk는 [(token_id 또는 embedding_tensor, weight), ...]
        """
        # 가중치 파싱
        weighted_segments = parse_prompt_weights(text)

        # 임베딩이 있는지 확인하고 세그먼트 처리
        has_embeddings = False
        processed_segments = []

        for segment_text, weight in weighted_segments:
            if not segment_text.strip():
                continue

            # 임베딩 구문 파싱
            if self.EMBEDDING_IDENTIFIER in segment_text and self.embedding_directory:
                emb_segments = self._parse_embedding_segment(segment_text)
                for seg, _ in emb_segments:
                    if isinstance(seg, dict):
                        # 임베딩
                        has_embeddings = True
                        processed_segments.append((seg, weight))
                    else:
                        # 일반 텍스트
                        processed_segments.append((seg, weight))
            else:
                processed_segments.append((segment_text, weight))

        # 토큰화 + 임베딩 통합
        # 토큰과 임베딩을 혼합한 리스트
        # [(tokens_or_embed, weight, is_embedding), ...]
        word_tokens = []

        for segment, weight in processed_segments:
            if isinstance(segment, dict):
                # 임베딩: {"clip_l": tensor, "clip_g": tensor}
                word_tokens.append((segment, weight, True))
            else:
                # 일반 텍스트: 토큰화
                if not segment.strip():
                    continue
                encoded = self.tokenizer(
                    segment,
                    truncation=False,
                    add_special_tokens=False
                )
                tokens = encoded["input_ids"]
                if tokens:
                    word_tokens.append((tokens, weight, False))

        # 청킹 (임베딩 벡터 수를 토큰 수로 계산)
        chunks_l = []
        chunks_g = []

        current_chunk_l: List[Tuple[Union[int, torch.Tensor], float]] = [(self.start_token, 1.0)]
        current_chunk_g: List[Tuple[Union[int, torch.Tensor], float]] = [(self.start_token, 1.0)]

        for item, weight, is_embedding in word_tokens:
            if is_embedding:
                # 임베딩: 각 벡터가 1토큰 공간 차지
                embed_dict = item
                embed_l = embed_dict.get("clip_l")
                embed_g = embed_dict.get("clip_g")

                # 벡터 수 결정 (CLIP-G 우선, 없으면 CLIP-L)
                if embed_g is not None:
                    num_vectors = embed_g.shape[0]
                elif embed_l is not None:
                    num_vectors = embed_l.shape[0]
                else:
                    continue

                # 청크 경계 확인
                batch_len = len(current_chunk_l)
                remaining = self.max_length - batch_len - 1  # end 토큰 공간

                if num_vectors > remaining:
                    # 청크를 먼저 완성
                    if batch_len > 1:
                        current_chunk_l.append((self.end_token, 1.0))
                        current_chunk_g.append((self.end_token, 1.0))
                        while len(current_chunk_l) < self.max_length:
                            current_chunk_l.append((self.pad_token_l, 1.0))
                            current_chunk_g.append((self.pad_token_g, 1.0))
                        chunks_l.append(current_chunk_l)
                        chunks_g.append(current_chunk_g)
                        current_chunk_l = [(self.start_token, 1.0)]
                        current_chunk_g = [(self.start_token, 1.0)]

                # 임베딩 벡터 추가 (각 행이 하나의 "토큰")
                for vec_idx in range(num_vectors):
                    vec_l = embed_l[vec_idx] if embed_l is not None else None
                    vec_g = embed_g[vec_idx] if embed_g is not None else None

                    current_chunk_l.append((vec_l, weight))
                    current_chunk_g.append((vec_g, weight))

                    # 청크가 가득 찼으면 새 청크 시작
                    if len(current_chunk_l) >= self.max_length - 1:
                        current_chunk_l.append((self.end_token, 1.0))
                        current_chunk_g.append((self.end_token, 1.0))
                        chunks_l.append(current_chunk_l)
                        chunks_g.append(current_chunk_g)
                        current_chunk_l = [(self.start_token, 1.0)]
                        current_chunk_g = [(self.start_token, 1.0)]

            else:
                # 일반 토큰
                tokens = list(item)
                is_large_word = len(tokens) >= self.max_word_length

                while len(tokens) > 0:
                    batch_len = len(current_chunk_l)

                    if len(tokens) + batch_len > self.max_length - 1:
                        remaining = self.max_length - batch_len - 1

                        if is_large_word and remaining > 0:
                            for t in tokens[:remaining]:
                                current_chunk_l.append((t, weight))
                                current_chunk_g.append((t, weight))
                            tokens = tokens[remaining:]

                        current_chunk_l.append((self.end_token, 1.0))
                        current_chunk_g.append((self.end_token, 1.0))

                        while len(current_chunk_l) < self.max_length:
                            current_chunk_l.append((self.pad_token_l, 1.0))
                            current_chunk_g.append((self.pad_token_g, 1.0))

                        chunks_l.append(current_chunk_l)
                        chunks_g.append(current_chunk_g)

                        current_chunk_l = [(self.start_token, 1.0)]
                        current_chunk_g = [(self.start_token, 1.0)]

                        if not is_large_word:
                            for t in tokens:
                                current_chunk_l.append((t, weight))
                                current_chunk_g.append((t, weight))
                            tokens = []
                    else:
                        for t in tokens:
                            current_chunk_l.append((t, weight))
                            current_chunk_g.append((t, weight))
                        tokens = []

        # 마지막 청크 완성
        if len(current_chunk_l) > 1:
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

        return list(zip(chunks_l, chunks_g)), has_embeddings

    def tokenize_with_weights(self, text: str) -> List[Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]]:
        """
        ComfyUI 방식: 가중치 파싱 + 세그먼트 토큰화 + 청킹

        ComfyUI sd1_clip.py:573 방식을 정확히 재현:
        - 각 가중치 세그먼트를 통째로 토큰화 (단어별 X)
        - 세그먼트 토큰 수가 max_word_length 이상이면 청크 경계에서 분할 허용
        - 아니면 통째로 다음 청크로 이동

        Returns:
            [(chunk_l, chunk_g), ...] where each chunk is [(token, weight), ...]
        """
        # 가중치 파싱
        weighted_segments = parse_prompt_weights(text)

        # ComfyUI 방식: 세그먼트 전체를 한번에 토큰화 (sd1_clip.py:573)
        # 각 세그먼트가 하나의 "word" (t_group)로 취급됨
        word_tokens = []  # [(tokens, weight), ...] where tokens = [token_id, ...]

        for segment_text, weight in weighted_segments:
            if not segment_text.strip():
                continue

            # ComfyUI: 세그먼트 전체를 한번에 토큰화
            # (embedding 처리는 생략 - PeroPix는 embedding 미지원)
            encoded = self.tokenizer(
                segment_text,
                truncation=False,
                add_special_tokens=False
            )
            tokens = encoded["input_ids"]
            if tokens:
                word_tokens.append((tokens, weight))

        # ComfyUI 방식 청킹 (sd1_clip.py:575-611)
        chunks_l = []
        chunks_g = []

        current_chunk_l = [(self.start_token, 1.0)]
        current_chunk_g = [(self.start_token, 1.0)]

        # ComfyUI 조건 (sd1_clip.py:590):
        # if len(t_group) + len(batch) > self.max_length - has_end_token:
        #     remaining_length = self.max_length - len(batch) - has_end_token
        #
        # len(batch) = start 토큰 포함 전체 길이
        # 넘침 조건: len(tokens) + len(batch) > 77 - 1 = 76
        # remaining_length = 77 - len(batch) - 1

        for tokens, weight in word_tokens:
            is_large_word = len(tokens) >= self.max_word_length

            while len(tokens) > 0:
                batch_len = len(current_chunk_l)  # start 포함 전체 길이

                # ComfyUI 조건: len(tokens) + batch_len > 76
                if len(tokens) + batch_len > self.max_length - 1:
                    # 청크 넘침
                    # ComfyUI: remaining_length = 77 - batch_len - 1
                    remaining = self.max_length - batch_len - 1

                    if is_large_word and remaining > 0:
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
                else:
                    # 현재 청크에 모두 들어감
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
    임베딩 지원: embedding:name 구문으로 텍스트 임베딩 사용
    """

    def __init__(self, device="cpu", dtype=None, embedding_directory: Optional[str] = None):
        super().__init__()
        self.clip_l = SDXLClipL(device=device, dtype=dtype)
        self.clip_g = SDXLClipG(device=device, dtype=dtype)
        self.tokenizer = SDXLTokenizer(embedding_directory=embedding_directory)
        self.embedding_directory = embedding_directory

    def set_embedding_directory(self, directory: str):
        """임베딩 디렉토리 설정"""
        self.embedding_directory = directory
        self.tokenizer.set_embedding_directory(directory)

    def tokenize(self, text: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """텍스트를 CLIP-L/G용 토큰 청크 리스트로 변환"""
        return self.tokenizer.tokenize(text)

    @torch.no_grad()
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

    @torch.no_grad()
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

    def _gen_empty_tokens(self, length: int) -> Tuple[List[int], List[int]]:
        """
        ComfyUI 방식 empty 토큰 생성

        [start, end, pad, pad, ...] 형태로 length만큼 생성
        """
        # CLIP-L: pad = 49407 (end token)
        # CLIP-G: pad = 0
        tokens_l = [self.tokenizer.start_token, self.tokenizer.end_token]
        tokens_g = [self.tokenizer.start_token, self.tokenizer.end_token]

        tokens_l += [self.tokenizer.pad_token_l] * (length - 2)
        tokens_g += [self.tokenizer.pad_token_g] * (length - 2)

        return tokens_l, tokens_g

    def _get_token_embedding(self, clip_model: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
        """
        토큰 ID를 토큰 임베딩으로 변환 (임베딩 레이어만 통과)

        Args:
            clip_model: SDXLClipL 또는 SDXLClipG
            token_ids: [B, seq_len] 토큰 텐서

        Returns:
            [B, seq_len, embed_dim] 토큰 임베딩
        """
        device = next(clip_model.transformer.parameters()).device
        token_ids = token_ids.to(device)
        # CLIPTextModel의 embeddings.token_embedding 사용
        token_embeds = clip_model.transformer.text_model.embeddings.token_embedding(token_ids)
        return token_embeds.float()

    def _encode_with_embeddings(
        self,
        clip_model: nn.Module,
        tokens_or_embeds: List[Tuple[Union[int, torch.Tensor, None], float]],
        pad_token: int,
        embed_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        토큰과 임베딩 벡터가 혼합된 시퀀스를 인코딩 (ComfyUI 방식)

        Args:
            clip_model: SDXLClipL 또는 SDXLClipG
            tokens_or_embeds: [(token_id 또는 embedding_vector 또는 None, weight), ...]
            pad_token: 패딩 토큰 ID
            embed_dim: 임베딩 차원 (768 for CLIP-L, 1280 for CLIP-G)

        Returns:
            (hidden_states, pooled_output)
        """
        device = next(clip_model.transformer.parameters()).device
        seq_len = len(tokens_or_embeds)

        # 토큰 ID와 임베딩 벡터 분리
        token_ids = []
        embedding_positions = []  # (position, embedding_vector, weight)

        for pos, (item, weight) in enumerate(tokens_or_embeds):
            if item is None:
                # None은 패딩으로 처리
                token_ids.append(pad_token)
            elif isinstance(item, torch.Tensor):
                # 임베딩 벡터
                token_ids.append(pad_token)  # placeholder
                embedding_positions.append((pos, item, weight))
            else:
                # 정수 토큰 ID
                token_ids.append(item)

        # 토큰 텐서 생성 [1, seq_len]
        tokens_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

        if not embedding_positions:
            # 임베딩 없음: 일반 인코딩
            hidden, pooled = clip_model(tokens_tensor)
            return hidden, pooled

        # 임베딩 있음: 토큰 임베딩을 먼저 가져온 후 임베딩 벡터로 대체
        token_embeds = self._get_token_embedding(clip_model, tokens_tensor)  # [1, seq_len, embed_dim]

        # 임베딩 벡터 대체
        for pos, embed_vec, weight in embedding_positions:
            if embed_vec.shape[-1] == embed_dim:
                token_embeds[0, pos] = embed_vec.to(device=device, dtype=token_embeds.dtype)

        # Position embedding 추가 및 트랜스포머 통과
        pos_embeds = cast_to(
            clip_model.transformer.text_model.embeddings.position_embedding.weight,
            dtype=token_embeds.dtype,
            device=device
        )
        x = token_embeds + pos_embeds[:seq_len]

        # Causal mask 생성
        causal_mask = torch.full(
            (seq_len, seq_len),
            -torch.finfo(x.dtype).max,
            dtype=x.dtype,
            device=device
        ).triu_(1)

        # 인코더 통과
        x, intermediate = clip_model.transformer.text_model.encoder(
            x,
            mask=causal_mask,
            intermediate_output=clip_model.layer_idx
        )
        x = clip_model.transformer.text_model.final_layer_norm(x)

        if intermediate is not None and clip_model.layer_norm_hidden_state:
            intermediate = clip_model.transformer.text_model.final_layer_norm(intermediate)

        hidden = intermediate.float() if intermediate is not None else x.float()

        # Pooled output (EOS 위치)
        eos_positions = (tokens_tensor == clip_model.transformer.text_model.eos_token_id).int().argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0], device=device), eos_positions]
        pooled = clip_model.transformer.text_projection(pooled).float()

        return hidden, pooled

    def encode_text(self, text: str, apply_weights: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        텍스트를 직접 CLIP 임베딩으로 인코딩

        ComfyUI 방식: 77토큰 초과 시 여러 청크로 처리, 가중치 적용
        임베딩 지원: embedding:name 구문으로 텍스트 임베딩 사용

        Args:
            text: 입력 텍스트
            apply_weights: 가중치 적용 여부 (default: True)

        Returns:
            (cond, pooled): SDXL conditioning 텐서들
        """
        # 임베딩 포함 토큰화 시도
        if self.embedding_directory:
            weighted_chunks, has_embeddings = self.tokenizer.tokenize_with_weights_and_embeddings(text)

            if has_embeddings:
                return self._encode_text_with_embeddings(weighted_chunks, apply_weights)

        # 임베딩 없음: 기존 로직
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
            weights_list.append(weights)

        # 가중치가 1.0이 아닌 토큰이 있는지 확인
        has_weights = False
        if apply_weights:
            for weights in weights_list:
                if any(w != 1.0 for w in weights):
                    has_weights = True
                    break

        # ComfyUI 방식: 가중치가 있으면 empty 토큰을 배치에 포함하여 함께 인코딩
        if has_weights:
            # 각 청크의 최대 길이 (보통 77)
            max_token_len = max(len(w) for w in weights_list)

            # Empty 토큰 생성 (ComfyUI gen_empty_tokens 방식)
            empty_tokens_l, empty_tokens_g = self._gen_empty_tokens(max_token_len)
            empty_chunk_l = torch.tensor([empty_tokens_l], dtype=torch.long)
            empty_chunk_g = torch.tensor([empty_tokens_g], dtype=torch.long)

            # Empty를 배치 마지막에 추가
            chunks_l.append(empty_chunk_l)
            chunks_g.append(empty_chunk_g)

        # 모든 청크를 배치로 인코딩
        batch_l = torch.cat(chunks_l, dim=0)  # [N+1, 77] if has_weights else [N, 77]
        batch_g = torch.cat(chunks_g, dim=0)

        l_hidden, l_pooled = self.clip_l(batch_l)  # [N+1, 77, 768]
        g_hidden, g_pooled = self.clip_g(batch_g)  # [N+1, 77, 1280]

        # 첫 번째 청크의 pooled만 사용
        first_pooled = g_pooled[0:1]  # [1, 1280]

        # 시퀀스 길이 맞추기
        min_len = min(l_hidden.shape[1], g_hidden.shape[1])
        l_hidden = l_hidden[:, :min_len]
        g_hidden = g_hidden[:, :min_len]

        # 가중치 적용 (ComfyUI 방식)
        if has_weights:
            # 마지막 배치가 empty embedding
            l_empty = l_hidden[-1]  # [77, 768]
            g_empty = g_hidden[-1]  # [77, 1280]

            # 실제 청크들만 추출 (empty 제외)
            l_hidden = l_hidden[:-1]  # [N, 77, 768]
            g_hidden = g_hidden[:-1]  # [N, 77, 1280]

            # 각 청크별로 가중치 적용
            for k in range(num_chunks):
                weights = weights_list[k]
                for j in range(len(weights)):
                    weight = weights[j]
                    if weight != 1.0:
                        # ComfyUI 공식: z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
                        l_hidden[k, j] = (l_hidden[k, j] - l_empty[j]) * weight + l_empty[j]
                        g_hidden[k, j] = (g_hidden[k, j] - g_empty[j]) * weight + g_empty[j]

        # 배치를 시퀀스로 변환: [N, 77, dim] -> [1, N*77, dim]
        l_concat = l_hidden.reshape(1, -1, l_hidden.shape[-1])  # [1, N*77, 768]
        g_concat = g_hidden.reshape(1, -1, g_hidden.shape[-1])  # [1, N*77, 1280]

        # 결합: [1, N*77, 768] + [1, N*77, 1280] -> [1, N*77, 2048]
        cond = torch.cat([l_concat, g_concat], dim=-1)

        logging.debug(f"[CLIP] Encoded cond shape: {cond.shape}")
        return cond, first_pooled

    def _encode_text_with_embeddings(
        self,
        weighted_chunks: List[Tuple[List[Tuple[Union[int, torch.Tensor], float]], List[Tuple[Union[int, torch.Tensor], float]]]],
        apply_weights: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        임베딩을 포함한 텍스트 인코딩 (ComfyUI 방식)

        Args:
            weighted_chunks: tokenize_with_weights_and_embeddings의 결과
            apply_weights: 가중치 적용 여부

        Returns:
            (cond, pooled): SDXL conditioning 텐서들
        """
        num_chunks = len(weighted_chunks)

        if num_chunks > 1:
            logging.info(f"[CLIP] Long prompt with embeddings: {num_chunks} chunks")

        l_hiddens = []
        g_hiddens = []
        first_pooled = None

        for chunk_idx, (chunk_l, chunk_g) in enumerate(weighted_chunks):
            # 각 청크별로 인코딩 (임베딩 포함)
            l_hidden, l_pooled = self._encode_with_embeddings(
                self.clip_l, chunk_l, self.tokenizer.pad_token_l, 768
            )
            g_hidden, g_pooled = self._encode_with_embeddings(
                self.clip_g, chunk_g, self.tokenizer.pad_token_g, 1280
            )

            if first_pooled is None:
                first_pooled = g_pooled

            # 가중치 적용 (필요시)
            if apply_weights:
                weights = [w for _, w in chunk_l]
                has_nonunit_weight = any(w != 1.0 for w in weights)

                if has_nonunit_weight:
                    # Empty 토큰 인코딩
                    empty_tokens_l, empty_tokens_g = self._gen_empty_tokens(len(chunk_l))
                    empty_l = self._get_token_embedding(
                        self.clip_l,
                        torch.tensor([empty_tokens_l], dtype=torch.long)
                    )
                    empty_g = self._get_token_embedding(
                        self.clip_g,
                        torch.tensor([empty_tokens_g], dtype=torch.long)
                    )

                    # 가중치 적용
                    for j, weight in enumerate(weights):
                        if weight != 1.0:
                            l_hidden[0, j] = (l_hidden[0, j] - empty_l[0, j]) * weight + empty_l[0, j]
                            g_hidden[0, j] = (g_hidden[0, j] - empty_g[0, j]) * weight + empty_g[0, j]

            l_hiddens.append(l_hidden)
            g_hiddens.append(g_hidden)

        # 모든 청크 결합
        l_concat = torch.cat(l_hiddens, dim=1)  # [1, N*77, 768]
        g_concat = torch.cat(g_hiddens, dim=1)  # [1, N*77, 1280]

        # 결합: [1, N*77, 768] + [1, N*77, 1280] -> [1, N*77, 2048]
        cond = torch.cat([l_concat, g_concat], dim=-1)

        logging.debug(f"[CLIP] Encoded cond shape (with embeddings): {cond.shape}")
        return cond, first_pooled

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
