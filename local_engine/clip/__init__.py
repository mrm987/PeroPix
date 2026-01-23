# ComfyUI CLIP 포팅
from .clip_model import CLIPTextModel, CLIP_L_CONFIG, CLIP_G_CONFIG
from .sdxl_clip import (
    SDXLClipModel, SDXLClipL, SDXLClipG, SDXLTokenizer, create_sdxl_clip,
    # 임베딩 관련
    EmbeddingDatabase, load_embed, get_embedding_database, set_embedding_directory
)
