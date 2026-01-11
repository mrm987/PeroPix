"""
ComfyUI CLIP 모델 포팅

원본: ComfyUI/comfy/clip_model.py
SDXL용 CLIP 텍스트 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import disable_weight_init as ops, cast_to


def attention_pytorch(q, k, v, heads, mask=None):
    """PyTorch SDPA 사용"""
    b, seq, dim = q.shape
    dim_head = dim // heads

    q = q.view(b, seq, heads, dim_head).transpose(1, 2)
    k = k.view(b, seq, heads, dim_head).transpose(1, 2)
    v = v.view(b, seq, heads, dim_head).transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    out = out.transpose(1, 2).reshape(b, seq, dim)
    return out


ACTIVATIONS = {
    "quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda a: F.gelu(a, approximate="tanh"),
}


class CLIPAttention(nn.Module):
    def __init__(self, embed_dim, heads, dtype, device, operations):
        super().__init__()
        self.heads = heads
        self.q_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.out_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = attention_pytorch(q, k, v, self.heads, mask)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    def __init__(self, embed_dim, intermediate_size, activation, dtype, device, operations):
        super().__init__()
        self.fc1 = operations.Linear(embed_dim, intermediate_size, bias=True, dtype=dtype, device=device)
        self.activation = ACTIVATIONS[activation]
        self.fc2 = operations.Linear(intermediate_size, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class CLIPLayer(nn.Module):
    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations):
        super().__init__()
        self.layer_norm1 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device, operations)
        self.layer_norm2 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.mlp = CLIPMLP(embed_dim, intermediate_size, intermediate_activation, dtype, device, operations)

    def forward(self, x, mask=None):
        x = x + self.self_attn(self.layer_norm1(x), mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None, intermediate_output=None):
        all_intermediate = None
        if intermediate_output is not None:
            if intermediate_output == "all":
                all_intermediate = []
                intermediate_output = None
            elif intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            if i == intermediate_output:
                intermediate = x.clone()
            if all_intermediate is not None:
                all_intermediate.append(x.unsqueeze(1).clone())

        if all_intermediate is not None:
            intermediate = torch.cat(all_intermediate, dim=1)

        return x, intermediate


class CLIPEmbeddings(nn.Module):
    def __init__(self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None, operations=None):
        super().__init__()
        self.token_embedding = operations.Embedding(vocab_size, embed_dim, dtype=dtype, device=device)
        self.position_embedding = operations.Embedding(num_positions, embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, dtype=torch.float32):
        token_embeds = self.token_embedding(input_tokens, out_dtype=dtype)
        pos_embeds = cast_to(self.position_embedding.weight, dtype=dtype, device=input_tokens.device)
        return token_embeds + pos_embeds


class CLIPTextModel_(nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        num_positions = config_dict["max_position_embeddings"]
        self.eos_token_id = config_dict.get("eos_token_id", 49407)

        self.embeddings = CLIPEmbeddings(embed_dim, num_positions=num_positions, dtype=dtype, device=device, operations=operations)
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device, operations)
        self.final_layer_norm = operations.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens=None, attention_mask=None, embeds=None, num_tokens=None,
                intermediate_output=None, final_layer_norm_intermediate=True, dtype=torch.float32, **kwargs):
        if embeds is not None:
            pos_embeds = cast_to(self.embeddings.position_embedding.weight, dtype=dtype, device=embeds.device)
            x = embeds + pos_embeds
        else:
            x = self.embeddings(input_tokens, dtype=dtype)

        # 마스크 처리
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1]))
            mask = mask.expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), -torch.finfo(x.dtype).max)

        # Causal mask
        causal_mask = torch.full((x.shape[1], x.shape[1]), -torch.finfo(x.dtype).max, dtype=x.dtype, device=x.device).triu_(1)

        if mask is not None:
            mask = mask + causal_mask
        else:
            mask = causal_mask

        x, intermediate = self.encoder(x, mask=mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)

        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)

        # Pooled output
        if num_tokens is not None:
            pooled_output = x[list(range(x.shape[0])), [t - 1 for t in num_tokens]]
        elif input_tokens is not None:
            eos_positions = (input_tokens == self.eos_token_id).int().argmax(dim=-1)
            pooled_output = x[torch.arange(x.shape[0], device=x.device), eos_positions]
        else:
            pooled_output = x[:, -1]  # fallback

        return x, intermediate, pooled_output


class CLIPTextModel(nn.Module):
    """CLIP 텍스트 모델 (CLIP-L/CLIP-G 공용)"""

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device, operations)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = operations.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        projected = self.text_projection(x[2])
        return (x[0], x[1], projected, x[2])  # (last, intermediate, projected_pooled, pooled)


# CLIP 설정
CLIP_L_CONFIG = {
    "num_hidden_layers": 12,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_act": "quick_gelu",
    "max_position_embeddings": 77,
    "eos_token_id": 49407,
}

CLIP_G_CONFIG = {
    "num_hidden_layers": 32,
    "hidden_size": 1280,
    "num_attention_heads": 20,
    "intermediate_size": 5120,
    "hidden_act": "gelu",
    "max_position_embeddings": 77,
    "eos_token_id": 49407,
}
