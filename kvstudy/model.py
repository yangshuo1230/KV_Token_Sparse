from __future__ import annotations

from collections.abc import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name: str, device: str, dtype: str):
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    model.to(device).eval()
    return model, tokenizer


def decoder_layers(model):
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base, base.layers
    raise TypeError(f"unsupported decoder layout: {type(model).__name__}")


@torch.inference_mode()
def hidden_states(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    base, _ = decoder_layers(model)
    out = base(input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
    return out.hidden_states


@torch.inference_mode()
def projected_qk(model, layer_index: int, hidden: torch.Tensor,
                 position_ids: torch.Tensor, apply_rope: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    base, layers = decoder_layers(model)
    attn = layers[layer_index].self_attn
    batch, length, _ = hidden.shape
    q_heads = attn.config.num_attention_heads
    kv_heads = attn.config.num_key_value_heads
    head_dim = getattr(attn, "head_dim", attn.config.hidden_size // q_heads)
    q = attn.q_proj(hidden).view(batch, length, q_heads, head_dim).transpose(1, 2)
    k = attn.k_proj(hidden).view(batch, length, kv_heads, head_dim).transpose(1, 2)
    if apply_rope:
        rotary = getattr(attn, "rotary_emb", getattr(base, "rotary_emb", None))
        if rotary is None:
            raise TypeError("model does not expose a compatible rotary embedding")
        cos, sin = rotary(hidden, position_ids)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q[0].float(), k[0].float()


def selected_layers(spec: list[int] | str, count: int) -> list[int]:
    result = list(range(count)) if spec == "all" else list(spec)
    invalid = [x for x in result if x < 0 or x >= count]
    if invalid:
        raise ValueError(f"invalid layers {invalid}; model has {count}")
    return result
