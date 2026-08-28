from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from ..config import Config
from ..model import decoder_layers, load_model
from ..backends.block_attention import page_landmarks, recent_page_indices, select_sparse_pages
from .experiment import _distribution_metrics


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _layer0_sparse_token_indices(
    base,
    ids: torch.Tensor,
    positions: torch.Tensor,
    query_index: int,
    recent_tokens: int,
    remote_tokens: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a page table from the first-layer query and RoPE key landmarks."""
    length = ids.shape[1]
    if length % page_size:
        raise ValueError("sparse quality experiment requires page-aligned context length")
    layer = base.layers[0].self_attn
    hidden = base.embed_tokens(ids)
    q_hidden = hidden[:, query_index : query_index + 1]
    q = layer.q_proj(q_hidden).view(1, 1, layer.config.num_attention_heads, layer.head_dim)
    q = q.transpose(1, 2)
    k = layer.k_proj(hidden).view(
        1, length, layer.config.num_key_value_heads, layer.head_dim
    ).transpose(1, 2)
    cos, sin = base.rotary_emb(hidden, positions)
    q_cos = cos[:, query_index : query_index + 1].unsqueeze(1)
    q_sin = sin[:, query_index : query_index + 1].unsqueeze(1)
    q = q * q_cos + _rotate_half(q) * q_sin
    k = k * cos.unsqueeze(1) + _rotate_half(k) * sin.unsqueeze(1)
    key_pages = k[0].transpose(0, 1).reshape(
        length // page_size,
        page_size,
        layer.config.num_key_value_heads,
        layer.head_dim,
    )
    landmarks = page_landmarks(key_pages)
    recent = recent_page_indices(length, recent_tokens, page_size).to(ids.device)
    selected = select_sparse_pages(
        q[0, :, 0],
        landmarks,
        recent,
        # The sink page is part of, rather than extra to, the remote budget.
        remote_pages=max(0, (remote_tokens + page_size - 1) // page_size - 1),
        sink_pages=1,
    )
    offsets = torch.arange(page_size, device=ids.device)
    token_indices = (selected.long()[:, None] * page_size + offsets).flatten()
    return token_indices, selected


@torch.inference_mode()
def run_sparse_context_ablation(
    cfg: Config,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Path:
    """Evaluate V2 first-layer landmark routing on full 32K documents."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    path = (cfg.data_dir or cfg.output_dir) / "contexts.jsonl"
    if not path.exists():
        raise FileNotFoundError("run context-prepare first")
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    records = records[shard_index::num_shards]
    model, tokenizer = load_model(cfg.model, cfg.device, cfg.dtype)
    base, _ = decoder_layers(model)
    rows: list[dict] = []
    eval_tokens = cfg.context.eval_tokens

    for record in tqdm(records, desc=f"sparse documents {shard_index + 1}/{num_shards}"):
        encoded = tokenizer(
            record["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        ids = encoded.input_ids.to(cfg.device)
        if ids.shape[1] != cfg.max_length:
            raise RuntimeError("sparse experiment document has unexpected token length")
        positions = torch.arange(cfg.max_length, device=cfg.device)[None]
        target_start = cfg.max_length - eval_tokens
        targets = ids[0, target_start:]
        full_hidden = base(
            input_ids=ids,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, -(eval_tokens + 1) : -1]
        full_logits = model.lm_head(full_hidden)[0]

        indices, pages = _layer0_sparse_token_indices(
            base,
            ids,
            positions,
            query_index=target_start - 1,
            recent_tokens=cfg.context.profile_recent_budget,
            remote_tokens=cfg.context.sparse_remote_budget,
            page_size=cfg.context.block_size,
        )
        compact_length = int(indices.numel())
        # Non-contiguous original position IDs trigger Transformers' packed-
        # sequence heuristic, which would isolate each selected run. Supply an
        # explicit compact-order causal mask while retaining original IDs for
        # RoPE. This lets every recent query attend all earlier selected pages.
        compact_mask = torch.full(
            (compact_length, compact_length),
            torch.finfo(getattr(torch, cfg.dtype)).min,
            dtype=getattr(torch, cfg.dtype),
            device=cfg.device,
        ).triu(diagonal=1)[None, None]
        sparse_hidden = base(
            input_ids=ids[:, indices],
            position_ids=positions[:, indices],
            attention_mask=compact_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, -(eval_tokens + 1) : -1]
        sparse_logits = model.lm_head(sparse_hidden)[0]
        metrics = _distribution_metrics(full_logits, sparse_logits, targets)
        recent_page_start = cfg.max_length // cfg.context.block_size - (
            cfg.context.profile_recent_budget // cfg.context.block_size
        )
        remote_selected = int((pages < recent_page_start).sum())
        for offset in range(eval_tokens):
            rows.append({
                "document": record["id"],
                "target_index": target_start + offset,
                "context_length": cfg.max_length,
                "page_size": cfg.context.block_size,
                "recent_tokens": cfg.context.profile_recent_budget,
                "remote_selected_pages": remote_selected,
                "attended_tokens": int(indices.numel()),
                "selection_layer": 0,
                "selection_refresh_tokens": eval_tokens,
                **{name: values[offset] for name, values in metrics.items()},
            })
        del full_hidden, sparse_hidden, full_logits, sparse_logits, compact_mask
        torch.cuda.empty_cache()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    name = (
        "sparse_context_metrics.parquet"
        if num_shards == 1
        else f"sparse_context_metrics-{shard_index:03d}-of-{num_shards:03d}.parquet"
    )
    output = cfg.output_dir / name
    pd.DataFrame(rows).to_parquet(output, index=False)
    return output
