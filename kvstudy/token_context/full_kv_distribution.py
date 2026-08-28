from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from ..config import Config
from ..model import load_model
from .routed_inference import configure_v1_route, enable_routed_decode


class FullKVDistributionRecorder:
    """Record attention mass over every fixed-size KV block."""

    def __init__(
        self,
        layers: list[int],
        block_size: int,
        document: str,
        context_length: int,
    ) -> None:
        self.layers = set(layers)
        self.block_size = block_size
        self.document = document
        self.context_length = context_length
        self.decode_step = -1
        self.rows: list[dict] = []

    @torch.inference_mode()
    def __call__(self, module, query, key, value, scaling: float) -> None:
        if module.layer_idx not in self.layers:
            return
        groups = query.shape[1] // key.shape[1]
        q = query[0, :, 0].float()
        k = key[0].repeat_interleave(groups, dim=0).float()
        scores = torch.einsum("hd,hld->hl", q, k) * scaling
        weights = torch.softmax(scores, dim=-1)
        length = key.shape[-2]
        blocks = math.ceil(length / self.block_size)
        padded = blocks * self.block_size - length
        if padded:
            weights = torch.nn.functional.pad(weights, (0, padded))
        block_mass = weights.view(weights.shape[0], blocks, self.block_size).sum(dim=-1)
        for block in range(blocks):
            start = block * self.block_size
            end = min(length, start + self.block_size)
            mass = block_mass[:, block]
            uniform_mass = (end - start) / length
            self.rows.append({
                "document": self.document,
                "context_length": self.context_length,
                "decode_step": self.decode_step,
                "key_length": length,
                "layer": module.layer_idx,
                "block_index": block,
                "key_start": start,
                "key_end": end,
                "uniform_mass": uniform_mass,
                "attention_mass_mean": float(mass.mean().cpu()),
                "attention_mass_median": float(mass.median().cpu()),
                "attention_mass_max": float(mass.max().cpu()),
                "concentration_mean": float((mass / uniform_mass).mean().cpu()),
            })


@torch.inference_mode()
def run_full_kv_distribution(
    cfg: Config,
    context_length: int = 16384,
    documents: int = 8,
    eval_tokens: int = 64,
    block_size: int = 128,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Path:
    """Profile dense decode attention across the complete KV position axis."""
    if context_length > cfg.max_length or eval_tokens < 1 or block_size < 1:
        raise ValueError("invalid context/evaluation/block size")
    path = (cfg.data_dir or cfg.output_dir) / "contexts.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8")][:documents]
    records = records[shard_index::num_shards]
    model, tokenizer = load_model(cfg.model, cfg.device, cfg.dtype)
    all_rows: list[dict] = []
    layers = [0, 1, 2, 4, 8, 12, 16, 20, 24, 27]

    for record in tqdm(records, desc=f"full KV distribution {shard_index + 1}/{num_shards}"):
        model.set_attn_implementation("sdpa")
        ids = tokenizer(
            record["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
        ).input_ids.to(cfg.device)
        target_start = context_length - eval_tokens
        prefill_length = target_start - 1
        prefill = model(input_ids=ids[:, :prefill_length], use_cache=True, return_dict=True)
        cache = prefill.past_key_values
        queries = ids[:, prefill_length : context_length - 1]
        recorder = FullKVDistributionRecorder(
            layers, block_size, record["id"], context_length
        )
        enable_routed_decode(model)
        configure_v1_route(model, recent_budget=context_length, use_long_context=True)
        for layer in model.model.layers:
            layer.self_attn._kv_mass_recorder = recorder
        for step in range(queries.shape[1]):
            recorder.decode_step = step
            position = prefill_length + step
            output = model(
                input_ids=queries[:, step : step + 1],
                position_ids=torch.tensor([[position]], device=cfg.device),
                cache_position=torch.tensor([position], device=cfg.device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = output.past_key_values
        for layer in model.model.layers:
            layer.self_attn._kv_mass_recorder = None
        all_rows.extend(recorder.rows)
        del cache, prefill, ids
        torch.cuda.empty_cache()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    name = (
        "full_kv_attention_distribution.parquet"
        if num_shards == 1
        else f"full_kv_attention_distribution-{shard_index:03d}-of-{num_shards:03d}.parquet"
    )
    output = cfg.output_dir / name
    pd.DataFrame(all_rows).to_parquet(output, index=False)
    return output
