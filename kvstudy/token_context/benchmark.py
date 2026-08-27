from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch

from ..config import Config
from ..model import load_model
from .kv_cache import cache_token_count, prune_dynamic_cache


def _cache_bytes(cache) -> int:
    legacy = cache.to_legacy_cache()
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in legacy
        for tensor in layer[:2]
    )


@torch.inference_mode()
def benchmark_cache(
    cfg: Config,
    prefill_tokens: int = 8192,
    repeats: int = 20,
) -> Path:
    """Measure one-token decode with real sink-plus-recent DynamicCache objects."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    data_dir = cfg.data_dir or cfg.output_dir
    context_path = data_dir / "contexts.jsonl"
    if not context_path.exists():
        raise FileNotFoundError("run context-prepare first")
    record = json.loads(context_path.open(encoding="utf-8").readline())
    model, tokenizer = load_model(cfg.model, cfg.device, cfg.dtype)
    encoded = tokenizer(
        record["text"],
        add_special_tokens=True,
        truncation=True,
        max_length=prefill_tokens + 1,
        return_tensors="pt",
    )
    ids = encoded.input_ids.to(cfg.device)
    if ids.shape[1] != prefill_tokens + 1:
        raise RuntimeError("benchmark document is too short")
    prefix, next_token = ids[:, :-1], ids[:, -1:]
    positions = torch.arange(prefill_tokens, device=cfg.device)[None, :]
    prefill = model(
        input_ids=prefix,
        position_ids=positions,
        use_cache=True,
        return_dict=True,
    )
    original = prefill.past_key_values
    cache_position = torch.tensor([prefill_tokens], device=cfg.device)
    rows = []

    for budget in [value for value in cfg.context.cache_budgets if value <= prefill_tokens]:
        for sink_size in cfg.context.sink_sizes:
            prune_times = []
            decode_times = []
            retained = None
            size_bytes = None
            for _ in range(repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                cache = prune_dynamic_cache(original, budget, sink_size)
                torch.cuda.synchronize()
                prune_times.append(time.perf_counter() - start)
                retained = cache_token_count(cache)
                size_bytes = _cache_bytes(cache)

                torch.cuda.synchronize()
                start = time.perf_counter()
                model(
                    input_ids=next_token,
                    position_ids=cache_position[None, :],
                    cache_position=cache_position,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                torch.cuda.synchronize()
                decode_times.append(time.perf_counter() - start)
            rows.append({
                "model": cfg.model,
                "prefill_tokens": prefill_tokens,
                "cache_budget": budget,
                "sink_size": sink_size,
                "retained_tokens": retained,
                "kv_bytes": size_bytes,
                "prune_latency_ms_median": 1000 * float(pd.Series(prune_times).median()),
                "decode_latency_ms_median": 1000 * float(pd.Series(decode_times).median()),
                "repeats": repeats,
            })

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.output_dir / "cache_benchmark.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
