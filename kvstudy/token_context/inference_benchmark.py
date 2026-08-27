from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch

from ..config import Config
from ..model import load_model
from .routed_inference import (
    SparseDecodeController,
    configure_v1_route,
    configure_v2_route,
    enable_routed_decode,
)
from .router import _load_metrics
from .kv_cache import clone_dynamic_cache


@torch.inference_mode()
def benchmark_v1_inference(
    cfg: Config,
    context_lengths: list[int],
    decode_tokens: int = 64,
) -> Path:
    """Measure real Qwen decode with full, recent, and oracle-rate mixed routes."""
    if decode_tokens < 8:
        raise ValueError("decode_tokens must be at least 8")
    context_path = (cfg.data_dir or cfg.output_dir) / "contexts.jsonl"
    if not context_path.exists():
        raise FileNotFoundError("run context-prepare first")
    text = json.loads(context_path.open(encoding="utf-8").readline())["text"]
    model, tokenizer = load_model(cfg.model, cfg.device, cfg.dtype)
    metrics = _load_metrics(cfg.output_dir)
    operating = metrics[
        metrics.sink_size.eq(0)
        & metrics.cache_budget.eq(cfg.context.profile_recent_budget)
    ]
    long_fraction = float(operating.delta_ce.gt(cfg.context.long_context_delta_ce).mean())
    rows: list[dict] = []

    for length in context_lengths:
        model.set_attn_implementation("sdpa")
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=length,
            return_tensors="pt",
        ).input_ids.to(cfg.device)
        if encoded.shape[1] != length:
            raise RuntimeError(f"benchmark text has fewer than {length} tokens")
        # Prefill is shared; this experiment isolates autoregressive decode.
        prefill = model(input_ids=encoded, use_cache=True, return_dict=True)
        initial_cache = prefill.past_key_values
        initial_token = prefill.logits[:, -1:].argmax(dim=-1)
        enable_routed_decode(model)

        controller = SparseDecodeController(
            cfg.context.block_size,
            cfg.context.profile_recent_budget,
            cfg.context.sparse_remote_budget,
            cfg.device,
        )
        policies = ("dense", "recent", "v1_oracle_rate_schedule", "v2_oracle_rate_schedule")
        for policy in policies:
            cache = clone_dynamic_cache(initial_cache)
            token = initial_token.clone()
            position = length
            controller.reset_page_table()
            latencies = []
            long_routes = 0
            for step in range(decode_tokens):
                if policy == "dense":
                    use_long = True
                elif policy == "recent":
                    use_long = False
                elif "oracle_rate" in policy:
                    # Deterministic low-discrepancy schedule at the empirical
                    # oracle rate. This measures compute only, not deployable
                    # router quality, which is reported separately.
                    use_long = ((step * 37) % decode_tokens) < round(
                        long_fraction * decode_tokens
                    )
                long_routes += int(use_long)
                if policy.startswith("v2"):
                    if step and step % cfg.context.sparse_selection_refresh == 0:
                        controller.reset_page_table()
                    configure_v2_route(model, controller, use_long)
                else:
                    configure_v1_route(model, cfg.context.profile_recent_budget, use_long)
                pos = torch.tensor([[position]], device=cfg.device)
                cache_pos = torch.tensor([position], device=cfg.device)
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = model(
                    input_ids=token,
                    position_ids=pos,
                    cache_position=cache_pos,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                torch.cuda.synchronize()
                latencies.append(1000 * (time.perf_counter() - start))
                cache = output.past_key_values
                token = output.logits[:, -1:].argmax(dim=-1)
                position += 1
            measured = latencies[4:]
            rows.append({
                "model": cfg.model,
                "device": torch.cuda.get_device_name(torch.device(cfg.device)),
                "dtype": cfg.dtype,
                "context_length": length,
                "policy": policy,
                "recent_budget": cfg.context.profile_recent_budget,
                "profile_long_fraction": long_fraction,
                "actual_long_fraction": long_routes / decode_tokens,
                "decode_latency_ms_median": float(pd.Series(measured).median()),
                "decode_latency_ms_mean": float(pd.Series(measured).mean()),
                "decode_tokens": decode_tokens,
                "warmup_tokens_excluded": 4,
            })
            del cache
        del initial_cache, prefill, encoded
        torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    dense = result[result.policy.eq("dense")].set_index("context_length")[
        "decode_latency_ms_median"
    ]
    result["median_speedup_vs_dense"] = result.apply(
        lambda row: dense.loc[row.context_length] / row.decode_latency_ms_median,
        axis=1,
    )
    dense_mean = result[result.policy.eq("dense")].set_index("context_length")[
        "decode_latency_ms_mean"
    ]
    result["mean_speedup_vs_dense"] = result.apply(
        lambda row: dense_mean.loc[row.context_length] / row.decode_latency_ms_mean,
        axis=1,
    )
    output = cfg.output_dir / "end_to_end_benchmark.csv"
    result.to_csv(output, index=False)
    return output
