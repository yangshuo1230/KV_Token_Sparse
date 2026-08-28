from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
import torch

from ..config import Config
from ..backends.block_attention import page_landmarks, recent_page_indices, select_sparse_pages
from .router import _load_metrics


def _median_ms(operation, repeats: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append(1000 * (time.perf_counter() - start))
    return float(pd.Series(samples).median())


@torch.inference_mode()
def benchmark_decode_attention(
    cfg: Config,
    context_lengths: list[int],
    repeats: int = 100,
) -> Path:
    """Benchmark V1/V2 decode attention on real high-performance kernels."""
    if repeats < 1 or not context_lengths or min(context_lengths) < 1:
        raise ValueError("positive repeats and context lengths are required")
    try:
        import flashinfer
    except ImportError as exc:
        raise RuntimeError("FlashInfer is required for the optimized benchmark") from exc

    # Qwen2.5-7B geometry. Read it from model config without loading weights.
    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(cfg.model)
    qo_heads = model_cfg.num_attention_heads
    kv_heads = model_cfg.num_key_value_heads
    head_dim = model_cfg.hidden_size // qo_heads
    dtype = getattr(torch, cfg.dtype)
    device = torch.device(cfg.device)
    page_size = cfg.context.block_size
    rows: list[dict] = []
    metrics = _load_metrics(cfg.output_dir)
    operating = metrics[
        metrics.sink_size.eq(0)
        & metrics.cache_budget.eq(cfg.context.profile_recent_budget)
    ]
    long_fraction = float(
        operating.delta_ce.gt(cfg.context.long_context_delta_ce).mean()
    )

    for length in context_lengths:
        pages = math.ceil(length / page_size)
        padded_length = pages * page_size
        q = torch.randn(qo_heads, head_dim, device=device, dtype=dtype)
        k_pages = torch.randn(pages, page_size, kv_heads, head_dim, device=device, dtype=dtype)
        v_pages = torch.randn_like(k_pages)
        k = k_pages.view(padded_length, kv_heads, head_dim)[:length]
        v = v_pages.view(padded_length, kv_heads, head_dim)[:length]
        recent_ids = recent_page_indices(length, cfg.context.profile_recent_budget, page_size).to(device)
        recent_tokens = min(length, len(recent_ids) * page_size)
        # One sink page is included in the configured remote-page budget.
        sparse_pages = max(0, math.ceil(cfg.context.sparse_remote_budget / page_size) - 1)
        landmarks = page_landmarks(k_pages)

        dense_op = lambda: flashinfer.single_decode_with_kv_cache(
            q, k, v, kv_layout="NHD", use_tensor_cores=True
        )
        recent_op = lambda: flashinfer.single_decode_with_kv_cache(
            q, k[-recent_tokens:], v[-recent_tokens:], kv_layout="NHD", use_tensor_cores=True
        )
        dense_ms = _median_ms(dense_op, repeats)
        recent_ms = _median_ms(recent_op, repeats)

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", use_tensor_cores=True
        )
        sparse_ids = select_sparse_pages(q, landmarks, recent_ids, sparse_pages)
        indptr = torch.tensor([0, sparse_ids.numel()], dtype=torch.int32, device=device)
        last_page_len = torch.tensor([page_size], dtype=torch.int32, device=device)
        wrapper.plan(
            indptr,
            sparse_ids,
            last_page_len,
            qo_heads,
            kv_heads,
            head_dim,
            page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        sparse_op = lambda: wrapper.run(q[None], (k_pages, v_pages))
        sparse_ms = _median_ms(sparse_op, repeats)
        selector_ms = _median_ms(
            lambda: select_sparse_pages(q, landmarks, recent_ids, sparse_pages), repeats
        )

        refresh = cfg.context.sparse_selection_refresh
        v1_mix_ms = (1 - long_fraction) * recent_ms + long_fraction * dense_ms
        v2_mix_kernel_ms = (1 - long_fraction) * recent_ms + long_fraction * sparse_ms
        v2_mix_selector_ms = selector_ms / refresh

        for policy, attended, kernel_ms, selection_ms in (
            ("dense", length, dense_ms, 0.0),
            ("v1_recent", recent_tokens, recent_ms, 0.0),
            ("v1_full", length, dense_ms, 0.0),
            ("v2_sparse", int(sparse_ids.numel()) * page_size, sparse_ms, selector_ms),
            (
                "v1_oracle_mix",
                (1 - long_fraction) * recent_tokens + long_fraction * length,
                v1_mix_ms,
                0.0,
            ),
            (
                "v2_oracle_mix",
                recent_tokens + long_fraction * cfg.context.sparse_remote_budget,
                v2_mix_kernel_ms,
                v2_mix_selector_ms,
            ),
        ):
            rows.append({
                "model": cfg.model,
                "device": torch.cuda.get_device_name(device),
                "dtype": cfg.dtype,
                "context_length": length,
                "page_size": page_size,
                "policy": policy,
                "attended_tokens": attended,
                "kernel_latency_ms_median": kernel_ms,
                "selection_latency_ms_median": selection_ms,
                "total_attention_latency_ms": kernel_ms + selection_ms,
                "speedup_vs_dense": dense_ms / (kernel_ms + selection_ms),
                "profile_long_fraction": long_fraction,
                "selection_refresh_tokens": refresh,
                "repeats": repeats,
            })
    output = cfg.output_dir / "decode_attention_benchmark.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
