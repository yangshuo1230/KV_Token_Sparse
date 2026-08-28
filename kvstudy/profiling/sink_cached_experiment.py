from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from ..config import Config
from ..model import load_model
from .experiment import _distribution_metrics
from ..runtime.kv_cache import clone_dynamic_cache
from ..backends import DenseBackend, FixedRemoteBackend, RecentBackend
from ..runtime.engine import DecodeEngine
from .sink_mass import AttentionSinkMassRecorder


def fixed_remote_indices(
    policy: str,
    count: int,
    remote_end: int,
    seed: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Build prefix or matched non-prefix remote-token controls."""
    if count < 1 or remote_end <= count:
        raise ValueError("remote region must be larger than the selected count")
    if policy == "prefix" or policy == "prefix_zero_value":
        result = torch.arange(count)
    else:
        # Exclude the first 128 positions so controls cannot accidentally use
        # the same sink region. This tests prefix specificity, not remote value.
        start = min(128, remote_end - count)
        candidates = torch.arange(start, remote_end)
        if policy == "random_remote":
            generator = torch.Generator().manual_seed(seed)
            result = candidates[torch.randperm(len(candidates), generator=generator)[:count]].sort().values
        elif policy == "strided_remote":
            offsets = torch.linspace(0, len(candidates) - 1, count).round().long()
            result = candidates[offsets].unique(sorted=True)
            if len(result) != count:
                raise RuntimeError("strided control produced duplicate indices")
        else:
            raise ValueError(f"unknown fixed remote policy {policy}")
    return result.to(device=device, dtype=torch.long)


def _stable_seed(document: str, context_length: int, count: int, seed: int) -> int:
    digest = hashlib.sha256(f"{document}:{context_length}:{count}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


@torch.inference_mode()
def run_cached_sink_experiment(
    cfg: Config,
    context_lengths: list[int],
    documents: int = 8,
    eval_tokens: int = 64,
    budgets: list[int] | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Path:
    """Test sink utility in real cached decode at exactly matched KV reads."""
    if documents < 2 or eval_tokens < 8:
        raise ValueError("use at least two documents and eight evaluation tokens")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard assignment")
    budgets = budgets or [128, 512, 2048, 8192]
    sink_sizes = [1, 4, 16, 64]
    path = (cfg.data_dir or cfg.output_dir) / "contexts.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8")][:documents]
    records = records[shard_index::num_shards]
    model, tokenizer = load_model(cfg.model, cfg.device, cfg.dtype)
    engine = DecodeEngine(model)
    rows: list[dict] = []
    mass_rows: list[dict] = []

    for context_length in context_lengths:
        if context_length > cfg.max_length or context_length <= eval_tokens + max(budgets):
            raise ValueError("context length must fit data and exceed evaluation/cache budgets")
        for record in tqdm(
            records,
            desc=f"cached sink {context_length} shard {shard_index + 1}/{num_shards}",
        ):
            model.set_attn_implementation("sdpa")
            ids = tokenizer(
                record["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=context_length,
                return_tensors="pt",
            ).input_ids.to(cfg.device)
            if ids.shape[1] != context_length:
                raise RuntimeError("sink document is shorter than requested context")
            target_start = context_length - eval_tokens
            prefill_length = target_start - 1
            prefill = model(input_ids=ids[:, :prefill_length], use_cache=True, return_dict=True)
            original_cache = prefill.past_key_values
            queries = ids[:, prefill_length : context_length - 1]
            targets = ids[0, target_start:context_length]
            dense_cache = clone_dynamic_cache(original_cache)
            recorder = AttentionSinkMassRecorder(
                layers=[0, 1, 2, 4, 8, 12, 16, 20, 24, 27],
                prefix_sizes=[1, 4, 16, 64, 128],
                document=record["id"],
                context_length=context_length,
            )
            for layer in model.model.layers:
                layer.self_attn._kv_mass_recorder = recorder
            full_logits = engine.decode(
                dense_cache,
                queries,
                prefill_length,
                DenseBackend(),
                on_step=lambda step: setattr(recorder, "decode_step", step),
            )
            mass_rows.extend(recorder.rows)
            for layer in model.model.layers:
                layer.self_attn._kv_mass_recorder = None
            del dense_cache

            policy_specs: list[tuple[str, int]] = [("recent_only", 0)]
            for sink_size in sink_sizes:
                policy_specs.append(("prefix", sink_size))
                if sink_size in {4, 16}:
                    policy_specs.append(("prefix_zero_value", sink_size))
                if sink_size in {16, 64}:
                    policy_specs.extend(
                        [("random_remote", sink_size), ("strided_remote", sink_size)]
                    )

            for budget in budgets:
                for policy, remote_count in policy_specs:
                    if remote_count >= budget:
                        continue
                    cache = clone_dynamic_cache(original_cache)
                    if policy == "recent_only":
                        backend = RecentBackend(recent_tokens=budget)
                    else:
                        recent_tokens = budget - remote_count
                        remote_end = prefill_length - recent_tokens
                        indices = fixed_remote_indices(
                            policy,
                            remote_count,
                            remote_end,
                            _stable_seed(record["id"], context_length, remote_count, cfg.seed),
                            cfg.device,
                        )
                        backend = FixedRemoteBackend(
                            remote_indices=indices,
                            recent_tokens=recent_tokens,
                            zero_remote_values=policy == "prefix_zero_value",
                        )
                    compact_logits = engine.decode(
                        cache, queries, prefill_length, backend
                    )
                    metrics = _distribution_metrics(full_logits, compact_logits, targets)
                    for offset in range(eval_tokens):
                        rows.append({
                            "document": record["id"],
                            "context_length": context_length,
                            "target_index": target_start + offset,
                            "decode_step": offset,
                            "cache_budget": budget,
                            "policy": policy,
                            "remote_count": remote_count,
                            "recent_count": budget - remote_count,
                            **{name: values[offset] for name, values in metrics.items()},
                        })
                    del cache, compact_logits
            del original_cache, prefill, full_logits, ids
            torch.cuda.empty_cache()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    name = (
        "cached_sink_metrics.parquet"
        if num_shards == 1
        else f"cached_sink_metrics-{shard_index:03d}-of-{num_shards:03d}.parquet"
    )
    output = cfg.output_dir / name
    pd.DataFrame(rows).to_parquet(output, index=False)
    mass_name = (
        "attention_sink_mass.parquet"
        if num_shards == 1
        else f"attention_sink_mass-{shard_index:03d}-of-{num_shards:03d}.parquet"
    )
    pd.DataFrame(mass_rows).to_parquet(cfg.output_dir / mass_name, index=False)
    return output
