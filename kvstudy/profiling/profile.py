from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..config import Config
from .router import _load_metrics


def theoretical_attention_cost(
    context_length: int,
    recent_tokens: int,
    long_fraction: float,
    sparse_remote_tokens: int,
    block_size: int,
) -> dict[str, float]:
    """Ideal decode-attention traffic model, measured in KV token reads."""
    if not 0 <= long_fraction <= 1:
        raise ValueError("long_fraction must be in [0, 1]")
    if context_length < 1 or recent_tokens < 1 or block_size < 1:
        raise ValueError("token counts and block_size must be positive")
    if recent_tokens > context_length:
        raise ValueError("recent_tokens cannot exceed context_length")
    rounded_recent = min(context_length, math.ceil(recent_tokens / block_size) * block_size)
    remote_capacity = context_length - rounded_recent
    rounded_sparse = min(
        remote_capacity,
        math.ceil(max(0, sparse_remote_tokens) / block_size) * block_size,
    )
    v1_reads = rounded_recent + long_fraction * remote_capacity
    v2_reads = rounded_recent + long_fraction * rounded_sparse
    return {
        "rounded_recent_tokens": float(rounded_recent),
        "rounded_sparse_remote_tokens": float(rounded_sparse),
        "dense_kv_reads": float(context_length),
        "v1_mean_kv_reads": float(v1_reads),
        "v2_mean_kv_reads": float(v2_reads),
        "v1_attention_upper_bound_speedup": context_length / v1_reads,
        "v2_attention_upper_bound_speedup": context_length / v2_reads,
        "v1_kv_read_reduction": 1 - v1_reads / context_length,
        "v2_kv_read_reduction": 1 - v2_reads / context_length,
    }


def profile_context_need(cfg: Config) -> tuple[Path, Path, Path]:
    """Profile empirical long-context need and emit a theoretical cost model."""
    frame = _load_metrics(cfg.output_dir)
    frame = frame[frame.sink_size.eq(0)].copy()
    identity = ["document", "target_index"]
    if frame.duplicated(identity + ["cache_budget"]).any():
        raise ValueError("context metrics contain duplicate target/policy rows")

    thresholds = sorted({0.0, 0.05, 0.1, 0.2, 0.5, cfg.context.long_context_delta_ce})
    rows: list[dict] = []
    for budget, group in frame.groupby("cache_budget"):
        for threshold in thresholds:
            need = group.delta_ce.gt(threshold)
            rows.append({
                "context_length": cfg.max_length,
                "recent_budget": int(budget),
                "criterion": "delta_ce_gt",
                "threshold": threshold,
                "long_context_tokens": int(need.sum()),
                "recent_only_tokens": int((~need).sum()),
                "long_context_fraction": float(need.mean()),
                "recent_only_fraction": float((~need).mean()),
                "documents": int(group.document.nunique()),
                "targets": len(group),
            })
        changed = group.top1_changed.astype(bool)
        rows.append({
            "context_length": cfg.max_length,
            "recent_budget": int(budget),
            "criterion": "top1_changed",
            "threshold": 1.0,
            "long_context_tokens": int(changed.sum()),
            "recent_only_tokens": int((~changed).sum()),
            "long_context_fraction": float(changed.mean()),
            "recent_only_fraction": float((~changed).mean()),
            "documents": int(group.document.nunique()),
            "targets": len(group),
        })

    profile = pd.DataFrame(rows)
    chosen = profile[
        profile.recent_budget.eq(cfg.context.profile_recent_budget)
        & profile.criterion.eq("delta_ce_gt")
        & profile.threshold.eq(cfg.context.long_context_delta_ce)
    ]
    if len(chosen) != 1:
        raise ValueError("configured profile operating point is missing or ambiguous")
    long_fraction = float(chosen.iloc[0].long_context_fraction)
    costs = theoretical_attention_cost(
        cfg.max_length,
        cfg.context.profile_recent_budget,
        long_fraction,
        cfg.context.sparse_remote_budget,
        cfg.context.block_size,
    )
    theory = pd.DataFrame([{
        "context_length": cfg.max_length,
        "recent_budget": cfg.context.profile_recent_budget,
        "long_context_delta_ce": cfg.context.long_context_delta_ce,
        "long_context_fraction": long_fraction,
        "block_size": cfg.context.block_size,
        "sparse_remote_budget": cfg.context.sparse_remote_budget,
        **costs,
    }])

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = cfg.output_dir / "context_need_profile.csv"
    theory_path = cfg.output_dir / "theoretical_speedup.csv"
    report_path = cfg.output_dir / "CONTEXT_NEED_AND_THEORY.md"
    profile.to_csv(profile_path, index=False)
    theory.to_csv(theory_path, index=False)
    c = chosen.iloc[0]
    t = theory.iloc[0]
    report_path.write_text(
        "# Long-context need profile and theoretical opportunity\n\n"
        f"This profile uses {int(c.targets):,} target tokens from "
        f"{int(c.documents)} PG-19 documents of exactly {cfg.max_length:,} tokens. "
        "A token is labelled as needing long context when recent-only decoding "
        f"increases target cross-entropy by more than {cfg.context.long_context_delta_ce:g} nat "
        "relative to the full context. This is an oracle analysis label, not a deployable feature.\n\n"
        f"At a {cfg.context.profile_recent_budget:,}-token recent window, "
        f"{int(c.long_context_tokens):,}/{int(c.targets):,} tokens "
        f"({100*c.long_context_fraction:.2f}%) need long context; "
        f"{100*c.recent_only_fraction:.2f}% do not.\n\n"
        "## Ideal attention-traffic bound\n\n"
        f"KV is paged in {cfg.context.block_size}-token blocks. V1 always reads the recent "
        "window and reads all remote blocks only for long-context tokens. Its mean KV reads "
        f"are {t.v1_mean_kv_reads:,.0f} tokens/query, a {100*t.v1_kv_read_reduction:.2f}% "
        f"reduction and {t.v1_attention_upper_bound_speedup:.2f}x decode-attention upper bound. "
        f"V2 gives long-context tokens {int(t.rounded_sparse_remote_tokens):,} selected remote "
        f"tokens; it reads {t.v2_mean_kv_reads:,.0f} tokens/query, a "
        f"{100*t.v2_kv_read_reduction:.2f}% reduction and "
        f"{t.v2_attention_upper_bound_speedup:.2f}x upper bound.\n\n"
        "These are bandwidth-only upper bounds. They exclude QKV/output projections, MLPs, "
        "router cost, block selection, kernel launches, and prediction errors; measured "
        "end-to-end results must be reported separately.\n",
        encoding="utf-8",
    )
    return profile_path, theory_path, report_path
