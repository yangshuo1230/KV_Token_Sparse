from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SegmentConfig:
    min_tokens: int = 6
    max_tokens: int = 32


@dataclass(frozen=True)
class AnalysisConfig:
    layers: list[int] | str = "all"
    block_sizes: list[int] = field(default_factory=lambda: [16, 32, 64])
    top_k: list[int] = field(default_factory=lambda: [4, 8, 16])
    local_window: int = 128
    sink_tokens: int = 4
    split_repeats: int = 20
    pair_samples: int = 32
    bootstrap_samples: int = 2000
    token_supplement: int = 2
    regions: list[str] = field(default_factory=lambda: ["remote", "local"])


@dataclass(frozen=True)
class ContextConfig:
    """Controls fixed-budget context ablations for target-token prediction."""

    cache_budgets: list[int] = field(default_factory=lambda: [128, 512, 2048, 8192])
    sink_sizes: list[int] = field(default_factory=lambda: [0, 4, 16])
    eval_tokens: int = 64
    bootstrap_samples: int = 2000
    # Deployment/profile defaults. 128-token pages keep routing granularity
    # small while remaining friendly to paged FlashAttention-style kernels.
    block_size: int = 128
    profile_recent_budget: int = 2048
    long_context_delta_ce: float = 0.1
    sparse_remote_budget: int = 4096
    sparse_selection_refresh: int = 128


@dataclass(frozen=True)
class Config:
    model: str
    output_dir: Path
    corpora: dict[str, int]
    seed: int = 17
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    max_length: int = 4096
    data_dir: Path | None = None
    position_mode: str = "rope"
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    context: ContextConfig = field(default_factory=ContextConfig)


def load_config(path: str | Path) -> Config:
    with Path(path).open() as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    raw["output_dir"] = Path(raw["output_dir"])
    if raw.get("data_dir"):
        raw["data_dir"] = Path(raw["data_dir"])
    raw["segment"] = SegmentConfig(**raw.get("segment", {}))
    raw["analysis"] = AnalysisConfig(**raw.get("analysis", {}))
    raw["context"] = ContextConfig(**raw.get("context", {}))
    cfg = Config(**raw)
    if cfg.position_mode not in {"rope", "raw"}:
        raise ValueError("position_mode must be rope or raw")
    if cfg.segment.min_tokens > cfg.segment.max_tokens:
        raise ValueError("segment.min_tokens must not exceed max_tokens")
    if not cfg.context.cache_budgets or min(cfg.context.cache_budgets) < 2:
        raise ValueError("context.cache_budgets must contain values >= 2")
    if not cfg.context.sink_sizes or min(cfg.context.sink_sizes) < 0:
        raise ValueError("context.sink_sizes must contain non-negative values")
    if max(cfg.context.sink_sizes) >= min(cfg.context.cache_budgets):
        raise ValueError("every sink size must be smaller than every cache budget")
    if cfg.context.eval_tokens < 1:
        raise ValueError("context.eval_tokens must be positive")
    if cfg.context.eval_tokens >= min(cfg.context.cache_budgets) - max(cfg.context.sink_sizes):
        raise ValueError("context.eval_tokens must fit in the smallest recent window")
    if cfg.context.block_size < 1:
        raise ValueError("context.block_size must be positive")
    if cfg.context.profile_recent_budget not in cfg.context.cache_budgets:
        raise ValueError("context.profile_recent_budget must be one of context.cache_budgets")
    if cfg.context.long_context_delta_ce < 0:
        raise ValueError("context.long_context_delta_ce must be non-negative")
    if cfg.context.sparse_remote_budget < 0:
        raise ValueError("context.sparse_remote_budget must be non-negative")
    if cfg.context.sparse_selection_refresh < 1:
        raise ValueError("context.sparse_selection_refresh must be positive")
    return cfg
