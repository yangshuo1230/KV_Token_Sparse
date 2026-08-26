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


def load_config(path: str | Path) -> Config:
    with Path(path).open() as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    raw["output_dir"] = Path(raw["output_dir"])
    if raw.get("data_dir"):
        raw["data_dir"] = Path(raw["data_dir"])
    raw["segment"] = SegmentConfig(**raw.get("segment", {}))
    raw["analysis"] = AnalysisConfig(**raw.get("analysis", {}))
    cfg = Config(**raw)
    if cfg.position_mode not in {"rope", "raw"}:
        raise ValueError("position_mode must be rope or raw")
    if cfg.segment.min_tokens > cfg.segment.max_tokens:
        raise ValueError("segment.min_tokens must not exceed max_tokens")
    return cfg
