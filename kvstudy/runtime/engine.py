from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch


class DecodeBackend(Protocol):
    """Minimal interface implemented by every KV attention policy."""

    name: str

    def install(self, model) -> None: ...

    def before_step(self, model, step: int) -> None: ...


@dataclass(frozen=True)
class DecodeBatch:
    query_ids: torch.Tensor
    start_position: int


@torch.inference_mode()
def teacher_forced_decode(
    model,
    cache,
    batch: DecodeBatch,
    backend: DecodeBackend,
    on_step: Callable[[int], None] | None = None,
) -> torch.Tensor:
    """Run one shared autoregressive loop with a pluggable KV backend."""
    backend.install(model)
    logits = []
    for step in range(batch.query_ids.shape[1]):
        if on_step is not None:
            on_step(step)
        backend.before_step(model, step)
        position = batch.start_position + step
        output = model(
            input_ids=batch.query_ids[:, step : step + 1],
            position_ids=torch.tensor([[position]], device=batch.query_ids.device),
            cache_position=torch.tensor([position], device=batch.query_ids.device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        logits.append(output.logits[:, -1])
    return torch.cat(logits, dim=0)


class DecodeEngine:
    """Small orchestration object shared by benchmarks and profiling jobs."""

    def __init__(self, model) -> None:
        self.model = model

    def decode(
        self,
        cache,
        query_ids: torch.Tensor,
        start_position: int,
        backend,
        on_step: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        return teacher_forced_decode(
            self.model,
            cache,
            DecodeBatch(query_ids=query_ids, start_position=start_position),
            backend,
            on_step,
        )
