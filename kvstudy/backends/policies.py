from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .routed_inference import (
    FixedRemoteDecodeController,
    SparseDecodeController,
    configure_fixed_remote_route,
    configure_v1_route,
    configure_v2_route,
    enable_routed_decode,
)


@dataclass
class DenseBackend:
    name: str = "dense"

    def install(self, model) -> None:
        enable_routed_decode(model)
        configure_v1_route(model, recent_budget=1, use_long_context=True)

    def before_step(self, model, step: int) -> None:
        return None


@dataclass
class RecentBackend:
    recent_tokens: int
    name: str = "recent"

    def install(self, model) -> None:
        enable_routed_decode(model)
        configure_v1_route(model, self.recent_tokens, use_long_context=False)

    def before_step(self, model, step: int) -> None:
        return None


@dataclass
class SinkRecentBackend:
    total_budget: int
    sink_tokens: int = 1
    concatenate_segments: bool = False
    name: str = "sink_recent"
    _controller: FixedRemoteDecodeController | None = field(default=None, init=False)

    def install(self, model) -> None:
        enable_routed_decode(model)
        device = next(model.parameters()).device
        self._controller = FixedRemoteDecodeController(
            torch.arange(self.sink_tokens, device=device),
            recent_tokens=self.total_budget - self.sink_tokens,
            concatenate_segments=self.concatenate_segments,
        )
        configure_fixed_remote_route(model, self._controller)

    def before_step(self, model, step: int) -> None:
        return None


@dataclass
class FixedRemoteBackend:
    remote_indices: torch.Tensor
    recent_tokens: int
    zero_remote_values: bool = False
    concatenate_segments: bool = False
    name: str = "fixed_remote"
    _controller: FixedRemoteDecodeController | None = field(default=None, init=False)

    def install(self, model) -> None:
        enable_routed_decode(model)
        device = next(model.parameters()).device
        self._controller = FixedRemoteDecodeController(
            self.remote_indices.to(device),
            recent_tokens=self.recent_tokens,
            zero_remote_values=self.zero_remote_values,
            concatenate_segments=self.concatenate_segments,
        )
        configure_fixed_remote_route(model, self._controller)

    def before_step(self, model, step: int) -> None:
        return None


@dataclass
class SparseBackend:
    page_size: int
    recent_tokens: int
    remote_tokens: int
    refresh_steps: int = 128
    use_remote: bool = True
    name: str = "sparse"
    _controller: SparseDecodeController | None = field(default=None, init=False)

    def install(self, model) -> None:
        enable_routed_decode(model)
        device = next(model.parameters()).device
        self._controller = SparseDecodeController(
            self.page_size,
            self.recent_tokens,
            self.remote_tokens,
            device,
        )
        configure_v2_route(model, self._controller, self.use_remote)

    def before_step(self, model, step: int) -> None:
        assert self._controller is not None
        if step and step % self.refresh_steps == 0:
            self._controller.reset_page_table()
        configure_v2_route(model, self._controller, self.use_remote)
