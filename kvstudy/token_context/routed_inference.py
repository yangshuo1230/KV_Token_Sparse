from __future__ import annotations

import torch

from .block_attention import page_landmarks, select_sparse_pages


ATTENTION_NAME = "kv_route_flashinfer"


class SparseDecodeController:
    """Shared paged-kernel state for all layers of one batch-one decode."""

    def __init__(
        self,
        page_size: int,
        recent_tokens: int,
        remote_tokens: int,
        device: str | torch.device,
    ) -> None:
        import flashinfer

        if min(page_size, recent_tokens, remote_tokens) < 1:
            raise ValueError("page, recent, and remote budgets must be positive")
        self.page_size = page_size
        self.recent_tokens = recent_tokens
        self.remote_pages = (remote_tokens + page_size - 1) // page_size
        self.workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace, kv_layout="NHD", use_tensor_cores=True
        )
        self.page_indices: torch.Tensor | None = None
        self.planned_length = 0

    def reset_page_table(self) -> None:
        """Force query-aware page reselection on the next layer-0 call."""
        self.page_indices = None

    def plan(self, query: torch.Tensor, key: torch.Tensor) -> None:
        """Select remote pages from layer-0 KV and prepare the reusable wrapper."""
        length = key.shape[-2]
        full_pages = length // self.page_size
        remote_end = max(0, (length - self.recent_tokens) // self.page_size)
        if remote_end == 0:
            self.page_indices = torch.empty(0, dtype=torch.int32, device=key.device)
            self.planned_length = length
            return
        paged = key[0, :, : full_pages * self.page_size, :].transpose(0, 1).reshape(
            full_pages, self.page_size, key.shape[1], key.shape[-1]
        )
        landmarks = page_landmarks(paged)
        later_pages = torch.arange(remote_end, full_pages, dtype=torch.int32, device=key.device)
        selected = select_sparse_pages(
            query,
            landmarks,
            later_pages,
            remote_pages=max(0, self.remote_pages - 1),
            sink_pages=1,
        )
        self.page_indices = selected[selected < remote_end]
        indptr = torch.tensor(
            [0, self.page_indices.numel()], dtype=torch.int32, device=key.device
        )
        last_page_len = torch.tensor([self.page_size], dtype=torch.int32, device=key.device)
        self.wrapper.plan(
            indptr,
            self.page_indices,
            last_page_len,
            query.shape[0],
            key.shape[1],
            key.shape[-1],
            self.page_size,
            q_data_type=query.dtype,
            kv_data_type=key.dtype,
        )
        self.planned_length = length

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scaling: float,
        select_pages: bool,
    ) -> torch.Tensor:
        """Attend separate remote pages and recent view, then merge via LSE."""
        import flashinfer

        if select_pages or self.page_indices is None:
            self.plan(query, key)
        recent = min(self.recent_tokens, key.shape[-2])
        recent_output, recent_lse = flashinfer.single_decode_with_kv_cache(
            query,
            key[0, :, -recent:, :].transpose(0, 1),
            value[0, :, -recent:, :].transpose(0, 1),
            kv_layout="NHD",
            use_tensor_cores=True,
            sm_scale=scaling,
            return_lse=True,
        )
        if self.page_indices is None or not self.page_indices.numel():
            return recent_output
        full_pages = key.shape[-2] // self.page_size
        k_pages = key[0, :, : full_pages * self.page_size, :].transpose(0, 1).reshape(
            full_pages, self.page_size, key.shape[1], key.shape[-1]
        )
        v_pages = value[0, :, : full_pages * self.page_size, :].transpose(0, 1).reshape_as(k_pages)
        remote_output, remote_lse = self.wrapper.run(
            query[None], (k_pages, v_pages), return_lse=True
        )
        remote_output = remote_output[0]
        remote_lse = remote_lse[0]
        combined_lse = torch.logaddexp(recent_lse, remote_lse)
        recent_weight = torch.exp(recent_lse - combined_lse)[:, None]
        remote_weight = torch.exp(remote_lse - combined_lse)[:, None]
        return (
            recent_output.float() * recent_weight
            + remote_output.float() * remote_weight
        ).to(query.dtype)


def routed_flashinfer_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    **kwargs,
):
    """Batch-one, single-query GQA over full or zero-copy recent KV views."""
    if query.shape[0] != 1 or query.shape[-2] != 1:
        raise ValueError("routed decode attention supports batch=1 and query_length=1")
    try:
        import flashinfer
    except ImportError as exc:
        raise RuntimeError("FlashInfer is required for routed decode") from exc

    budget = int(getattr(module, "_kv_recent_budget", key.shape[-2]))
    use_long = bool(getattr(module, "_kv_route_long", True))
    controller = getattr(module, "_kv_sparse_controller", None)
    if use_long and controller is not None:
        output = controller.attend(
            query[0, :, 0, :],
            key,
            value,
            scaling,
            select_pages=(module.layer_idx == 0 and controller.page_indices is None),
        )
        return output[None, None, :, :], None
    if not use_long and key.shape[-2] > budget:
        key = key[..., -budget:, :]
        value = value[..., -budget:, :]
    q = query[0, :, 0, :]
    k = key[0].transpose(0, 1)
    v = value[0].transpose(0, 1)
    output = flashinfer.single_decode_with_kv_cache(
        q,
        k,
        v,
        kv_layout="NHD",
        use_tensor_cores=True,
        sm_scale=scaling,
    )
    return output[None, None, :, :], None


def register_routed_attention() -> None:
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(ATTENTION_NAME, routed_flashinfer_attention)


def configure_v1_route(model, recent_budget: int, use_long_context: bool) -> None:
    """Set one route for every decoder layer before the next-token forward."""
    if recent_budget < 1:
        raise ValueError("recent_budget must be positive")
    base = getattr(model, "model", model)
    for layer in base.layers:
        layer.self_attn._kv_recent_budget = recent_budget
        layer.self_attn._kv_route_long = use_long_context
        layer.self_attn._kv_sparse_controller = None


def configure_v2_route(
    model,
    controller: SparseDecodeController,
    use_long_context: bool,
) -> None:
    """Use recent-only for cheap tokens and paged sparse remote KV for long tokens."""
    base = getattr(model, "model", model)
    for layer in base.layers:
        layer.self_attn._kv_recent_budget = controller.recent_tokens
        layer.self_attn._kv_route_long = use_long_context
        layer.self_attn._kv_sparse_controller = controller


def enable_routed_decode(model) -> None:
    register_routed_attention()
    model.set_attn_implementation(ATTENTION_NAME)
