from __future__ import annotations

import math

import torch


def recent_page_indices(context_length: int, recent_tokens: int, page_size: int) -> torch.Tensor:
    """Return complete page IDs covering the recent window, in causal order."""
    if min(context_length, recent_tokens, page_size) < 1:
        raise ValueError("lengths and page_size must be positive")
    pages = math.ceil(context_length / page_size)
    recent_pages = min(pages, math.ceil(recent_tokens / page_size))
    return torch.arange(pages - recent_pages, pages, dtype=torch.int32)


def select_sparse_pages(
    query: torch.Tensor,
    key_landmarks: torch.Tensor,
    recent_pages: torch.Tensor,
    remote_pages: int,
    sink_pages: int = 1,
) -> torch.Tensor:
    """Select remote KV pages using one key landmark per page.

    This is an intentionally small query-aware selector: grouped-query heads
    score cached page landmarks, scores are averaged across heads, and top
    remote pages are merged with a prefix sink and the mandatory recent pages.
    It reads O(number_of_pages) landmarks rather than O(context_length) KV.
    """
    if query.ndim != 2 or key_landmarks.ndim != 3:
        raise ValueError("expected query [Hq,D] and landmarks [P,Hkv,D]")
    pages, kv_heads, head_dim = key_landmarks.shape
    if query.shape[1] != head_dim or query.shape[0] % kv_heads:
        raise ValueError("query heads must be a multiple of KV heads")
    if remote_pages < 0 or sink_pages < 0:
        raise ValueError("remote_pages and sink_pages must be non-negative")
    if recent_pages.numel() and int(recent_pages.max()) >= pages:
        raise ValueError("recent page index is outside the landmark cache")

    recent_start = int(recent_pages.min()) if recent_pages.numel() else pages
    sink_count = min(sink_pages, recent_start)
    candidate_start = sink_count
    candidate_end = recent_start
    candidate_count = max(0, candidate_end - candidate_start)
    take = min(remote_pages, candidate_count)
    device = query.device
    sink = torch.arange(sink_count, device=device, dtype=torch.int32)
    recent = recent_pages.to(device=device, dtype=torch.int32)
    if take:
        groups = query.shape[0] // kv_heads
        # Averaging GQA queries before one GEMV is algebraically equivalent to
        # averaging all per-head landmark scores, with fewer launches and no
        # expanded [page, head, group] intermediate.
        grouped_q = query.view(kv_heads, groups, head_dim).mean(dim=1).flatten()
        candidates = key_landmarks[candidate_start:candidate_end].flatten(1)
        scores = torch.mv(candidates, grouped_q)
        selected = scores.topk(take, sorted=False).indices.add(candidate_start).to(torch.int32)
        selected = selected.sort().values
    else:
        selected = torch.empty(0, device=device, dtype=torch.int32)
    return torch.cat((sink, selected, recent))


def page_landmarks(key_pages: torch.Tensor) -> torch.Tensor:
    """Create one persistent landmark per page from page-local mean keys."""
    if key_pages.ndim != 4:
        raise ValueError("expected paged keys [P,page,Hkv,D]")
    return key_pages.float().mean(dim=1).to(key_pages.dtype)
