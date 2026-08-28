from __future__ import annotations

from collections.abc import Sequence

import torch


def retained_indices(length: int, cache_budget: int, sink_size: int) -> list[int]:
    """Return fixed-budget prefix-sink plus recent indices in causal order."""
    if length < cache_budget:
        raise ValueError("cache budget cannot exceed sequence length")
    if not 0 <= sink_size < cache_budget:
        raise ValueError("sink size must be in [0, cache_budget)")
    recent_size = cache_budget - sink_size
    recent_start = length - recent_size
    return list(range(min(sink_size, recent_start))) + list(range(recent_start, length))


LegacyCache = Sequence[tuple[torch.Tensor, ...]]


def prune_legacy_cache(
    past_key_values: LegacyCache,
    cache_budget: int,
    sink_size: int,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Prune a legacy Hugging Face KV cache to sink-plus-recent positions.

    Keys and values are indexed on their sequence dimension (`-2`). Any
    layer-specific tuple entries after K/V are preserved unchanged.
    """
    if not past_key_values:
        return ()
    length = past_key_values[0][0].shape[-2]
    if length <= cache_budget:
        return tuple(tuple(layer) for layer in past_key_values)
    indices = retained_indices(length, cache_budget, sink_size)
    result = []
    for layer in past_key_values:
        key, value, *extra = layer
        index = torch.tensor(indices, device=key.device)
        result.append((key.index_select(-2, index), value.index_select(-2, index), *extra))
    return tuple(result)


def prune_dynamic_cache(cache, cache_budget: int, sink_size: int):
    """Return a pruned `transformers.DynamicCache` without mutating the input."""
    from transformers import DynamicCache

    if not isinstance(cache, DynamicCache):
        raise TypeError(f"expected DynamicCache, got {type(cache).__name__}")
    legacy = prune_legacy_cache(cache.to_legacy_cache(), cache_budget, sink_size)
    return DynamicCache.from_legacy_cache(legacy)


def cache_token_count(cache) -> int:
    """Return retained sequence positions for legacy or dynamic caches."""
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return int(cache[0][0].shape[-2]) if cache else 0


def clone_dynamic_cache(cache):
    """Clone a DynamicCache so benchmark policies start from identical KV."""
    from transformers import DynamicCache

    if not isinstance(cache, DynamicCache):
        raise TypeError(f"expected DynamicCache, got {type(cache).__name__}")
    legacy = tuple(
        tuple(tensor.clone() for tensor in layer)
        for layer in cache.to_legacy_cache()
    )
    return DynamicCache.from_legacy_cache(legacy)
