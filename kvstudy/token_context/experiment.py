from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from ..config import Config
from ..data import _pg19
from ..model import decoder_layers, load_model
from ..segments import load_nlp
from .categories import annotate_targets


def retained_indices(length: int, cache_budget: int, sink_size: int) -> list[int]:
    """Return a fixed-budget sink-plus-recent cache in causal order."""
    if length < cache_budget:
        raise ValueError("cache budget cannot exceed sequence length")
    if not 0 <= sink_size < cache_budget:
        raise ValueError("sink size must be in [0, cache_budget)")
    recent_size = cache_budget - sink_size
    recent_start = length - recent_size
    # Very short sequences can make the two ranges overlap. The length check
    # above prevents that for fixed-budget policies, but keep this robust for
    # direct callers.
    return list(range(min(sink_size, recent_start))) + list(range(recent_start, length))


def prepare_contexts(cfg: Config) -> Path:
    """Cache enough PG-19 text to produce exactly ``max_length`` tokens."""
    data_dir = cfg.data_dir or cfg.output_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    output = data_dir / "contexts.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, use_fast=True)
    count = cfg.corpora.get("pg19", 0)
    with output.open("w", encoding="utf-8") as handle:
        written = 0
        for document in tqdm(_pg19(), total=count, desc="prepare long PG-19"):
            encoded = tokenizer(
                document.text,
                add_special_tokens=True,
                truncation=True,
                max_length=cfg.max_length,
                return_offsets_mapping=True,
            )
            if len(encoded["input_ids"]) < cfg.max_length:
                continue
            char_end = encoded["offset_mapping"][-1][1]
            handle.write(json.dumps({
                "id": f"pg19:{document.doc_id}",
                "corpus": "pg19",
                "text": document.text[:char_end],
            }, ensure_ascii=False) + "\n")
            written += 1
            if written == count:
                break
    if written != count:
        raise RuntimeError(f"requested {count} long documents, found {written}")
    return output


def _distribution_metrics(
    full_logits: torch.Tensor,
    compact_logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, list[float]]:
    full_logp = torch.log_softmax(full_logits.float(), dim=-1)
    compact_logp = torch.log_softmax(compact_logits.float(), dim=-1)
    compact_p = compact_logp.exp()
    rows = torch.arange(targets.numel(), device=targets.device)
    ce_full = -full_logp[rows, targets]
    ce_compact = -compact_logp[rows, targets]
    kl = (full_logp.exp() * (full_logp - compact_logp)).sum(dim=-1)
    top_values, top_ids = compact_p.topk(2, dim=-1)
    return {
        "ce_full": ce_full.cpu().tolist(),
        "ce_compact": ce_compact.cpu().tolist(),
        "delta_ce": (ce_compact - ce_full).cpu().tolist(),
        "kl_full_to_compact": kl.cpu().tolist(),
        "top1_changed": (full_logits.argmax(-1) != compact_logits.argmax(-1)).cpu().tolist(),
        "predicted_token_id": top_ids[:, 0].cpu().tolist(),
        "predicted_token_probability": top_values[:, 0].cpu().tolist(),
        "prediction_margin": (top_values[:, 0] - top_values[:, 1]).cpu().tolist(),
        "prediction_entropy": (-(compact_p * compact_logp).sum(dim=-1)).cpu().tolist(),
    }


@torch.inference_mode()
def run_context_ablation(
    cfg: Config,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Path:
    """Compare full context with fixed-budget sink-plus-recent policies.

    Every policy retains exactly ``cache_budget`` positions. A sink size of zero
    is the recent-only control; positive values replace the same number of
    recent positions with the prefix, so sink comparisons do not get extra KV.
    Original absolute position IDs are retained.
    """
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    data_dir = cfg.data_dir or cfg.output_dir
    path = data_dir / "contexts.jsonl"
    if not path.exists():
        raise FileNotFoundError("run context-prepare first")

    all_records = [json.loads(line) for line in path.open(encoding="utf-8")]
    records = all_records[shard_index::num_shards]
    model, tokenizer = load_model(cfg.model, cfg.device, cfg.dtype)
    base, _ = decoder_layers(model)
    nlp = load_nlp()
    rows: list[dict] = []
    eval_tokens = cfg.context.eval_tokens

    for record in tqdm(records, desc=f"context documents {shard_index + 1}/{num_shards}"):
        encoded = tokenizer(
            record["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=cfg.max_length,
            return_offsets_mapping=True,
        )
        ids = torch.tensor([encoded["input_ids"]], device=cfg.device)
        length = ids.shape[1]
        if length != cfg.max_length:
            raise RuntimeError(f"{record['id']} has {length} tokens, expected {cfg.max_length}")
        positions = torch.arange(length, device=cfg.device)[None, :]
        target_start = length - eval_tokens
        targets = ids[0, target_start:]
        annotations = annotate_targets(
            record["text"],
            [tuple(offset) for offset in encoded["offset_mapping"]],
            target_start,
            nlp,
        )

        full_hidden = base(
            input_ids=ids,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, -(eval_tokens + 1):-1]
        full_logits = model.lm_head(full_hidden)[0]

        for cache_budget in cfg.context.cache_budgets:
            for sink_size in cfg.context.sink_sizes:
                indices = retained_indices(length, cache_budget, sink_size)
                compact_ids = ids[:, indices]
                compact_positions = positions[:, indices]
                compact_mask = None
                if sink_size:
                    # Preserve compact causal order while retaining original
                    # absolute positions for RoPE. Otherwise Transformers sees
                    # the sink/recent gap as a packed-sequence boundary and
                    # masks the sink from every recent query.
                    compact_length = len(indices)
                    compact_mask = torch.full(
                        (compact_length, compact_length),
                        torch.finfo(getattr(torch, cfg.dtype)).min,
                        dtype=getattr(torch, cfg.dtype),
                        device=cfg.device,
                    ).triu(diagonal=1)[None, None]
                compact_hidden = base(
                    input_ids=compact_ids,
                    position_ids=compact_positions,
                    attention_mask=compact_mask,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state[:, -(eval_tokens + 1):-1]
                compact_logits = model.lm_head(compact_hidden)[0]
                metrics = _distribution_metrics(full_logits, compact_logits, targets)
                recent_size = cache_budget - sink_size

                for offset, annotation in enumerate(annotations):
                    target_index = target_start + offset
                    token_id = int(targets[offset])
                    query_token_id = int(ids[0, target_index - 1])
                    rows.append({
                        "document": record["id"],
                        "corpus": record["corpus"],
                        "target_index": target_index,
                        "distance_from_end": length - target_index,
                        "token_id": token_id,
                        "token": tokenizer.decode([token_id]),
                        "query_token_id": query_token_id,
                        "query_token": tokenizer.decode([query_token_id]),
                        "coarse_category": annotation.coarse_category,
                        "fine_category": annotation.fine_category,
                        "pos": annotation.pos,
                        "dependency": annotation.dependency,
                        "is_first_subtoken": annotation.is_first_subtoken,
                        "subtoken_index": annotation.subtoken_index,
                        "cache_budget": cache_budget,
                        "sink_size": sink_size,
                        "recent_size": recent_size,
                        **{name: values[offset] for name, values in metrics.items()},
                    })
                del compact_hidden, compact_logits, compact_mask
        del full_hidden, full_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    name = (
        "context_metrics.parquet" if num_shards == 1
        else f"context_metrics-{shard_index:03d}-of-{num_shards:03d}.parquet"
    )
    output = cfg.output_dir / name
    pd.DataFrame(rows).to_parquet(output, index=False)
    return output
