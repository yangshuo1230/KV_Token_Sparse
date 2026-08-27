from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from tqdm import tqdm

from ..config import Config
from ..data import read_contexts
from ..model import decoder_layers, load_model
from ..segments import annotate, load_nlp


def _mask(length: int, query_indices: list[int], local_window: int,
          sink_tokens: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Causal additive mask; selected rows cannot see remote history."""
    mask = torch.full((1, 1, length, length), -torch.inf, device=device, dtype=dtype)
    causal = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
    base = torch.zeros(length, length, device=device, dtype=dtype)
    base.masked_fill_(causal, -torch.inf)
    mask[0, 0] = base
    for i in query_indices:
        remote_end = max(sink_tokens, i - local_window)
        mask[0, 0, i, sink_tokens:remote_end] = -torch.inf
    return mask


def _token_rows(ann, rng: random.Random, budget: int = 6) -> list[tuple[int, str]]:
    """Sample a fixed number per token category, avoiding group interventions."""
    categories = defaultdict(list)
    for i, category in enumerate(ann.categories):
        if category not in {"other"}:
            categories[category].append(i)
    rows = []
    for category, indices in sorted(categories.items()):
        for index in rng.sample(indices, min(budget, len(indices))):
            rows.append((index, category))
    return rows


@torch.inference_mode()
def diagnose_context_length(cfg: Config, budget: int = 16) -> Path:
    data_dir = cfg.data_dir or cfg.output_dir
    idf = json.loads((data_dir / "idf.json").read_text(encoding="utf-8"))["idf"]
    # Eager attention is required for a per-query 4-D additive mask.
    model, tokenizer = load_model(cfg.model, cfg.device, cfg.dtype)
    model.config._attn_implementation = "eager"
    nlp = load_nlp()
    rng = random.Random(cfg.seed)
    records, counts = [], defaultdict(int)
    for record in read_contexts(data_dir / "contexts.jsonl"):
        if counts[record["corpus"]] < cfg.corpora.get(record["corpus"], 0):
            records.append(record)
            counts[record["corpus"]] += 1
    rows = []
    for record in tqdm(records, desc="context-length ablation"):
        ann = annotate(record["text"], tokenizer, nlp, idf, cfg.segment.min_tokens,
                       cfg.segment.max_tokens, cfg.max_length, cfg.analysis.block_sizes,
                       record.get("evidence_spans", []))
        selected = _token_rows(ann, rng, budget)
        if not selected:
            continue
        ids = torch.tensor([ann.input_ids], device=cfg.device)
        length = ids.shape[1]
        full = model(input_ids=ids, use_cache=False, return_dict=True, output_attentions=False)
        full_logits = full.logits[0].float().cpu().numpy()
        for index, token_type in selected:
            if index >= length - 1:
                continue
            # One query row per forward pass: no cross-token intervention
            # contamination. Other query rows retain the normal causal mask.
            masked = _mask(length, [index], cfg.analysis.local_window,
                           cfg.analysis.sink_tokens, cfg.device, next(model.parameters()).dtype)
            masked_out = model(input_ids=ids, attention_mask=masked, use_cache=False,
                               return_dict=True, output_attentions=False)
            masked_logits = masked_out.logits[0].float().cpu().numpy()
            target = int(ids[0, index + 1])
            p = softmax(full_logits[index] - full_logits[index].max())
            q = softmax(masked_logits[index] - masked_logits[index].max())
            ce_full = -np.log(max(p[target], 1e-12))
            ce_masked = -np.log(max(q[target], 1e-12))
            kl = float(np.sum(p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(q, 1e-12)))))
            rows.append({"document": record["id"], "corpus": record["corpus"],
                         "token_index": index, "token_type": token_type,
                         "token": tokenizer.decode([ann.input_ids[index]]),
                         "next_token": tokenizer.decode([target]),
                         "remote_available": index > cfg.analysis.local_window + cfg.analysis.sink_tokens,
                         "ce_full": ce_full, "ce_masked": ce_masked,
                         "delta_ce": ce_masked - ce_full, "kl_full_to_masked": kl,
                         "top1_changed": int(np.argmax(full_logits[index]) != np.argmax(masked_logits[index]))})
            del masked_out
        del full
        torch.cuda.empty_cache()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.output_dir / "context_length_metrics.parquet"
    pd.DataFrame(rows).to_parquet(output, index=False)
    return output
