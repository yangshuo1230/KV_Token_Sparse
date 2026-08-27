from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoTokenizer

from ..config import Config
from ..data import _pg19
from ..model import decoder_layers, load_model
from ..segments import _category, load_nlp


def prepare_long_contexts(cfg: Config) -> Path:
    """Cache only enough text to produce exactly max_length model tokens."""
    data_dir = cfg.data_dir or cfg.output_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    output = data_dir / "contexts.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, use_fast=True)
    count = cfg.corpora.get("pg19", 0)
    with output.open("w", encoding="utf-8") as handle:
        written = 0
        for document in tqdm(_pg19(), total=count, desc="prepare 32K PG-19"):
            encoded = tokenizer(document.text, add_special_tokens=True, truncation=True,
                                max_length=cfg.max_length, return_offsets_mapping=True)
            if len(encoded["input_ids"]) < cfg.max_length:
                continue
            char_end = encoded["offset_mapping"][-1][1]
            handle.write(json.dumps({"id": f"pg19:{document.doc_id}", "corpus": "pg19",
                                     "text": document.text[:char_end]}, ensure_ascii=False) + "\n")
            written += 1
            if written == count:
                break
    if written != count:
        raise RuntimeError(f"requested {count} long documents, found {written}")
    return output


def _last_categories(text: str, offsets: list[tuple[int, int]], start: int, nlp) -> list[str]:
    char_start = offsets[start][0]
    suffix = text[char_start: offsets[-1][1]]
    categories = ["other"] * (len(offsets) - start)
    for token in nlp(suffix):
        absolute_start = char_start + token.idx
        absolute_end = absolute_start + len(token)
        category = _category(token)
        for index in range(start, len(offsets)):
            left, right = offsets[index]
            if right > absolute_start and left < absolute_end:
                categories[index - start] = category
    return categories


@torch.inference_mode()
def run_long_context(cfg: Config, eval_tokens: int = 128,
                     windows: tuple[int, ...] = (128, 512, 2048),
                     target_token: bool = False) -> Path:
    data_dir = cfg.data_dir or cfg.output_dir
    path = data_dir / "contexts.jsonl"
    if not path.exists():
        raise FileNotFoundError("run context-prepare-32k first")
    model, tokenizer = load_model(cfg.model, cfg.device, cfg.dtype)
    base, _ = decoder_layers(model)
    nlp = load_nlp()
    rows = []
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    for record in tqdm(records, desc="32K context ablation"):
        encoded = tokenizer(record["text"], add_special_tokens=True, truncation=True,
                            max_length=cfg.max_length, return_offsets_mapping=True)
        ids = torch.tensor([encoded["input_ids"]], device=cfg.device)
        length = ids.shape[1]
        if length != cfg.max_length:
            raise RuntimeError(f"{record['id']} has {length}, expected {cfg.max_length}")
        positions = torch.arange(length, device=cfg.device)[None, :]
        full_hidden = base(input_ids=ids, position_ids=positions, use_cache=False,
                           return_dict=True).last_hidden_state[:, -eval_tokens:]
        full_logits = model.lm_head(full_hidden)[0].float().cpu().numpy()
        start = length - eval_tokens
        categories = _last_categories(record["text"], [tuple(x) for x in encoded["offset_mapping"]],
                                      start, nlp)
        for window in windows:
            suffix_ids = ids[:, -window:]
            suffix_positions = positions[:, -window:]
            short_hidden = base(input_ids=suffix_ids, position_ids=suffix_positions,
                                use_cache=False, return_dict=True).last_hidden_state[:, -eval_tokens:]
            short_logits = model.lm_head(short_hidden)[0].float().cpu().numpy()
            for local_index in range(eval_tokens - 1):
                # In target mode, query at i-1 predicts target token x_i. The
                # category is therefore taken from x_i, not from the query.
                query_index = start + local_index
                absolute_index = query_index + 1 if target_token else query_index
                target = int(ids[0, absolute_index if target_token else absolute_index + 1])
                logit_index = local_index
                p = softmax(full_logits[logit_index] - full_logits[logit_index].max())
                q = softmax(short_logits[logit_index] - short_logits[logit_index].max())
                rows.append({
                    "document": record["id"], "corpus": record["corpus"],
                    "absolute_index": absolute_index, "query_index": query_index,
                    "distance_from_end": eval_tokens - local_index,
                    "token_type": categories[absolute_index - start],
                    "token": tokenizer.decode([ids[0, absolute_index]]),
                    "next_token": tokenizer.decode([target]), "window": window,
                    "ce_full": -np.log(max(float(p[target]), 1e-12)),
                    "ce_short": -np.log(max(float(q[target]), 1e-12)),
                    "delta_ce": np.log(max(float(p[target]), 1e-12)) - np.log(max(float(q[target]), 1e-12)),
                    "kl_full_to_short": float(np.sum(p * (np.log(np.maximum(p, 1e-12)) -
                                                           np.log(np.maximum(q, 1e-12))))),
                    "top1_changed": int(np.argmax(p) != np.argmax(q)),
                })
            del short_hidden
        del full_hidden
        torch.cuda.empty_cache()
    suffix = "target" if target_token else "post"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.output_dir / f"long_context_metrics_{suffix}.parquet"
    pd.DataFrame(rows).to_parquet(output, index=False)
    return output
