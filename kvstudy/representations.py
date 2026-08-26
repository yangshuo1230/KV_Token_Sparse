from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from .config import Config
from .data import read_contexts
from .model import decoder_layers, hidden_states, load_model, projected_qk, selected_layers
from .segments import CONTENT_POS, FUNCTION_POS, annotate, load_nlp


@dataclass(frozen=True)
class Word:
    index: int
    lemma: str
    text: str
    pos: str


def _words(text: str, offsets: list[tuple[int, int]], nlp) -> tuple[list[Word], list[Word]]:
    content, function = [], []
    for token in nlp(text[: offsets[-1][1]]):
        indices = [i for i, (a, b) in enumerate(offsets)
                   if b > token.idx and a < token.idx + len(token)]
        if not indices or not token.is_alpha:
            continue
        word = Word(indices[-1], token.lemma_.lower(), token.text, token.pos_)
        if token.pos_ in CONTENT_POS:
            content.append(word)
        elif token.pos_ in FUNCTION_POS:
            function.append(word)
    return content, function


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(cosine_similarity(a.float(), b.float(), dim=-1).mean().cpu())


def _top_jaccard(q: torch.Tensor, k: torch.Tensor, a: int, b: int,
                 end: int, top_k: int = 10) -> float:
    if end <= 4:
        return np.nan
    candidates = torch.arange(4, end, device=q.device)
    overlaps = []
    heads_per_group = q.shape[0] // k.shape[0]
    for head in range(q.shape[0]):
        group = head // heads_per_group
        left = q[head, a] @ k[group, candidates].T
        right = q[head, b] @ k[group, candidates].T
        size = min(top_k, candidates.numel())
        li = set(torch.topk(left, size).indices.cpu().tolist())
        ri = set(torch.topk(right, size).indices.cpu().tolist())
        overlaps.append(len(li & ri) / len(li | ri))
    return float(np.mean(overlaps))


def _sample_pairs(ann, content: list[Word], function: list[Word], rng: random.Random,
                  budget: int) -> dict[str, list[tuple[Word, Word, int]]]:
    pairs: dict[str, list[tuple[Word, Word, int]]] = defaultdict(list)
    for seg in ann.segments:
        c = [x for x in content if seg.start <= x.index < seg.end]
        f = [x for x in function if seg.start <= x.index < seg.end]
        content_pairs = list(itertools.combinations(c, 2))
        if c and content_pairs:
            for word in f:
                nearest = min(c, key=lambda x: abs(x.index - word.index))
                distance = abs(nearest.index - word.index)
                matched = min(content_pairs,
                              key=lambda pair: abs(abs(pair[1].index - pair[0].index) - distance))
                pairs["function_content"].append((word, nearest, seg.start - 128))
                pairs["content_content"].append((*matched, seg.start - 128))

    by_lemma: dict[str, list[Word]] = defaultdict(list)
    for word in content:
        by_lemma[word.lemma].append(word)
    all_content = sorted(content, key=lambda x: x.index)
    for occurrences in by_lemma.values():
        if len(occurrences) < 2:
            continue
        for first, second in itertools.combinations(occurrences, 2):
            if second.index - first.index < 10 or first.index <= 132:
                continue
            end = first.index - 128
            pairs["repeated_content"].append((first, second, end))
            controls = [x for x in all_content
                        if x.lemma != first.lemma and abs(x.index - second.index) <= 32]
            if controls:
                control = min(controls, key=lambda x: abs(x.index - second.index))
                pairs["position_matched_different_content"].append((first, control, end))

    return {name: rng.sample(values, min(budget, len(values)))
            for name, values in pairs.items()}


def diagnose_representations(cfg: Config, budget: int = 32) -> Path:
    data_dir = cfg.data_dir or cfg.output_dir
    idf = json.loads((data_dir / "idf.json").read_text(encoding="utf-8"))["idf"]
    model, tokenizer = load_model(cfg.model, cfg.device, cfg.dtype)
    nlp = load_nlp()
    _, layers = decoder_layers(model)
    layer_ids = selected_layers(cfg.analysis.layers, len(layers))
    rng = random.Random(cfg.seed)

    counts = {corpus: 0 for corpus in cfg.corpora}
    records = []
    for record in read_contexts(data_dir / "contexts.jsonl"):
        corpus = record["corpus"]
        if corpus in counts and counts[corpus] < cfg.corpora[corpus]:
            records.append(record)
            counts[corpus] += 1

    rows, examples = [], []
    for record in tqdm(records, desc="representation diagnostics"):
        ann = annotate(record["text"], tokenizer, nlp, idf, cfg.segment.min_tokens,
                       cfg.segment.max_tokens, cfg.max_length, cfg.analysis.block_sizes,
                       record.get("evidence_spans", []))
        content, function = _words(record["text"], ann.offsets, nlp)
        pairs = _sample_pairs(ann, content, function, rng, budget)
        ids = torch.tensor([ann.input_ids], device=cfg.device)
        positions = torch.arange(ids.shape[1], device=cfg.device)[None, :]
        states = hidden_states(model, ids)
        for layer in layer_ids:
            raw_q, _ = projected_qk(model, layer, states[layer], positions, apply_rope=False)
            rope_q, rope_k = projected_qk(model, layer, states[layer], positions, apply_rope=True)
            for pair_type, values in pairs.items():
                metrics: dict[str, list[float]] = defaultdict(list)
                for left, right, candidate_end in values:
                    metrics["h_cosine"].append(_cos(states[layer][0, left.index],
                                                     states[layer][0, right.index]))
                    metrics["raw_q_cosine"].append(_cos(raw_q[:, left.index], raw_q[:, right.index]))
                    metrics["rope_q_cosine"].append(_cos(rope_q[:, left.index], rope_q[:, right.index]))
                    metrics["token_top10_jaccard"].append(
                        _top_jaccard(rope_q, rope_k, left.index, right.index, candidate_end)
                    )
                    if pair_type == "repeated_content" and len(examples) < 100:
                        examples.append({"document": record["id"], "layer": layer,
                                         "word": left.lemma, "left": left.index,
                                         "right": right.index, "distance": right.index - left.index})
                for metric, metric_values in metrics.items():
                    finite = [x for x in metric_values if np.isfinite(x)]
                    if finite:
                        rows.append({"document": record["id"], "corpus": record["corpus"],
                                     "layer": layer, "pair_type": pair_type, "metric": metric,
                                     "value": float(np.mean(finite)), "pairs": len(finite)})
            del raw_q, rope_q, rope_k
        del states
        torch.cuda.empty_cache()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.output_dir / "representation_metrics.parquet"
    pd.DataFrame(rows).to_parquet(output, index=False)
    (cfg.output_dir / "repeated_word_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output
