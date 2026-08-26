from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import Config
from .data import read_contexts
from .metrics import Accumulator, coverage, pair_metrics, top_indices
from .model import decoder_layers, hidden_states, load_model, projected_qk, selected_layers
from .segments import AnnotatedText, annotate, load_nlp


def _token_distributions(q: torch.Tensor, k: torch.Tensor, query_indices: list[int],
                         region: str, local_window: int, sink_tokens: int,
                         segment_starts: dict[int, int]) -> torch.Tensor:
    if not query_indices:
        return torch.empty((0, k.shape[0]), device=q.device)
    q_idx = torch.tensor(query_indices, device=q.device)
    logits = q[q_idx] @ k.T / (q.shape[-1] ** 0.5)
    positions = torch.arange(k.shape[0], device=q.device)[None, :]
    segment_begin = torch.tensor([segment_starts[i] for i in query_indices], device=q.device)[:, None]
    # Use a common candidate set for every token in a segment. Otherwise a
    # later query gains extra candidate keys and token-type comparisons are
    # confounded by position within the segment.
    window_begin = segment_begin - local_window
    if region == "remote":
        valid = (positions >= sink_tokens) & (positions < torch.minimum(segment_begin, window_begin))
    elif region == "local":
        valid = (positions >= torch.maximum(window_begin, torch.full_like(window_begin, sink_tokens))) & (positions < segment_begin)
    else:
        raise ValueError(f"unknown attention region: {region}")
    logits.masked_fill_(~valid, -torch.inf)
    usable = valid.any(dim=1)
    logits[~usable] = 0
    probs = torch.softmax(logits, dim=-1)
    probs[~usable] = 0
    return probs


def _to_blocks(probs: torch.Tensor, block_size: int) -> np.ndarray:
    pad = (-probs.shape[1]) % block_size
    if pad:
        probs = torch.nn.functional.pad(probs, (0, pad))
    blocks = probs.view(probs.shape[0], -1, block_size).sum(-1)
    return blocks.cpu().numpy()


def _pairs(rng: random.Random, indices: list[int], count: int) -> list[tuple[int, int]]:
    if len(indices) < 2:
        return []
    pairs = []
    for _ in range(min(count, len(indices) * (len(indices) - 1) // 2)):
        a, b = rng.sample(indices, 2)
        pairs.append((a, b))
    return pairs


def _analyze_head(acc: Accumulator, ann: AnnotatedText, distributions: dict[int, np.ndarray],
                  query_row: dict[int, int], layer: int, head: int, group: int,
                  region: str, block_size: int, top_ks: list[int], cfg: Config, rng: random.Random) -> None:
    usable = {i for i in query_row if distributions[i].sum() > 0}
    content = {i for i, cat in enumerate(ann.categories) if cat == "content" and i in usable}
    function = {i for i, cat in enumerate(ann.categories) if cat == "function" and i in usable}
    semantic_pairs: list[tuple[int, int]] = []
    boundary_pairs: list[tuple[int, int]] = []
    fixed_pairs: list[tuple[int, int]] = []
    special_pairs = {name: [] for name in ("negation", "pronoun", "question", "punctuation", "number")}
    direct_content: list[tuple[int, list[int]]] = []
    direct_function: list[tuple[int, list[int]]] = []
    for pos, seg in enumerate(ann.segments):
        members = sorted(content & set(range(seg.start, seg.end)))
        segment_function = sorted(function & set(range(seg.start, seg.end)))
        if len(members) >= 2:
            direct_content.extend((i, [j for j in members if j != i]) for i in members)
            direct_function.extend((i, members) for i in segment_function)
        semantic_pairs.extend(_pairs(rng, members, cfg.analysis.pair_samples))
        for name in special_pairs:
            special = [i for i in range(seg.start, seg.end) if i in usable and ann.categories[i] == name]
            if members and special:
                special_pairs[name].append((rng.choice(members), rng.choice(special)))
        if pos + 1 < len(ann.segments):
            right = sorted(content & set(range(ann.segments[pos + 1].start, ann.segments[pos + 1].end)))
            if members and right:
                boundary_pairs.append((members[-1], right[0]))
        length = seg.end - seg.start
        offset = max(1, length // 2)
        fixed = sorted(content & set(range(seg.start + offset, min(seg.end + offset, len(ann.input_ids)))))
        fixed_pairs.extend(_pairs(rng, fixed, cfg.analysis.pair_samples))
    # Pair sampling is a document-level budget. Applying it per segment makes
    # rank metrics dominate runtime and overweights documents with short clauses.
    budget = cfg.analysis.pair_samples
    semantic_pairs = rng.sample(semantic_pairs, min(budget, len(semantic_pairs)))
    boundary_pairs = rng.sample(boundary_pairs, min(budget, len(boundary_pairs)))
    fixed_pairs = rng.sample(fixed_pairs, min(budget, len(fixed_pairs)))
    special_pairs = {name: rng.sample(pairs, min(budget, len(pairs)))
                     for name, pairs in special_pairs.items()}
    direct_content = rng.sample(direct_content, min(budget, len(direct_content)))
    direct_function = rng.sample(direct_function, min(budget, len(direct_function)))
    all_content = sorted(content)
    random_pairs = _pairs(rng, all_content, min(len(semantic_pairs), budget))

    for top_k in top_ks:
        for condition, pairs in [("semantic", semantic_pairs), ("boundary", boundary_pairs),
                                 ("fixed", fixed_pairs), ("random", random_pairs)]:
            for a, b in pairs:
                values = pair_metrics(distributions[a], distributions[b], top_k)
                for metric, value in values.items():
                    acc.add((layer, head, group, region, block_size, top_k, condition, metric), value)
        for name, pairs in special_pairs.items():
            for a, b in pairs:
                for metric, value in pair_metrics(distributions[a], distributions[b], top_k).items():
                    acc.add((layer, head, group, region, block_size, top_k,
                             f"content_vs_{name}", metric), value)

        # Directly test token-level retrieval differences. A content token is
        # compared with a leave-one-out content route; a function token is
        # compared with the content route from the same semantic segment.
        for token_index, reference_indices in direct_content:
            row = distributions[token_index]
            reference = np.stack([distributions[i] for i in reference_indices]).mean(axis=0)
            for metric, value in pair_metrics(row, reference, top_k).items():
                acc.add((layer, head, group, region, block_size, top_k,
                         "content_to_content_route", metric), value)
            independent = row[top_indices(row, top_k)].sum()
            reused = row[top_indices(reference, top_k)].sum()
            acc.add((layer, head, group, region, block_size, top_k,
                     "content_reuse_content_route", "reuse_ratio"),
                    reused / independent if independent else 0.0)
        for token_index, reference_indices in direct_function:
            row = distributions[token_index]
            reference = np.stack([distributions[i] for i in reference_indices]).mean(axis=0)
            conditions = ["function_to_content_route",
                          f"function_{ann.pos_tags[token_index]}_to_content_route"]
            values = pair_metrics(row, reference, top_k)
            independent = row[top_indices(row, top_k)].sum()
            reused = row[top_indices(reference, top_k)].sum()
            for condition in conditions:
                for metric, value in values.items():
                    acc.add((layer, head, group, region, block_size, top_k,
                             condition, metric), value)
                acc.add((layer, head, group, region, block_size, top_k,
                         condition, "reuse_ratio"), reused / independent if independent else 0.0)

        for seg in ann.segments:
            c = sorted(content & set(range(seg.start, seg.end)))
            f = sorted(function & set(range(seg.start, seg.end)))
            if len(c) < 2:
                continue
            for repeat in range(cfg.analysis.split_repeats):
                shuffled = c.copy()
                rng.shuffle(shuffled)
                cut = max(1, len(shuffled) // 2)
                builder, held = shuffled[:cut], shuffled[cut:]
                if not held:
                    continue
                pc = np.stack([distributions[i] for i in builder])
                pt = np.stack([distributions[i] for i in held])
                rc = pc.mean(axis=0)
                rf = np.stack([distributions[i] for i in f]).mean(axis=0) if f else np.zeros_like(rc)
                non_content = [i for i in range(seg.start, seg.end) if i in usable and i not in content]
                all_tokens = builder + non_content
                rall = np.stack([distributions[i] for i in all_tokens]).mean(axis=0)
                for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    route = (1 - lam) * rc + lam * rf
                    acc.add((layer, head, group, region, block_size, top_k, f"lambda_{lam:g}", "coverage"),
                            coverage(route, pt, top_k))
                cov_c = coverage(rc, pt, top_k)
                cov_all = coverage(rall, pt, top_k)
                acc.add((layer, head, group, region, block_size, top_k, "content_vs_all", "delta_pull"), cov_c - cov_all)

            pseg = np.stack([distributions[i] for i in c])
            shared = pseg.mean(axis=0)
            shared_mass = coverage(shared, pseg, top_k)
            independent = float(np.mean([row[top_indices(row, top_k)].sum() for row in pseg]))
            acc.add((layer, head, group, region, block_size, top_k, "semantic_shared", "retained_mass"), shared_mass)
            acc.add((layer, head, group, region, block_size, top_k, "independent", "retained_mass"), independent)
            acc.add((layer, head, group, region, block_size, top_k, "semantic_shared", "retained_ratio"),
                    shared_mass / independent if independent else 0.0)
            supplement = cfg.analysis.token_supplement
            chosen = set(top_indices(shared, top_k))
            supplemented = []
            for row in pseg:
                extra = set(top_indices(row, supplement))
                supplemented.append(row[list(chosen | extra)].sum())
            acc.add((layer, head, group, region, block_size, top_k, "semantic_plus_supplement", "retained_mass"),
                    float(np.mean(supplemented)))
            acc.add((layer, head, group, region, block_size, top_k, "semantic_plus_supplement", "retained_ratio"),
                    float(np.mean(supplemented)) / independent if independent else 0.0)

            evidence = ann.evidence_blocks_by_size[block_size]
            if evidence and seg is ann.segments[-1]:
                segment_tokens = [i for i in range(seg.start, seg.end) if i in query_row]
                pall = np.stack([distributions[i] for i in segment_tokens])
                # Learn a function-token gate on earlier segments without using evidence.
                gate_scores = {alpha: [] for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)}
                for train in ann.segments[:-1]:
                    tc = sorted(content & set(range(train.start, train.end)))
                    tf = sorted(function & set(range(train.start, train.end)))
                    if len(tc) < 2:
                        continue
                    builder, held = tc[::2], tc[1::2]
                    if not held:
                        continue
                    target = np.stack([distributions[i] for i in held])
                    for alpha in gate_scores:
                        route_tokens = builder + tf
                        route_weights = np.array([1.0] * len(builder) + [alpha] * len(tf))
                        if route_weights.sum() > 0:
                            route = np.average(np.stack([distributions[i] for i in route_tokens]), axis=0,
                                               weights=route_weights)
                            gate_scores[alpha].append(coverage(route, target, top_k))
                learned_alpha = max(gate_scores, key=lambda x: np.mean(gate_scores[x]) if gate_scores[x] else -1)
                learned_tokens = c + sorted(function & set(range(seg.start, seg.end)))
                learned_weights = np.array([1.0] * len(c) + [learned_alpha] * (len(learned_tokens) - len(c)))
                routes = {
                    "all_route": pall.mean(axis=0),
                    "content_route": pseg.mean(axis=0),
                    "idf_route": np.average(pseg, axis=0, weights=ann.idf[c]),
                    "learned_route": np.average(np.stack([distributions[i] for i in learned_tokens]), axis=0,
                                                weights=learned_weights),
                }
                for name, route in routes.items():
                    selected = set(top_indices(route, top_k))
                    recall = len(selected & evidence) / len(evidence)
                    acc.add((layer, head, group, region, block_size, top_k, name, "evidence_recall"), recall)
                upper = set().union(*(set(top_indices(row, top_k)) for row in pseg))
                acc.add((layer, head, group, region, block_size, top_k, "independent_upper", "evidence_recall"),
                        len(upper & evidence) / len(evidence))

        for seg in ann.segments:
            length = seg.end - seg.start
            start = seg.start + max(1, length // 2)
            fixed = sorted(content & set(range(start, min(start + length, len(ann.input_ids)))))
            if len(fixed) < 2:
                continue
            pfix = np.stack([distributions[i] for i in fixed])
            fixed_mass = coverage(pfix.mean(axis=0), pfix, top_k)
            fixed_independent = float(np.mean([row[top_indices(row, top_k)].sum() for row in pfix]))
            acc.add((layer, head, group, region, block_size, top_k, "fixed_shared", "retained_mass"), fixed_mass)
            acc.add((layer, head, group, region, block_size, top_k, "fixed_shared", "retained_ratio"),
                    fixed_mass / fixed_independent if fixed_independent else 0.0)


def run(cfg: Config, shard_index: int = 0, num_shards: int = 1) -> Path:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    data_dir = cfg.data_dir or cfg.output_dir
    contexts = data_dir / "contexts.jsonl"
    idf_path = data_dir / "idf.json"
    if not contexts.exists() or not idf_path.exists():
        raise FileNotFoundError("prepared data missing; run the prepare command first")
    idf_map = json.loads(idf_path.read_text(encoding="utf-8"))["idf"]
    model, tokenizer = load_model(cfg.model, cfg.device, cfg.dtype)
    nlp = load_nlp()
    base, layers = decoder_layers(model)
    layer_ids = selected_layers(cfg.analysis.layers, len(layers))
    rows: list[dict] = []
    rng = random.Random(cfg.seed)
    counts = {corpus: 0 for corpus in cfg.corpora}
    selected = []
    for record in read_contexts(contexts):
        corpus = record["corpus"]
        if corpus in counts and counts[corpus] < cfg.corpora[corpus]:
            selected.append(record)
            counts[corpus] += 1
    if counts != cfg.corpora:
        raise RuntimeError(f"requested corpus counts not available: got {counts}")
    records = (record for index, record in enumerate(selected)
               if index % num_shards == shard_index)
    total = (sum(cfg.corpora.values()) + num_shards - 1 - shard_index) // num_shards
    for record in tqdm(records, total=total, desc=f"documents {shard_index + 1}/{num_shards}"):
        ann = annotate(record["text"], tokenizer, nlp, idf_map, cfg.segment.min_tokens,
                       cfg.segment.max_tokens, cfg.max_length, cfg.analysis.block_sizes,
                       record.get("evidence_spans", []))
        if not ann.segments:
            continue
        ids = torch.tensor([ann.input_ids], device=cfg.device)
        position_ids = torch.arange(ids.shape[1], device=cfg.device)[None, :]
        states = hidden_states(model, ids)
        segment_starts = {i: seg.start for seg in ann.segments for i in range(seg.start, seg.end)}
        query_indices = sorted(segment_starts)
        acc = Accumulator()
        for layer in layer_ids:
            q, k = projected_qk(model, layer, states[layer], position_ids,
                                apply_rope=cfg.position_mode == "rope")
            q_heads, kv_heads = q.shape[0], k.shape[0]
            heads_per_group = q_heads // kv_heads
            for head in range(q_heads):
                group = head // heads_per_group
                for region in cfg.analysis.regions:
                    token_probs = _token_distributions(q[head], k[group], query_indices, region,
                                                       cfg.analysis.local_window, cfg.analysis.sink_tokens,
                                                       segment_starts)
                    for block_size in cfg.analysis.block_sizes:
                        matrix = _to_blocks(token_probs, block_size)
                        distribution = {idx: matrix[row] for row, idx in enumerate(query_indices)}
                        _analyze_head(acc, ann, distribution, {x: n for n, x in enumerate(query_indices)},
                                      layer, head, group, region, block_size, cfg.analysis.top_k, cfg, rng)
                    del token_probs
            del q, k
        rows.extend(acc.rows(record["id"], record["corpus"]))
        del states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    name = "document_metrics.parquet" if num_shards == 1 else f"document_metrics-{shard_index:03d}-of-{num_shards:03d}.parquet"
    output = cfg.output_dir / name
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    return output
