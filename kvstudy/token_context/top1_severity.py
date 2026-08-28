from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

from ..config import Config
from ..model import load_model
from .experiment import _distribution_metrics
from .kv_cache import clone_dynamic_cache
from .routed_inference import (
    FixedRemoteDecodeController,
    configure_fixed_remote_route,
    configure_v1_route,
    enable_routed_decode,
)
from .sink_cached_experiment import _decode_teacher_forced
from .sink_category_analysis import _category_metadata
from .sink_report import _complete_shards


@torch.inference_mode()
def run_top1_severity_experiment(
    cfg: Config,
    context_length: int = 16384,
    documents: int = 8,
    eval_tokens: int = 64,
    budgets: list[int] | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Path:
    """Capture actual dense/compact top-1 tokens for severity analysis."""
    budgets = budgets or [128, 512, 2048, 8192]
    path = (cfg.data_dir or cfg.output_dir) / "contexts.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8")][:documents]
    records = records[shard_index::num_shards]
    model, tokenizer = load_model(cfg.model, cfg.device, cfg.dtype)
    embedding = model.get_input_embeddings().weight
    rows: list[dict] = []

    for record in tqdm(records, desc=f"top1 severity {shard_index + 1}/{num_shards}"):
        model.set_attn_implementation("sdpa")
        ids = tokenizer(
            record["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
        ).input_ids.to(cfg.device)
        target_start = context_length - eval_tokens
        prefill_length = target_start - 1
        prefill = model(input_ids=ids[:, :prefill_length], use_cache=True, return_dict=True)
        original_cache = prefill.past_key_values
        queries = ids[:, prefill_length : context_length - 1]
        targets = ids[0, target_start:context_length]
        enable_routed_decode(model)
        dense_cache = clone_dynamic_cache(original_cache)
        configure_v1_route(model, recent_budget=max(budgets), use_long_context=True)
        full_logits = _decode_teacher_forced(model, dense_cache, queries, prefill_length)
        del dense_cache
        full_probability = torch.softmax(full_logits.float(), dim=-1)
        full_values, full_ids = full_probability.topk(2, dim=-1)
        target_rows = torch.arange(eval_tokens, device=cfg.device)
        full_target_logits = full_logits[target_rows, targets]

        for budget in budgets:
            cache = clone_dynamic_cache(original_cache)
            controller = FixedRemoteDecodeController(
                torch.tensor([0], device=cfg.device),
                recent_tokens=budget - 1,
            )
            configure_fixed_remote_route(model, controller)
            compact_logits = _decode_teacher_forced(model, cache, queries, prefill_length)
            compact_probability = torch.softmax(compact_logits.float(), dim=-1)
            compact_values, compact_ids = compact_probability.topk(2, dim=-1)
            compact_target_logits = compact_logits[target_rows, targets]
            metrics = _distribution_metrics(full_logits, compact_logits, targets)
            full_vectors = F.normalize(embedding[full_ids[:, 0]].float(), dim=-1)
            compact_vectors = F.normalize(embedding[compact_ids[:, 0]].float(), dim=-1)
            target_vectors = F.normalize(embedding[targets].float(), dim=-1)
            pair_cosine = (full_vectors * compact_vectors).sum(dim=-1)
            full_target_cosine = (full_vectors * target_vectors).sum(dim=-1)
            compact_target_cosine = (compact_vectors * target_vectors).sum(dim=-1)
            full_ranks = full_logits.gt(full_target_logits[:, None]).sum(dim=-1) + 1
            compact_ranks = compact_logits.gt(compact_target_logits[:, None]).sum(dim=-1) + 1

            for offset in range(eval_tokens):
                full_id = int(full_ids[offset, 0])
                compact_id = int(compact_ids[offset, 0])
                target_id = int(targets[offset])
                rows.append({
                    "document": record["id"],
                    "context_length": context_length,
                    "target_index": target_start + offset,
                    "decode_step": offset,
                    "cache_budget": budget,
                    "recent_count": budget - 1,
                    "target_token_id": target_id,
                    "target_token": tokenizer.decode([target_id], clean_up_tokenization_spaces=False),
                    "full_top1_id": full_id,
                    "full_top1_token": tokenizer.decode([full_id], clean_up_tokenization_spaces=False),
                    "compact_top1_id": compact_id,
                    "compact_top1_token": tokenizer.decode(
                        [compact_id], clean_up_tokenization_spaces=False
                    ),
                    "full_top1_probability": float(full_values[offset, 0]),
                    "compact_top1_probability": float(compact_values[offset, 0]),
                    "full_margin": float(full_values[offset, 0] - full_values[offset, 1]),
                    "compact_margin": float(
                        compact_values[offset, 0] - compact_values[offset, 1]
                    ),
                    "full_correct": full_id == target_id,
                    "compact_correct": compact_id == target_id,
                    "full_target_rank": int(full_ranks[offset]),
                    "compact_target_rank": int(compact_ranks[offset]),
                    "top1_embedding_cosine": float(pair_cosine[offset]),
                    "full_target_embedding_cosine": float(full_target_cosine[offset]),
                    "compact_target_embedding_cosine": float(compact_target_cosine[offset]),
                    **{name: values[offset] for name, values in metrics.items()},
                })
            del cache, compact_logits, compact_probability
        del original_cache, prefill, full_logits, full_probability, ids
        torch.cuda.empty_cache()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    name = (
        "top1_severity_metrics.parquet"
        if num_shards == 1
        else f"top1_severity_metrics-{shard_index:03d}-of-{num_shards:03d}.parquet"
    )
    output = cfg.output_dir / name
    pd.DataFrame(rows).to_parquet(output, index=False)
    return output


def _normalize_surface(text: str) -> str:
    text = text.replace("Ġ", " ").replace("▁", " ")
    return re.sub(r"\s+", "", text).casefold()


def _punctuation_or_space(text: str) -> bool:
    return bool(text) and all(
        character.isspace() or unicodedata.category(character)[0] in {"P", "S"}
        for character in text
    )


def _severity(row: pd.Series) -> str:
    if not row.top1_changed:
        return "unchanged"
    if row.full_correct and not row.compact_correct:
        return "definite_harm_full_correct_lost"
    if not row.full_correct and row.compact_correct:
        return "definite_benefit_compact_correct"
    if (
        _normalize_surface(row.full_top1_token)
        == _normalize_surface(row.compact_top1_token)
        or (
            _punctuation_or_space(row.full_top1_token)
            and _punctuation_or_space(row.compact_top1_token)
        )
    ):
        return "surface_or_punctuation"
    if row.top1_embedding_cosine >= 0.8:
        return "embedding_near_cosine_ge_0.8"
    return "potential_semantic_change"


def summarize_top1_severity(cfg: Config) -> tuple[Path, ...]:
    frame = pd.concat(
        (
            pd.read_parquet(path)
            for path in _complete_shards(cfg.output_dir, "top1_severity_metrics")
        ),
        ignore_index=True,
    )
    metadata = _category_metadata(cfg, frame)
    frame = frame.merge(
        metadata,
        on=["document", "context_length", "target_index", "decode_step"],
        validate="many_to_one",
    )
    frame["severity"] = frame.apply(_severity, axis=1)
    frame["probabilistic_harm"] = frame.delta_ce.gt(0.1)
    changed = frame[frame.top1_changed].copy()

    summary = frame.groupby(["cache_budget", "recent_count"], as_index=False).agg(
        tokens=("target_index", "size"),
        top1_change_rate=("top1_changed", "mean"),
        full_accuracy=("full_correct", "mean"),
        compact_accuracy=("compact_correct", "mean"),
        definite_harm_rate=(
            "severity",
            lambda values: values.eq("definite_harm_full_correct_lost").mean(),
        ),
        definite_benefit_rate=(
            "severity",
            lambda values: values.eq("definite_benefit_compact_correct").mean(),
        ),
        probabilistic_harm_rate=("probabilistic_harm", "mean"),
        mean_delta_ce=("delta_ce", "mean"),
    )
    severity = changed.groupby(
        ["cache_budget", "recent_count", "coarse_category", "severity"],
        as_index=False,
    ).agg(
        changes=("target_index", "size"),
        documents=("document", "nunique"),
        mean_delta_ce=("delta_ce", "mean"),
        mean_embedding_cosine=("top1_embedding_cosine", "mean"),
        probabilistic_harm_rate=("probabilistic_harm", "mean"),
    )
    totals = severity.groupby(
        ["cache_budget", "coarse_category"]
    ).changes.transform("sum")
    severity["fraction_of_changes"] = severity.changes / totals
    category_outcomes = frame.assign(
        definite_harm=frame.severity.eq("definite_harm_full_correct_lost"),
        definite_benefit=frame.severity.eq("definite_benefit_compact_correct"),
    ).groupby(
        ["cache_budget", "recent_count", "coarse_category"], as_index=False
    ).agg(
        tokens=("target_index", "size"),
        top1_change_rate=("top1_changed", "mean"),
        definite_harm_rate=("definite_harm", "mean"),
        definite_benefit_rate=("definite_benefit", "mean"),
        probabilistic_harm_rate=("probabilistic_harm", "mean"),
        mean_delta_ce=("delta_ce", "mean"),
    )
    thresholds = []
    for budget, group in changed.groupby("cache_budget"):
        thresholds.append({
            "cache_budget": budget,
            "changed_tokens": len(group),
            "cosine_lt_0.5": group.top1_embedding_cosine.lt(0.5).mean(),
            "cosine_lt_0.7": group.top1_embedding_cosine.lt(0.7).mean(),
            "cosine_lt_0.8": group.top1_embedding_cosine.lt(0.8).mean(),
            "delta_ce_gt_0.1": group.delta_ce.gt(0.1).mean(),
            "delta_ce_gt_0.5": group.delta_ce.gt(0.5).mean(),
        })
    threshold_frame = pd.DataFrame(thresholds)

    summary_path = cfg.output_dir / "top1_change_severity_summary.csv"
    severity_path = cfg.output_dir / "top1_change_severity_by_category.csv"
    threshold_path = cfg.output_dir / "top1_change_semantic_thresholds.csv"
    category_path = cfg.output_dir / "top1_change_outcomes_by_category.csv"
    examples_path = cfg.output_dir / "top1_change_examples.csv"
    report_path = cfg.output_dir / "TOP1_CHANGE_SEVERITY_RESULTS.md"
    summary.to_csv(summary_path, index=False)
    severity.to_csv(severity_path, index=False)
    threshold_frame.to_csv(threshold_path, index=False)
    category_outcomes.to_csv(category_path, index=False)
    changed[[
        "document",
        "target_index",
        "recent_count",
        "coarse_category",
        "fine_category",
        "target_token",
        "full_top1_token",
        "compact_top1_token",
        "severity",
        "delta_ce",
        "top1_embedding_cosine",
        "full_correct",
        "compact_correct",
        "full_target_rank",
        "compact_target_rank",
    ]].sort_values(
        ["recent_count", "severity", "document", "target_index"]
    ).to_csv(examples_path, index=False)
    lines = []
    for row in summary.itertuples():
        changed_group = changed[changed.cache_budget.eq(row.cache_budget)]
        surface = changed_group.severity.eq("surface_or_punctuation").mean()
        near = changed_group.severity.eq("embedding_near_cosine_ge_0.8").mean()
        semantic = changed_group.severity.eq("potential_semantic_change").mean()
        harmful_changed = changed_group.severity.eq(
            "definite_harm_full_correct_lost"
        ).mean()
        beneficial_changed = changed_group.severity.eq(
            "definite_benefit_compact_correct"
        ).mean()
        lines.append(
            f"| {int(row.recent_count):,} | {100*row.top1_change_rate:.1f}% | "
            f"{100*harmful_changed:.1f}% | {100*beneficial_changed:.1f}% | "
            f"{100*surface:.1f}% | {100*near:.1f}% | {100*semantic:.1f}% |"
        )
    category_lines = []
    for recent_count in (127, 2047, 8191):
        selected = category_outcomes[category_outcomes.recent_count.eq(recent_count)].set_index(
            "coarse_category"
        )
        category_lines.append(
            f"| {recent_count:,} | {100*selected.loc['content'].top1_change_rate:.1f}% | "
            f"{100*selected.loc['content'].definite_harm_rate:.1f}% | "
            f"{100*selected.loc['function'].top1_change_rate:.1f}% | "
            f"{100*selected.loc['function'].definite_harm_rate:.1f}% |"
        )
    threshold_lines = [
        f"| {int(row.cache_budget - 1):,} | {100*row.delta_ce_gt_0_1:.1f}% | "
        f"{100*row.delta_ce_gt_0_5:.1f}% | {100*row.cosine_lt_0_5:.1f}% |"
        for row in threshold_frame.rename(columns={
            "delta_ce_gt_0.1": "delta_ce_gt_0_1",
            "delta_ce_gt_0.5": "delta_ce_gt_0_5",
            "cosine_lt_0.5": "cosine_lt_0_5",
        }).itertuples()
    ]
    report_path.write_text(
        "# What kind of Top-1 changes occur?\n\n"
        "This analysis compares dense and prefix-1 + recent cached decode at 16K. "
        "`Definite harm` means dense Top-1 equals the ground-truth token but compact Top-1 "
        "does not; `definite benefit` is the reverse. Surface/punctuation changes normalize "
        "whitespace and case. `Embedding-near` uses Qwen input-embedding cosine >= 0.8. "
        "The remaining bucket is a potential semantic change, not a human semantic-error "
        "judgment. Percentages after the first column are fractions of changed tokens.\n\n"
        "| Recent | Top-1 change rate | Definite harm | Definite benefit | Surface/punct | Embedding-near | Potential semantic |\n"
        "|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(lines)
        + "\n\n## Content versus function outcomes\n\n"
        "Rates below use all tokens in that category, rather than only changed tokens.\n\n"
        "| Recent | Content changed | Content definite harm | Function changed | Function definite harm |\n"
        "|---:|---:|---:|---:|---:|\n" + "\n".join(category_lines)
        + "\n\n## Severity diagnostics among changed tokens\n\n"
        "| Recent | ΔCE > 0.1 | ΔCE > 0.5 | Embedding cosine < 0.5 |\n"
        "|---:|---:|---:|---:|\n" + "\n".join(threshold_lines)
        + "\n\nGround-truth correctness transitions are the strongest evidence. Embedding "
        "similarity is only a diagnostic proxy; single-token substitutions can change "
        "meaning even at high cosine, while two different tokens can be equivalent in a "
        "larger phrase. Severity buckets are assigned in priority order: correctness "
        "transition, surface/punctuation, then embedding similarity. CSV files include "
        "target ranks, probabilities, ΔCE, category, cosine-threshold sensitivity, and all "
        "changed token triples for independent inspection.\n",
        encoding="utf-8",
    )
    return (
        summary_path,
        severity_path,
        threshold_path,
        category_path,
        examples_path,
        report_path,
    )
