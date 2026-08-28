from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from ..config import Config
from ..segments import load_nlp
from .categories import annotate_targets
from .sink_report import _complete_shards


def _category_metadata(cfg: Config, metrics: pd.DataFrame) -> pd.DataFrame:
    path = (cfg.data_dir or cfg.output_dir) / "contexts.jsonl"
    records = {row["id"]: row for row in map(json.loads, path.open(encoding="utf-8"))}
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, use_fast=True)
    nlp = load_nlp()
    rows = []
    for document in sorted(metrics.document.unique()):
        for context_length in sorted(metrics.context_length.unique()):
            encoded = tokenizer(
                records[document]["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=int(context_length),
                return_offsets_mapping=True,
            )
            if len(encoded["input_ids"]) != context_length:
                raise RuntimeError("category document is shorter than requested context")
            target_start = int(context_length) - 64
            annotations = annotate_targets(
                records[document]["text"],
                [tuple(value) for value in encoded["offset_mapping"]],
                target_start,
                nlp,
            )
            for step, annotation in enumerate(annotations):
                rows.append({
                    "document": document,
                    "context_length": int(context_length),
                    "target_index": target_start + step,
                    "decode_step": step,
                    "token_id": int(encoded["input_ids"][target_start + step]),
                    "coarse_category": annotation.coarse_category,
                    "fine_category": annotation.fine_category,
                    "pos": annotation.pos,
                    "is_first_subtoken": annotation.is_first_subtoken,
                })
    return pd.DataFrame(rows)


def _bootstrap(values: np.ndarray, seed: int, samples: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _plot_attention_heatmap(joined: pd.DataFrame, output: Path) -> None:
    selected = joined[
        joined.context_length.eq(16384) & joined.prefix_size.eq(4)
    ]
    counts = selected.groupby("fine_category").size()
    categories = counts[counts >= 10].index
    matrix = selected[selected.fine_category.isin(categories)].pivot_table(
        index="fine_category",
        columns="layer",
        values="attention_mass_mean",
        aggfunc="mean",
    )
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(matrix))))
    image = ax.imshow(100 * matrix.to_numpy(), aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Next-token lexical category")
    ax.set_title("16K: attention mass on first 4 sink tokens")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean attention mass (%)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_attention_distributions(joined: pd.DataFrame, output: Path) -> None:
    selected = joined[
        joined.context_length.eq(16384)
        & joined.prefix_size.eq(4)
        & joined.layer.isin([4, 8, 24])
    ]
    categories = ["content", "function", "special", "other"]
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, layer in zip(axes, [4, 8, 24], strict=True):
        layer_frame = selected[selected.layer.eq(layer)]
        arrays = [
            100 * layer_frame[layer_frame.coarse_category.eq(category)].attention_mass_mean
            for category in categories
        ]
        boxes = ax.boxplot(arrays, tick_labels=categories, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Next-token category")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Attention mass on first 4 tokens (%)")
    fig.suptitle("16K sink-attention distribution by next-token category")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_recent_tightening(frame: pd.DataFrame, output: Path) -> None:
    coarse = frame.groupby(
        ["coarse_category", "recent_count"], as_index=False
    ).agg(
        delta_ce=("delta_ce", "mean"),
        need_rate=("delta_ce", lambda values: values.gt(0.1).mean()),
        top1_changed=("top1_changed", "mean"),
    )
    categories = ["content", "function", "special", "other"]
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for category, color in zip(categories, colors, strict=True):
        group = coarse[coarse.coarse_category.eq(category)].sort_values("recent_count")
        axes[0].plot(group.recent_count, group.delta_ce, "o-", label=category, color=color)
        axes[1].plot(group.recent_count, 100 * group.need_rate, "o-", color=color)
        axes[2].plot(group.recent_count, 100 * group.top1_changed, "o-", color=color)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([127, 511, 2047, 8191], ["127", "511", "2,047", "8,191"])
        ax.grid(alpha=0.25)
        ax.set_xlabel("Recent KV tokens (plus prefix-1)")
    axes[0].set_ylabel("Mean ΔCE vs dense")
    axes[1].set_ylabel("Tokens with ΔCE > 0.1 (%)")
    axes[2].set_ylabel("Top-1 changed (%)")
    axes[0].legend(frameon=False)
    fig.suptitle("16K: category sensitivity as the recent window tightens")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_full_kv_coarse(distribution: pd.DataFrame, output: Path) -> None:
    grouped = distribution.groupby(
        ["coarse_category", "layer", "block_index"], as_index=False
    ).agg(
        key_start=("key_start", "mean"),
        attention_mass=("attention_mass_mean", "mean"),
    )
    layers = sorted(grouped.layer.unique())
    categories = ["content", "function", "special", "other"]
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]
    fig, axes = plt.subplots(5, 2, figsize=(15, 17), sharex=True, sharey=True)
    for ax, layer in zip(axes.flat, layers, strict=True):
        layer_frame = grouped[grouped.layer.eq(layer)]
        for category, color in zip(categories, colors, strict=True):
            group = layer_frame[layer_frame.coarse_category.eq(category)].sort_values(
                "block_index"
            )
            ax.plot(
                group.key_start / 1024,
                group.attention_mass.clip(lower=1e-8),
                label=category,
                color=color,
                linewidth=1.2,
            )
        ax.set_yscale("log")
        ax.set_title(f"Layer {layer}")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Absolute KV position (K tokens)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Attention mass per 128-token block")
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("16K full-KV attention distributions by next-token category")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_full_kv_fine_heatmaps(distribution: pd.DataFrame, output: Path) -> None:
    query_counts = distribution.drop_duplicates(
        ["document", "decode_step", "fine_category"]
    ).groupby("fine_category").size()
    categories = query_counts[query_counts >= 10].index
    selected = distribution[distribution.fine_category.isin(categories)]
    grouped = selected.groupby(
        ["fine_category", "layer", "block_index"], as_index=False
    ).concentration_mean.mean()
    ordering = selected.groupby("fine_category").attention_mass_mean.mean().sort_values(
        ascending=False
    ).index
    layers = sorted(grouped.layer.unique())
    fig, axes = plt.subplots(5, 2, figsize=(17, max(17, len(ordering) * 1.1)))
    image = None
    for ax, layer in zip(axes.flat, layers, strict=True):
        matrix = grouped[grouped.layer.eq(layer)].pivot(
            index="fine_category", columns="block_index", values="concentration_mean"
        ).reindex(ordering)
        values = np.log10(matrix.clip(lower=1e-3).to_numpy())
        image = ax.imshow(
            values,
            aspect="auto",
            cmap="coolwarm",
            vmin=-2,
            vmax=3,
            interpolation="nearest",
        )
        ax.set_title(f"Layer {layer}")
        ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
        ticks = [0, 32, 64, 96, 127]
        ax.set_xticks(ticks, ["0", "4K", "8K", "12K", "16K"])
    assert image is not None
    color_axis = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    fig.colorbar(image, cax=color_axis, label="log10(mass / uniform mass)")
    fig.suptitle("16K all-KV attention concentration by fine next-token category")
    fig.subplots_adjust(left=0.15, right=0.88, top=0.96, bottom=0.04, hspace=0.32)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_full_kv_regions(regions: pd.DataFrame, output: Path) -> None:
    categories = ["content", "function", "special", "other"]
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]
    names = ["sink_block_0", "remote_middle", "recent_2k"]
    titles = ["First 128-token block", "Middle remote KV", "Most recent 2K KV"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, region, title in zip(axes, names, titles, strict=True):
        selected = regions[regions.region.eq(region)]
        for category, color in zip(categories, colors, strict=True):
            group = selected[selected.coarse_category.eq(category)].sort_values("layer")
            ax.plot(group.layer, 100 * group.attention_mass, "o-", color=color, label=category)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Attention mass (%)")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("16K attention allocation across full-KV regions by next-token category")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze_sink_categories(cfg: Config) -> tuple[Path, ...]:
    metrics = pd.concat(
        (pd.read_parquet(path) for path in _complete_shards(cfg.output_dir, "cached_sink_metrics")),
        ignore_index=True,
    )
    mass = pd.concat(
        (pd.read_parquet(path) for path in _complete_shards(cfg.output_dir, "attention_sink_mass")),
        ignore_index=True,
    )
    full_distribution = pd.concat(
        (
            pd.read_parquet(path)
            for path in _complete_shards(
                cfg.output_dir, "full_kv_attention_distribution"
            )
        ),
        ignore_index=True,
    )
    metadata = _category_metadata(cfg, metrics)
    sink = metrics[
        metrics.context_length.eq(16384)
        & metrics.policy.eq("prefix")
        & metrics.remote_count.eq(1)
    ].merge(
        metadata,
        on=["document", "context_length", "target_index", "decode_step"],
        validate="many_to_one",
    )
    mass_joined = mass.merge(
        metadata,
        on=["document", "context_length", "decode_step"],
        validate="many_to_one",
    )
    full_joined = full_distribution.merge(
        metadata,
        on=["document", "context_length", "decode_step"],
        validate="many_to_one",
    )

    attention_summary = mass_joined[
        mass_joined.context_length.eq(16384) & mass_joined.prefix_size.eq(4)
    ].groupby(
        ["coarse_category", "fine_category", "layer"], as_index=False
    ).agg(
        tokens=("decode_step", "count"),
        documents=("document", "nunique"),
        attention_mass_mean=("attention_mass_mean", "mean"),
        attention_mass_median=("attention_mass_mean", "median"),
        concentration_mean=("concentration_mean", "mean"),
        contribution_norm_ratio_mean=("contribution_norm_ratio_mean", "mean"),
    )
    category_summary = sink.assign(
        need=sink.delta_ce.gt(0.1).astype(float)
    ).groupby(
        ["coarse_category", "fine_category", "cache_budget", "recent_count"],
        as_index=False,
    ).agg(
        tokens=("delta_ce", "size"),
        documents=("document", "nunique"),
        delta_ce=("delta_ce", "mean"),
        need_rate=("need", "mean"),
        top1_changed=("top1_changed", "mean"),
        kl=("kl_full_to_compact", "mean"),
    )
    full_distribution_summary = full_joined.groupby(
        [
            "coarse_category",
            "fine_category",
            "layer",
            "block_index",
            "key_start",
            "key_end",
        ],
        as_index=False,
    ).agg(
        queries=("decode_step", "count"),
        documents=("document", "nunique"),
        attention_mass_mean=("attention_mass_mean", "mean"),
        attention_mass_median=("attention_mass_mean", "median"),
        concentration_mean=("concentration_mean", "mean"),
    )
    region_frame = full_joined.copy()
    region_frame["region"] = np.select(
        [
            region_frame.block_index.eq(0),
            region_frame.block_index.ge(112),
        ],
        ["sink_block_0", "recent_2k"],
        default="remote_middle",
    )
    query_regions = region_frame.groupby(
        ["document", "decode_step", "coarse_category", "layer", "region"],
        as_index=False,
    ).attention_mass_mean.sum()
    region_summary = query_regions.groupby(
        ["coarse_category", "layer", "region"], as_index=False
    ).agg(
        attention_mass=("attention_mass_mean", "mean"),
        attention_mass_median=("attention_mass_mean", "median"),
        documents=("document", "nunique"),
        queries=("decode_step", "count"),
    )

    paired_rows = []
    sink = sink.assign(need=sink.delta_ce.gt(0.1).astype(float))
    for budget in sorted(sink.cache_budget.unique()):
        group = sink[sink.cache_budget.eq(budget)]
        for metric in ("delta_ce", "need", "top1_changed"):
            wide = group.groupby(["document", "coarse_category"])[metric].mean().unstack()
            if not {"content", "function"} <= set(wide.columns):
                continue
            difference = (wide.content - wide.function).dropna().to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap(difference, cfg.seed + int(budget))
            paired_rows.append({
                "cache_budget": budget,
                "recent_count": budget - 1,
                "metric": metric,
                "content_minus_function": float(difference.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "documents": len(difference),
            })
    paired = pd.DataFrame(paired_rows)
    interaction_rows = []
    for metric in ("delta_ce", "need", "top1_changed"):
        by_doc = sink.groupby(["document", "cache_budget", "coarse_category"])[metric].mean().unstack()
        difference = (by_doc.content - by_doc.function).rename("difference").reset_index()
        wide = difference.pivot(index="document", columns="cache_budget", values="difference")
        interaction = (wide[128] - wide[8192]).dropna().to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap(interaction, cfg.seed + 999)
        interaction_rows.append({
            "metric": metric,
            "difference_at_128_minus_difference_at_8192": float(interaction.mean()),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "documents": len(interaction),
        })
    interactions = pd.DataFrame(interaction_rows)

    attention_csv = cfg.output_dir / "sink_attention_by_category_16k.csv"
    category_csv = cfg.output_dir / "sink_recent_tightening_by_category_16k.csv"
    paired_csv = cfg.output_dir / "sink_category_content_function_16k.csv"
    interaction_csv = cfg.output_dir / "sink_category_interactions_16k.csv"
    full_distribution_csv = cfg.output_dir / "sink_full_kv_distribution_by_category_16k.csv"
    region_csv = cfg.output_dir / "sink_full_kv_regions_by_category_16k.csv"
    heatmap_path = cfg.output_dir / "sink_attention_fine_heatmap_16k.png"
    distribution_path = cfg.output_dir / "sink_attention_coarse_distribution_16k.png"
    tightening_path = cfg.output_dir / "sink_recent_tightening_categories_16k.png"
    full_coarse_path = cfg.output_dir / "sink_full_kv_coarse_distributions_16k.png"
    full_fine_path = cfg.output_dir / "sink_full_kv_fine_heatmaps_16k.png"
    region_path = cfg.output_dir / "sink_full_kv_regions_by_category_16k.png"
    report_path = cfg.output_dir / "SINK_CATEGORY_16K_RESULTS.md"
    attention_summary.to_csv(attention_csv, index=False)
    category_summary.to_csv(category_csv, index=False)
    paired.to_csv(paired_csv, index=False)
    interactions.to_csv(interaction_csv, index=False)
    full_distribution_summary.to_csv(full_distribution_csv, index=False)
    region_summary.to_csv(region_csv, index=False)
    _plot_attention_heatmap(mass_joined, heatmap_path)
    _plot_attention_distributions(mass_joined, distribution_path)
    _plot_recent_tightening(sink, tightening_path)
    _plot_full_kv_coarse(full_joined, full_coarse_path)
    _plot_full_kv_fine_heatmaps(full_joined, full_fine_path)
    _plot_full_kv_regions(region_summary, region_path)

    delta = paired[paired.metric.eq("delta_ce")].sort_values("cache_budget")
    need = paired[paired.metric.eq("need")].sort_values("cache_budget")
    lines = []
    for row in delta.itertuples():
        need_row = need[need.cache_budget.eq(row.cache_budget)].iloc[0]
        lines.append(
            f"| {int(row.recent_count):,} | {row.content_minus_function:+.4f} "
            f"[{row.ci_low:+.4f},{row.ci_high:+.4f}] | "
            f"{100*need_row.content_minus_function:+.1f} pp "
            f"[{100*need_row.ci_low:+.1f},{100*need_row.ci_high:+.1f}] |"
        )
    delta_interaction = interactions[interactions.metric.eq("delta_ce")].iloc[0]
    remote_wide = region_summary[region_summary.region.eq("remote_middle")].pivot(
        index="layer", columns="coarse_category", values="attention_mass"
    )
    remote_difference = 100 * (remote_wide.content - remote_wide.function)
    report_path.write_text(
        "# 16K sink-aware category analysis\n\n"
        "Queries use real cached decode with one prefix sink token and a recent window. "
        "Categories describe the next ground-truth token and are used for analysis only; "
        "they are not available to a causal router before generation.\n\n"
        "## Content versus function as recent KV tightens\n\n"
        "| Recent tokens | Content − function ΔCE (95% CI) | Need-rate difference (95% CI) |\n"
        "|---:|---:|---:|\n" + "\n".join(lines)
        + "\n\nThe difference-of-differences between 127 and 8,191 recent tokens is "
        f"{delta_interaction.difference_at_128_minus_difference_at_8192:+.4f} "
        f"(95% CI [{delta_interaction.ci_low:+.4f}, "
        f"{delta_interaction.ci_high:+.4f}]). A positive value means tightening recent KV "
        "increases the content/function gap.\n\n"
        "## Distribution over the complete KV axis\n\n"
        "The content and function position curves overlap strongly. Content-minus-function "
        "mass on the middle remote region is not consistently positive across layers; it "
        f"ranges from {remote_difference.min():+.2f} to {remote_difference.max():+.2f} "
        "percentage points. Therefore the larger content-word quality loss under tight "
        "recent KV is not explained by a simple global rule that content queries always put "
        "more total mass on remote positions. Selective blocks, head specialization, and "
        "value sensitivity are more plausible explanations.\n\n"
        "## Figures\n\n"
        "PNG figures are generated locally and intentionally excluded from Git.\n\n"
        "- `sink_full_kv_coarse_distributions_16k.png`: all 128 KV blocks, coarse categories, and all profiled layers.\n"
        "- `sink_full_kv_fine_heatmaps_16k.png`: all KV blocks × fine lexical categories for each layer.\n"
        "- `sink_full_kv_regions_by_category_16k.png`: sink block, middle remote, and recent-2K mass by layer.\n"
        "- `sink_attention_fine_heatmap_16k.png`: sink-prefix detail by fine category.\n"
        "- `sink_attention_coarse_distribution_16k.png`: sink-prefix per-query distributions at layers 4, 8, and 24.\n"
        "- `sink_recent_tightening_categories_16k.png`: ΔCE, need rate, and top-1 change versus recent size.\n\n"
        "All confidence intervals resample the eight documents, not individual adjacent tokens.\n",
        encoding="utf-8",
    )
    return (
        attention_csv,
        category_csv,
        paired_csv,
        interaction_csv,
        full_distribution_csv,
        region_csv,
        heatmap_path,
        distribution_path,
        tightening_path,
        full_coarse_path,
        full_fine_path,
        region_path,
        report_path,
    )
