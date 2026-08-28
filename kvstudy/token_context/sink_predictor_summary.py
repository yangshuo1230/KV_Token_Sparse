from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config


def write_sink_predictor_summary(cfg: Config) -> Path:
    contrasts = pd.read_csv(cfg.output_dir / "cached_sink_contrasts.csv")
    quality = pd.read_csv(cfg.output_dir / "cached_sink_quality_summary.csv")
    mass = pd.read_csv(cfg.output_dir / "attention_sink_mass_summary.csv")
    predictors = pd.read_csv(cfg.output_dir / "predictor_mechanism_comparison.csv")
    latency = pd.read_csv(cfg.output_dir / "end_to_end_benchmark.csv")

    def contrast(name: str, remote: int, metric: str = "delta_ce") -> pd.Series:
        return contrasts[
            contrasts.contrast.eq(name)
            & contrasts.context_length.eq(32768)
            & contrasts.cache_budget.eq(cfg.context.profile_recent_budget)
            & contrasts.remote_count.eq(remote)
            & contrasts.metric.eq(metric)
        ].iloc[0]

    prefix1 = contrast("prefix_minus_recent", 1)
    random16 = contrast("prefix_minus_random", 16)
    strided16 = contrast("prefix_minus_strided", 16)
    zero4 = contrast("zero_value_minus_prefix", 4)
    recent = quality[
        quality.context_length.eq(32768)
        & quality.cache_budget.eq(cfg.context.profile_recent_budget)
        & quality.policy.eq("recent_only")
    ].iloc[0]
    sink1 = quality[
        quality.context_length.eq(32768)
        & quality.cache_budget.eq(cfg.context.profile_recent_budget)
        & quality.policy.eq("prefix")
        & quality.remote_count.eq(1)
    ].iloc[0]
    strongest = mass[mass.prefix_size.eq(4)].sort_values(
        "attention_mass_mean", ascending=False
    ).iloc[0]
    latency_mean = latency.groupby(["context_length", "policy"]).mean(numeric_only=True)

    need = predictors[
        predictors.target.eq("delta_ce_gt_0.1")
        & predictors.route_fraction.sub(0.25).abs().lt(0.01)
    ].sort_values("auc", ascending=False)
    top = predictors[
        predictors.target.eq("top1_changed")
        & predictors.route_fraction.sub(0.4).abs().lt(0.01)
    ].sort_values("auc", ascending=False)
    predictor_lines = [
        f"| {row.mechanism} | {row.availability} | {row.auc:.3f} | "
        f"{100*row.recall:.1f}% |"
        for row in need.itertuples()
    ]
    latency_lines = []
    for length in (16384, 24576):
        group = latency_mean.loc[length]
        latency_lines.append(
            f"| {length:,} | {group.loc['recent'].mean_speedup_vs_dense:.3f}x | "
            f"{group.loc['sink1_recent'].mean_speedup_vs_dense:.3f}x | "
            f"{group.loc['sink1_copy_fused'].mean_speedup_vs_dense:.3f}x |"
        )

    output = cfg.output_dir / "SINK_AND_PREDICTOR_SUMMARY.md"
    output.write_text(
        "# Attention sink and router mechanisms: consolidated result\n\n"
        "## What was measured\n\n"
        "Eight PG-19 documents were evaluated at 16K, 24K, and 32K. Each policy starts "
        "from the same dense prefill KV cache, then processes 64 teacher-forced decode "
        "queries. KV budgets are fixed at 128, 512, 2,048, or 8,192. Prefix sink policies "
        "are paired against recent-only, equal-count random remote, equal-count strided "
        "remote, and zero-value prefix controls. Confidence intervals bootstrap documents.\n\n"
        "## Sink is real and useful\n\n"
        f"At 32K with a 2,048-position budget, recent-only has mean ΔCE "
        f"{recent.delta_ce:.4f}; prefix-1 + recent-2,047 reduces it to "
        f"{sink1.delta_ce:.4f}. The paired improvement is "
        f"{prefix1.mean_difference:+.4f} (95% CI "
        f"[{prefix1.ci_low:+.4f}, {prefix1.ci_high:+.4f}]). Prefix-16 also beats "
        f"equal random remote by {random16.mean_difference:+.4f} and equal strided "
        f"remote by {strided16.mean_difference:+.4f}. Zeroing prefix values makes ΔCE "
        f"worse by {zero4.mean_difference:+.4f}, so retained values carry functional "
        "state; keys are not merely absorbing softmax probability.\n\n"
        f"The strongest observed case is {int(strongest.context_length):,}-token layer "
        f"{int(strongest.layer)}: its first four "
        f"tokens receive {100*strongest.attention_mass_mean:.2f}% mean attention mass, "
        f"{strongest.concentration_mean:.0f}x the uniform expectation. Prefix-1 captures "
        "nearly all functional benefit, so the recommended cache allocation is one sink "
        "token rather than a 16–128-token sink block. This differs from compact-sequence "
        "recomputation: cached recent K/V retain representations built during dense prefill, "
        "whereas recomputation rebuilds every retained token under the compact context. The "
        "cached experiment is the relevant one for post-prefill KV eviction.\n\n"
        "## Current implementation cost\n\n"
        "| Context | Recent-only | Two-kernel sink-1 | Copy-then-one-kernel sink-1 |\n"
        "|---:|---:|---:|---:|\n" + "\n".join(latency_lines)
        + "\n\nBoth generic implementations are too slow. A deployable path needs one fused "
        "kernel that streams one prefix KV and the contiguous recent window through the "
        "same online softmax without concatenation, a second attention launch, or LSE merge.\n\n"
        "## Predictor mechanisms on the corrected sink-aware baseline\n\n"
        "The target is whether 32K cached decode with prefix-1 + recent-2,047 still has "
        "ΔCE > 0.1 versus dense. Results below use a 25% full-route rate and held-out "
        "documents.\n\n"
        "| Mechanism | Availability | AUC | Recall |\n|---|---|---:|---:|\n"
        + "\n".join(predictor_lines)
        + "\n\nNo pre-forward mechanism is reliable: page retrieval, token/bigram memory, "
        "stateful surprise, embedding, and early hidden all remain near random. The best "
        f"post-forward verifier is speculative confidence (top-1-change AUC "
        f"{top.iloc[0].auc:.3f}, 40% route recall {100*top.iloc[0].recall:.1f}%), but it "
        "requires a full replay and is therefore a quality safeguard rather than a speed "
        "optimization.\n\n"
        "## Engineering decision\n\n"
        "Always retain at least the first KV token. Do not spend a full 128-token page on "
        "sink if the kernel can represent a one-token segment. Keep long-context routing "
        "conservative until a materially better pre-forward signal is found; the immediate "
        "high-confidence optimization target is the fused sink+recent(+sparse-remote) "
        "decode kernel.\n\n"
        "## Limits\n\n"
        "The causal evidence covers Qwen2.5-7B, PG-19, BF16, dense prefill followed by 64 "
        "decode steps, and one accelerator family. It does not yet prove the same magnitude "
        "for chat/code data, other model families, quantized KV, or thousands of streaming "
        "steps. Those are replication targets, not assumptions hidden in the conclusion.\n",
        encoding="utf-8",
    )
    return output
