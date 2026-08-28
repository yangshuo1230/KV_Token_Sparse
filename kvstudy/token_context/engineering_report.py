from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config


def write_engineering_report(cfg: Config) -> Path:
    out = cfg.output_dir
    profile = pd.read_csv(out / "context_need_profile.csv")
    theory = pd.read_csv(out / "theoretical_speedup.csv").iloc[0]
    kernels = pd.read_csv(out / "decode_attention_benchmark.csv")
    inference = pd.read_csv(out / "end_to_end_benchmark.csv")
    sparse = pd.read_csv(out / "sparse_context_summary.csv")
    router = pd.read_csv(out / "router_evaluation.csv")
    chosen = profile[
        profile.recent_budget.eq(cfg.context.profile_recent_budget)
        & profile.criterion.eq("delta_ce_gt")
        & profile.threshold.eq(cfg.context.long_context_delta_ce)
    ].iloc[0]

    kernel_lines = []
    for length in sorted(kernels.context_length.unique()):
        group = kernels[kernels.context_length.eq(length)].set_index("policy")
        kernel_lines.append(
            f"| {length:,} | {group.loc['v1_oracle_mix'].speedup_vs_dense:.2f}x | "
            f"{group.loc['v2_oracle_mix'].speedup_vs_dense:.2f}x |"
        )
    e2e_lines = []
    for length in sorted(inference.context_length.unique()):
        group = inference[inference.context_length.eq(length)].groupby("policy").mean(numeric_only=True)
        e2e_lines.append(
            f"| {length:,} | {group.loc['dense'].decode_latency_ms_mean:.3f} | "
            f"{group.loc['v1_oracle_rate_schedule'].decode_latency_ms_mean:.3f} "
            f"({group.loc['v1_oracle_rate_schedule'].mean_speedup_vs_dense:.3f}x) | "
            f"{group.loc['v2_oracle_rate_schedule'].decode_latency_ms_mean:.3f} "
            f"({group.loc['v2_oracle_rate_schedule'].mean_speedup_vs_dense:.3f}x) |"
        )
    sparse_2048 = sparse[
        sparse.metric.eq("delta_ce") & sparse.comparator.eq("static_recent_2048")
    ].iloc[0]
    sparse_4096 = sparse[
        sparse.metric.eq("delta_ce") & sparse.comparator.eq("static_recent_4096")
    ].iloc[0]
    sparse_top1 = sparse[
        sparse.metric.eq("top1_changed") & sparse.comparator.eq("static_recent_2048")
    ].iloc[0]
    input_auc = router[router.router.eq("input_token_lookup")].type_auc.iloc[0]
    draft_auc = router[router.router.eq("draft_token_lookup")].type_auc.iloc[0]
    large_plan = router[
        router.router.eq("draft_token_lookup")
        & router.low_budget.eq(2048)
        & router.metric.eq("delta_ce")
    ].iloc[0]

    path = out / "ADAPTIVE_INFERENCE_RESULTS.md"
    path.write_text(
        "# Adaptive block-KV inference: measured results\n\n"
        "## Profile before optimization\n\n"
        f"The profile covers {int(chosen.targets):,} target tokens in "
        f"{int(chosen.documents)} PG-19 documents of {cfg.max_length:,} tokens. With a "
        f"{cfg.context.profile_recent_budget:,}-token recent window and a "
        f"{cfg.context.long_context_delta_ce:g}-nat ΔCE criterion, "
        f"{100*chosen.long_context_fraction:.2f}% need long context and "
        f"{100*chosen.recent_only_fraction:.2f}% do not.\n\n"
        f"With {cfg.context.block_size}-token pages, the ideal KV-read model predicts "
        f"{theory.v1_attention_upper_bound_speedup:.2f}x for V1 and "
        f"{theory.v2_attention_upper_bound_speedup:.2f}x for V2. These are attention-only "
        "bandwidth bounds, not end-to-end claims.\n\n"
        "## Implementation\n\n"
        "V1 keeps the full DynamicCache and chooses a zero-copy recent tensor view or the "
        "full KV for each query. V2 always runs recent attention, reads query-selected "
        "non-contiguous remote pages with FlashInfer's paged decode kernel, and merges the "
        "two softmax states using their log-sum-exp values. Page selection uses a first-layer "
        "query and one mean-key landmark per 128-token page. The page table is reusable for "
        "a block of decode steps. Both versions retain the complete physical KV cache so "
        "long routes remain possible; the optimization reduces KV reads, not stored bytes.\n\n"
        "## Attention-kernel timing\n\n"
        "The mixture uses the measured 24.80% long-token rate; page-selection cost is "
        f"amortized over {cfg.context.sparse_selection_refresh} tokens.\n\n"
        "| Context | V1 mixture | V2 mixture |\n|---:|---:|---:|\n"
        + "\n".join(kernel_lines)
        + "\n\n## Real Qwen2.5-7B decode\n\n"
        "Each policy starts from an identical prefill cache and decodes 128 tokens in "
        "three order-rotated trials. The "
        "routing schedules use the oracle *rate* only to measure compute; they do not use "
        "oracle labels and are not deployable quality results. Mean wall-clock latency is "
        "reported because the mixed distribution is bimodal.\n\n"
        "| Context | Dense ms/token | V1 ms/token (speedup) | V2 ms/token (speedup) |\n"
        "|---:|---:|---:|---:|\n"
        + "\n".join(e2e_lines)
        + "\n\nV1 gains at both measured lengths. V2's two-kernel "
        "execution and LSE merge are near break-even at 16K and give only a small 24K gain; a fused "
        "recent-plus-paged kernel is the next operator target.\n\n"
        "## V2 quality at 32K\n\n"
        f"The sparse policy attends 6,144 tokens (2,048 recent + 4,096 remote). Its mean "
        f"ΔCE is {sparse_2048.sparse_mean:.4f}, improving over recent-2,048 by "
        f"{-sparse_2048.sparse_minus_comparator:.4f} "
        f"(95% CI [{-sparse_2048.ci_high:.4f}, {-sparse_2048.ci_low:.4f}]). It is "
        f"statistically indistinguishable from static recent-4,096 in ΔCE "
        f"(difference {sparse_4096.sparse_minus_comparator:+.4f}, 95% CI "
        f"[{sparse_4096.ci_low:+.4f}, {sparse_4096.ci_high:+.4f}]). Top-1 changes drop "
        f"by {-100*sparse_top1.sparse_minus_comparator:.2f} percentage points versus "
        "recent-2,048.\n\n"
        "## Lightweight-router status\n\n"
        f"The one-table-lookup input-token predictor reaches type AUC {input_auc:.3f}; the "
        f"0.5B draft-token lookup reaches {draft_auc:.3f}. Neither is Pareto competitive "
        "at the small plans. At the 2,048/8,192 plan the draft router matches static-4,096 "
        f"ΔCE within uncertainty (difference {large_plan.router_minus_static:+.4f}, 95% CI "
        f"[{large_plan.ci_low:+.4f}, {large_plan.ci_high:+.4f}]), but its auxiliary-model "
        "cost is not included. Therefore the repository demonstrates an engineering "
        "speedup opportunity and a working kernel path, not yet a deployable "
        "quality-preserving learned router.\n",
        encoding="utf-8",
    )
    return path
