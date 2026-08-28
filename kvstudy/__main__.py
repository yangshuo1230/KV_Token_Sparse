from __future__ import annotations

import argparse

from .config import load_config
from .data import prepare
from .semantic.experiment import run as run_semantic
from .semantic.report import summarize as summarize_semantic
from .semantic.representations import diagnose_representations
from .token_context.ablation import diagnose_context_length
from .token_context.attention_benchmark import benchmark_decode_attention
from .token_context.benchmark import benchmark_cache
from .token_context.experiment import prepare_contexts, run_context_ablation
from .token_context.inference_benchmark import benchmark_v1_inference
from .token_context.engineering_report import write_engineering_report
from .token_context.profile import profile_context_need
from .token_context.predictor_mechanisms import compare_predictor_mechanisms
from .token_context.report import summarize_context
from .token_context.router import evaluate_router
from .token_context.router_exploration import explore_lightweight_routers
from .token_context.sparse_experiment import run_sparse_context_ablation
from .token_context.sparse_report import summarize_sparse_context
from .token_context.sink_cached_experiment import run_cached_sink_experiment
from .token_context.sink_report import summarize_attention_sink
from .token_context.sink_predictor_summary import write_sink_predictor_summary
from .token_context.sink_category_analysis import analyze_sink_categories
from .token_context.full_kv_distribution import (
    run_full_kv_distribution,
    run_head_resolved_distribution,
)
from .token_context.top1_severity import (
    run_top1_severity_experiment,
    summarize_top1_severity,
)
from .token_context.fine_attention_analysis import analyze_fine_attention


COMMANDS = (
    "prepare",
    "semantic-run",
    "semantic-summarize",
    "semantic-diagnose-representations",
    "context-diagnose-2k",
    "context-prepare",
    "context-run",
    "context-summarize",
    "context-profile-need",
    "context-evaluate-router",
    "context-benchmark-cache",
    "context-benchmark-attention",
    "context-benchmark-v1-inference",
    "context-benchmark-inference",
    "context-run-sparse",
    "context-summarize-sparse",
    "context-report-engineering",
    "context-explore-router",
    "context-run-cached-sink",
    "context-summarize-sink",
    "context-compare-predictors",
    "context-report-sink-predictors",
    "context-analyze-sink-categories",
    "context-run-full-kv-distribution",
    "context-run-head-resolved-distribution",
    "context-run-top1-severity",
    "context-summarize-top1-severity",
    "context-analyze-fine-attention",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic KV routing and token-specific context experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "context-evaluate-router":
            command.add_argument("--draft-config", required=True)
        if name == "context-benchmark-cache":
            command.add_argument("--prefill-tokens", type=int, default=8192)
            command.add_argument("--repeats", type=int, default=20)
        if name == "context-benchmark-attention":
            command.add_argument("--context-lengths", type=int, nargs="+", default=[16384, 24576])
            command.add_argument("--repeats", type=int, default=100)
        if name in {"context-benchmark-v1-inference", "context-benchmark-inference"}:
            command.add_argument("--context-lengths", type=int, nargs="+", default=[16384, 24576])
            command.add_argument("--decode-tokens", type=int, default=64)
            command.add_argument("--trials", type=int, default=3)
        if name in {"semantic-run", "context-run", "context-run-sparse"}:
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--num-shards", type=int, default=1)
        if name == "context-run-cached-sink":
            command.add_argument("--context-lengths", type=int, nargs="+", default=[16384, 24576, 32768])
            command.add_argument("--documents", type=int, default=8)
            command.add_argument("--eval-tokens", type=int, default=64)
            command.add_argument("--budgets", type=int, nargs="+", default=[128, 512, 2048, 8192])
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--num-shards", type=int, default=1)
        if name == "context-run-full-kv-distribution":
            command.add_argument("--context-length", type=int, default=16384)
            command.add_argument("--documents", type=int, default=8)
            command.add_argument("--eval-tokens", type=int, default=64)
            command.add_argument("--block-size", type=int, default=128)
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--num-shards", type=int, default=1)
        if name == "context-run-head-resolved-distribution":
            command.add_argument("--context-length", type=int, default=16384)
            command.add_argument("--documents", type=int, default=8)
            command.add_argument("--eval-tokens", type=int, default=64)
            command.add_argument("--block-size", type=int, default=128)
            command.add_argument("--recent-tokens", type=int, default=2048)
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--num-shards", type=int, default=1)
        if name == "context-run-top1-severity":
            command.add_argument("--context-length", type=int, default=16384)
            command.add_argument("--documents", type=int, default=8)
            command.add_argument("--eval-tokens", type=int, default=64)
            command.add_argument("--budgets", type=int, nargs="+", default=[128, 512, 2048, 8192])
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare":
        print(*prepare(cfg), sep="\n")
    elif args.command == "semantic-run":
        print(run_semantic(cfg, args.shard_index, args.num_shards))
    elif args.command == "semantic-summarize":
        print(summarize_semantic(cfg))
    elif args.command == "semantic-diagnose-representations":
        print(diagnose_representations(cfg))
    elif args.command == "context-diagnose-2k":
        print(diagnose_context_length(cfg))
    elif args.command == "context-prepare":
        print(prepare_contexts(cfg))
    elif args.command == "context-run":
        print(run_context_ablation(cfg, args.shard_index, args.num_shards))
    elif args.command == "context-summarize":
        print(*summarize_context(cfg), sep="\n")
    elif args.command == "context-profile-need":
        print(*profile_context_need(cfg), sep="\n")
    elif args.command == "context-evaluate-router":
        print(evaluate_router(cfg, load_config(args.draft_config)))
    elif args.command == "context-benchmark-cache":
        print(benchmark_cache(cfg, args.prefill_tokens, args.repeats))
    elif args.command == "context-benchmark-attention":
        print(benchmark_decode_attention(cfg, args.context_lengths, args.repeats))
    elif args.command in {"context-benchmark-v1-inference", "context-benchmark-inference"}:
        print(benchmark_v1_inference(cfg, args.context_lengths, args.decode_tokens, args.trials))
    elif args.command == "context-run-sparse":
        print(run_sparse_context_ablation(cfg, args.shard_index, args.num_shards))
    elif args.command == "context-summarize-sparse":
        print(*summarize_sparse_context(cfg), sep="\n")
    elif args.command == "context-report-engineering":
        print(write_engineering_report(cfg))
    elif args.command == "context-explore-router":
        print(*explore_lightweight_routers(cfg), sep="\n")
    elif args.command == "context-run-cached-sink":
        print(run_cached_sink_experiment(
            cfg,
            args.context_lengths,
            args.documents,
            args.eval_tokens,
            args.budgets,
            args.shard_index,
            args.num_shards,
        ))
    elif args.command == "context-summarize-sink":
        print(*summarize_attention_sink(cfg), sep="\n")
    elif args.command == "context-compare-predictors":
        print(*compare_predictor_mechanisms(cfg), sep="\n")
    elif args.command == "context-report-sink-predictors":
        print(write_sink_predictor_summary(cfg))
    elif args.command == "context-analyze-sink-categories":
        print(*analyze_sink_categories(cfg), sep="\n")
    elif args.command == "context-run-full-kv-distribution":
        print(run_full_kv_distribution(
            cfg,
            args.context_length,
            args.documents,
            args.eval_tokens,
            args.block_size,
            args.shard_index,
            args.num_shards,
        ))
    elif args.command == "context-run-head-resolved-distribution":
        print(run_head_resolved_distribution(
            cfg,
            args.context_length,
            args.documents,
            args.eval_tokens,
            args.block_size,
            args.recent_tokens,
            args.shard_index,
            args.num_shards,
        ))
    elif args.command == "context-run-top1-severity":
        print(run_top1_severity_experiment(
            cfg,
            args.context_length,
            args.documents,
            args.eval_tokens,
            args.budgets,
            args.shard_index,
            args.num_shards,
        ))
    elif args.command == "context-summarize-top1-severity":
        print(*summarize_top1_severity(cfg), sep="\n")
    elif args.command == "context-analyze-fine-attention":
        print(*analyze_fine_attention(cfg), sep="\n")


if __name__ == "__main__":
    main()
