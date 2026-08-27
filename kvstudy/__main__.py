from __future__ import annotations

import argparse

from .config import load_config
from .data import prepare
from .semantic.experiment import run as run_semantic
from .semantic.report import summarize as summarize_semantic
from .semantic.representations import diagnose_representations
from .token_context.ablation import diagnose_context_length
from .token_context.experiment import prepare_contexts, run_context_ablation
from .token_context.report import summarize_context


COMMANDS = (
    "prepare",
    "semantic-run",
    "semantic-summarize",
    "semantic-diagnose-representations",
    "context-diagnose-2k",
    "context-prepare",
    "context-run",
    "context-summarize",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic KV routing and token-specific context experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        if name in {"semantic-run", "context-run"}:
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


if __name__ == "__main__":
    main()
