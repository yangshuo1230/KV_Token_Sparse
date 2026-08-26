from __future__ import annotations

import argparse

from .config import load_config
from .data import prepare
from .experiment import run
from .report import summarize
from .representations import diagnose_representations
from .context_ablation import diagnose_context_length
from .long_context import prepare_long_contexts, run_long_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic KV-block routing experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "summarize", "diagnose-representations", "diagnose-context-length",
                 "prepare-long-context", "run-long-context", "run-target-long-context"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "run":
            command.add_argument("--shard-index", type=int, default=0)
            command.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare":
        print(*prepare(cfg), sep="\n")
    elif args.command == "run":
        print(run(cfg, args.shard_index, args.num_shards))
    elif args.command == "diagnose-representations":
        print(diagnose_representations(cfg))
    elif args.command == "diagnose-context-length":
        print(diagnose_context_length(cfg))
    elif args.command == "prepare-long-context":
        print(prepare_long_contexts(cfg))
    elif args.command == "run-long-context":
        print(run_long_context(cfg, target_token=False))
    elif args.command == "run-target-long-context":
        print(run_long_context(cfg, target_token=True))
    else:
        print(summarize(cfg))


if __name__ == "__main__":
    main()
