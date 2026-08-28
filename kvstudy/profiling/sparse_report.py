from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from ..config import Config
from .router import _bootstrap_difference, _load_metrics


METRICS = ("delta_ce", "kl_full_to_compact", "top1_changed")


def summarize_sparse_context(cfg: Config) -> tuple[Path, Path]:
    combined = cfg.output_dir / "sparse_context_metrics.parquet"
    if combined.exists():
        paths = [combined]
    else:
        groups: dict[int, list[Path]] = {}
        for path in cfg.output_dir.glob("sparse_context_metrics-*-of-*.parquet"):
            match = re.search(r"-of-(\d+)\.parquet$", path.name)
            if match:
                groups.setdefault(int(match.group(1)), []).append(path)
        complete = {count: paths for count, paths in groups.items() if len(paths) == count}
        if not complete:
            raise FileNotFoundError("no complete sparse context metric shard set found")
        paths = sorted(complete[max(complete)])
    sparse = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    dense = _load_metrics(cfg.output_dir)
    dense = dense[dense.sink_size.eq(0)]
    identity = ["document", "target_index"]
    rows: list[dict] = []

    for comparator_budget in (cfg.context.profile_recent_budget, 4096, 8192):
        comparator = dense[dense.cache_budget.eq(comparator_budget)].set_index(identity)
        candidate = sparse.set_index(identity).reindex(comparator.index)
        if candidate.delta_ce.isna().any():
            raise ValueError("sparse and dense observations do not align")
        for metric in METRICS:
            left = candidate[metric].astype(float)
            right = comparator[metric].astype(float)
            difference = left - right
            documents = pd.Series(
                candidate.index.get_level_values("document"), index=difference.index
            )
            ci_low, ci_high = _bootstrap_difference(
                difference,
                documents,
                cfg.seed,
                cfg.context.bootstrap_samples,
            )
            rows.append({
                "metric": metric,
                "sparse_mean": float(left.mean()),
                "comparator": f"static_recent_{comparator_budget}",
                "comparator_mean": float(right.mean()),
                "sparse_minus_comparator": float(difference.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "documents": sparse.document.nunique(),
                "targets": len(sparse),
                "sparse_attended_tokens": int(sparse.attended_tokens.iloc[0]),
            })
    result = pd.DataFrame(rows)
    csv_path = cfg.output_dir / "sparse_context_summary.csv"
    report_path = cfg.output_dir / "SPARSE_CONTEXT_RESULTS.md"
    result.to_csv(csv_path, index=False)
    means = sparse[list(METRICS)].astype(float).mean()
    report_path.write_text(
        "# V2 sparse-context quality\n\n"
        f"Qwen2.5-7B was evaluated on {sparse.document.nunique()} PG-19 documents of "
        f"{cfg.max_length:,} tokens ({len(sparse):,} target tokens). Every query keeps the "
        f"most recent {cfg.context.profile_recent_budget:,} tokens and attends "
        f"{int(sparse.attended_tokens.iloc[0]) - cfg.context.profile_recent_budget:,} remote "
        f"tokens selected in {cfg.context.block_size}-token pages. Page relevance is the "
        "first-layer query/key-landmark score, refreshed once for each 64-token evaluation block.\n\n"
        f"Mean sparse ΔCE is {means.delta_ce:.4f}, full-to-sparse KL is "
        f"{means.kl_full_to_compact:.4f}, and top-1 changes on "
        f"{100*means.top1_changed:.2f}% of targets. Paired comparisons and document-level "
        "bootstrap confidence intervals are in `sparse_context_summary.csv`.\n\n"
        "The quality run recomputes the selected compact sequence with an explicit causal "
        "mask and original RoPE positions. Kernel timing is measured independently with "
        "the paged FlashInfer path in `decode_attention_benchmark.csv`.\n",
        encoding="utf-8",
    )
    return csv_path, report_path
