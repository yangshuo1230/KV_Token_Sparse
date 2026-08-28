from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config


METRICS = ("delta_ce", "kl_full_to_compact", "top1_changed")
LABEL_SCHEMES = ("coarse_category", "fine_category", "pos", "is_first_subtoken")


def _bootstrap(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _metric_paths(output_dir: Path) -> list[Path]:
    combined = output_dir / "context_metrics.parquet"
    return [combined] if combined.exists() else sorted(output_dir.glob("context_metrics-*-of-*.parquet"))


def summarize_context(cfg: Config) -> tuple[Path, Path, Path, Path]:
    paths = _metric_paths(cfg.output_dir)
    if not paths:
        raise FileNotFoundError("no context metrics found")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    rng = np.random.default_rng(cfg.seed)
    samples = cfg.context.bootstrap_samples
    summary_rows: list[dict] = []
    contrast_rows: list[dict] = []
    sink_rows: list[dict] = []
    pair_rows: list[dict] = []

    for scheme in LABEL_SCHEMES:
        keys = ["cache_budget", "sink_size", scheme]
        for key, group in frame.groupby(keys, dropna=False):
            cache_budget, sink_size, label = key
            for metric in METRICS:
                by_doc = group.groupby("document")[metric].mean().to_numpy(dtype=float)
                ci_low, ci_high = _bootstrap(by_doc, rng, samples)
                summary_rows.append({
                    "label_scheme": scheme,
                    "label": label,
                    "cache_budget": cache_budget,
                    "sink_size": sink_size,
                    "metric": metric,
                    "mean": float(by_doc.mean()),
                    "documents": len(by_doc),
                    "tokens": len(group),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                })

        # Center category effects within each document and policy. This controls
        # corpus/document difficulty while retaining an interpretable effect in
        # the original metric units.
        for metric in METRICS:
            overall = frame.groupby(
                ["document", "cache_budget", "sink_size"], dropna=False
            )[metric].mean().rename("document_mean")
            categorized = frame.groupby(
                ["document", "cache_budget", "sink_size", scheme], dropna=False
            )[metric].mean().rename("category_mean").reset_index()
            categorized = categorized.join(
                overall,
                on=["document", "cache_budget", "sink_size"],
            )
            categorized["difference"] = categorized["category_mean"] - categorized["document_mean"]
            for key, group in categorized.groupby(["cache_budget", "sink_size", scheme], dropna=False):
                cache_budget, sink_size, label = key
                values = group["difference"].to_numpy(dtype=float)
                ci_low, ci_high = _bootstrap(values, rng, samples)
                contrast_rows.append({
                    "label_scheme": scheme,
                    "label": label,
                    "cache_budget": cache_budget,
                    "sink_size": sink_size,
                    "metric": metric,
                    "contrast": "category_minus_document_mean",
                    "mean_difference": float(values.mean()),
                    "documents": len(values),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                })

        # Exact target-paired sink comparisons at a fixed total KV budget.
        identity = ["document", "target_index", "cache_budget", scheme]
        for metric in METRICS:
            pivot = frame.pivot_table(index=identity, columns="sink_size", values=metric, aggfunc="mean")
            if 0 not in pivot:
                continue
            for sink_size in sorted(value for value in frame.sink_size.unique() if value != 0):
                if sink_size not in pivot:
                    continue
                paired = (pivot[sink_size] - pivot[0]).rename("difference").reset_index()
                by_doc = paired.groupby(["document", "cache_budget", scheme], dropna=False)[
                    "difference"
                ].mean().reset_index()
                for key, group in by_doc.groupby(["cache_budget", scheme], dropna=False):
                    cache_budget, label = key
                    values = group["difference"].to_numpy(dtype=float)
                    ci_low, ci_high = _bootstrap(values, rng, samples)
                    sink_rows.append({
                        "label_scheme": scheme,
                        "label": label,
                        "cache_budget": cache_budget,
                        "sink_size": sink_size,
                        "metric": metric,
                        "contrast": "sink_plus_recent_minus_recent_only",
                        "mean_difference": float(values.mean()),
                        "documents": len(values),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    })

    summary_path = cfg.output_dir / "category_summary.csv"
    contrast_path = cfg.output_dir / "category_contrasts.csv"
    sink_path = cfg.output_dir / "sink_contrasts.csv"
    pair_path = cfg.output_dir / "category_pair_contrasts.csv"

    pair_specs = (
        ("coarse_category", "content", "function", "content_minus_function"),
        ("is_first_subtoken", True, False, "first_minus_continuation"),
    )
    for scheme, left, right, name in pair_specs:
        for metric in METRICS:
            by_doc = frame.groupby(
                ["document", "cache_budget", "sink_size", scheme], dropna=False
            )[metric].mean().unstack(scheme)
            if left not in by_doc or right not in by_doc:
                continue
            paired = (by_doc[left] - by_doc[right]).rename("difference").reset_index()
            for key, group in paired.groupby(["cache_budget", "sink_size"], dropna=False):
                values = group["difference"].dropna().to_numpy(dtype=float)
                if not len(values):
                    continue
                ci_low, ci_high = _bootstrap(values, rng, samples)
                pair_rows.append({
                    "label_scheme": scheme,
                    "metric": metric,
                    "contrast": name,
                    "cache_budget": key[0],
                    "sink_size": key[1],
                    "mean_difference": float(values.mean()),
                    "documents": len(values),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                })

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(contrast_rows).to_csv(contrast_path, index=False)
    pd.DataFrame(sink_rows).to_csv(sink_path, index=False)
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False)
    return summary_path, contrast_path, sink_path, pair_path
