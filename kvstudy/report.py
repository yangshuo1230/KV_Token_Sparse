from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def summarize(cfg: Config):
    path = cfg.output_dir / "document_metrics.parquet"
    paths = [path] if path.exists() else sorted(cfg.output_dir.glob("document_metrics-*-of-*.parquet"))
    if not paths:
        raise FileNotFoundError("no document metrics found")
    df = pd.concat((pd.read_parquet(item) for item in paths), ignore_index=True)
    keys = ["corpus", "layer", "head", "gqa_group", "region", "block_size", "top_k", "condition", "metric"]
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for key, group in df.groupby(keys, dropna=False):
        by_doc = group.groupby("document")["value"].mean().to_numpy()
        if len(by_doc) == 1:
            estimates = np.repeat(by_doc, cfg.analysis.bootstrap_samples)
        else:
            indices = rng.integers(0, len(by_doc), size=(cfg.analysis.bootstrap_samples, len(by_doc)))
            estimates = by_doc[indices].mean(axis=1)
        rows.append(dict(zip(keys, key), mean=float(by_doc.mean()), documents=len(by_doc),
                         ci_low=float(np.quantile(estimates, 0.025)),
                         ci_high=float(np.quantile(estimates, 0.975))))
    output = cfg.output_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    contrast_specs = [
        ("jaccard", "semantic", "fixed", "semantic_minus_fixed"),
        ("jaccard", "fixed", "random", "fixed_minus_random"),
        ("evidence_recall", "content_route", "all_route", "content_minus_all"),
        ("retained_ratio", "semantic_shared", "fixed_shared", "semantic_minus_fixed"),
    ]
    identity = ["document", "corpus", "layer", "head", "gqa_group", "region", "block_size", "top_k"]
    contrast_rows = []
    for metric, left, right, contrast in contrast_specs:
        subset = df[(df.metric == metric) & df.condition.isin([left, right])]
        paired = subset.pivot_table(index=identity, columns="condition", values="value", aggfunc="mean")
        if left not in paired or right not in paired:
            continue
        differences = (paired[left] - paired[right]).rename("difference").reset_index()
        strata = identity[1:]
        for key, group in differences.groupby(strata, dropna=False):
            values = group["difference"].to_numpy()
            if len(values) == 1:
                estimates = np.repeat(values, cfg.analysis.bootstrap_samples)
            else:
                indices = rng.integers(0, len(values), size=(cfg.analysis.bootstrap_samples, len(values)))
                estimates = values[indices].mean(axis=1)
            contrast_rows.append(dict(zip(strata, key), metric=metric, contrast=contrast,
                                      mean_difference=float(values.mean()), documents=len(values),
                                      ci_low=float(np.quantile(estimates, 0.025)),
                                      ci_high=float(np.quantile(estimates, 0.975))))
    pd.DataFrame(contrast_rows).to_csv(cfg.output_dir / "contrasts.csv", index=False)
    return output
