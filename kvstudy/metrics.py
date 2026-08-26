from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


def top_indices(p: np.ndarray, k: int) -> np.ndarray:
    k = min(k, p.size)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    return np.argpartition(p, -k)[-k:]


def pair_metrics(a: np.ndarray, b: np.ndarray, k: int) -> dict[str, float]:
    eps = 1e-12
    a = a / max(a.sum(), eps)
    b = b / max(b.sum(), eps)
    ta, tb = set(top_indices(a, k)), set(top_indices(b, k))
    midpoint = 0.5 * (a + b)
    js = 0.5 * np.sum(a * np.log((a + eps) / (midpoint + eps)))
    js += 0.5 * np.sum(b * np.log((b + eps) / (midpoint + eps)))
    if np.std(a) == 0 or np.std(b) == 0:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
    union = ta | tb
    return {
        "jaccard": len(ta & tb) / len(union) if union else 0.0,
        "js_similarity": 1.0 - float(js / math.log(2)),
        "spearman": spearman,
        "top_mass_overlap": float(np.minimum(a[list(union)], b[list(union)]).sum()) if union else 0.0,
    }


def coverage(route: np.ndarray, targets: np.ndarray, k: int) -> float:
    chosen = top_indices(route, k)
    return float(targets[:, chosen].sum(axis=1).mean()) if targets.size else math.nan


@dataclass
class Accumulator:
    values: dict[tuple, list[float]]

    def __init__(self):
        self.values = defaultdict(lambda: [0.0, 0.0])

    def add(self, key: tuple, value: float) -> None:
        if np.isfinite(value):
            self.values[key][0] += float(value)
            self.values[key][1] += 1.0

    def rows(self, doc_id: str, corpus: str) -> list[dict]:
        rows = []
        for key, (total, count) in self.values.items():
            layer, head, group, region, block_size, top_k, condition, metric = key
            rows.append({"document": doc_id, "corpus": corpus, "layer": layer, "head": head,
                         "gqa_group": group, "region": region, "block_size": block_size, "top_k": top_k,
                         "condition": condition, "metric": metric, "value": total / count,
                         "observations": int(count)})
        return rows
