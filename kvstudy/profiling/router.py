from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config


IDENTITY = ["document", "target_index"]
PLANS = (
    (128, 2048, 512),
    (512, 2048, 1024),
    (2048, 8192, 4096),
)


def _load_metrics(output_dir: Path) -> pd.DataFrame:
    combined = output_dir / "context_metrics.parquet"
    paths = (
        [combined]
        if combined.exists()
        else sorted(output_dir.glob("context_metrics-*-of-*.parquet"))
    )
    if not paths:
        raise FileNotFoundError(f"no context metrics in {output_dir}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels.astype(bool)
    negative = ~positive
    if not positive.any() or not negative.any():
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    count = positive.sum()
    return float(
        (ranks[positive].sum() - count * (count + 1) / 2) / (count * negative.sum())
    )


def cross_validated_type_scores(
    frame: pd.DataFrame,
    feature: str,
    folds: int = 4,
    smoothing: float = 4.0,
) -> np.ndarray:
    """Predict next-token content probability with a tiny smoothed lookup."""
    documents = sorted(frame.document.unique())
    fold_by_document = {document: index % folds for index, document in enumerate(documents)}
    scores = np.zeros(len(frame), dtype=float)
    labels = frame.coarse_category.eq("content").astype(float)
    row_folds = frame.document.map(fold_by_document).to_numpy()

    for fold in range(folds):
        train = row_folds != fold
        test = ~train
        prior = float(labels[train].mean())
        stats = pd.DataFrame({feature: frame.loc[train, feature], "label": labels[train]}).groupby(
            feature
        ).label.agg(["sum", "count"])
        lookup = ((stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)).to_dict()
        scores[test] = frame.loc[test, feature].map(lookup).fillna(prior)
    return scores


def _bootstrap_difference(
    values: pd.Series,
    documents: pd.Series,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    by_document = values.groupby(documents).mean().to_numpy()
    if len(by_document) == 1:
        return float(by_document[0]), float(by_document[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(by_document), size=(samples, len(by_document)))
    estimates = by_document[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def evaluate_router(target_cfg: Config, draft_cfg: Config) -> Path:
    """Evaluate causal type lookups against equal-average-budget static caches."""
    target = _load_metrics(target_cfg.output_dir)
    draft = _load_metrics(draft_cfg.output_dir)
    required = {budget for plan in PLANS for budget in plan}
    available = set(target.cache_budget.unique())
    if not required <= available:
        raise ValueError(f"router evaluation requires budgets {sorted(required)}")

    # Route the deployed recent-only policy. Sink variants require an explicit
    # compact causal mask and are not features available to this predictor.
    target = target[target.sink_size.eq(0)]
    draft = draft[(draft.sink_size.eq(0)) & (draft.cache_budget.eq(128))]
    base = target[target.cache_budget.eq(128)].set_index(IDENTITY).sort_index()
    draft = draft.set_index(IDENTITY).reindex(base.index)
    if draft.predicted_token_id.isna().any():
        raise ValueError("target and draft observations do not align")
    base = base.reset_index()
    base["draft_predicted_token_id"] = draft.predicted_token_id.to_numpy()
    wide = target.pivot_table(
        index=IDENTITY,
        columns="cache_budget",
        values=["delta_ce", "top1_changed"],
    ).reindex(pd.MultiIndex.from_frame(base[IDENTITY]))

    router_features = {
        "input_token_lookup": "query_token_id",
        "draft_token_lookup": "draft_predicted_token_id",
    }
    rows: list[dict] = []
    for router, feature in router_features.items():
        scores = cross_validated_type_scores(base, feature)
        labels = base.coarse_category.eq("content").to_numpy()
        auc = _auc(labels, scores)
        for low, high, static in PLANS:
            high_fraction = (static - low) / (high - low)
            high_count = round(high_fraction * len(base))
            selected = np.zeros(len(base), dtype=bool)
            selected[np.argsort(scores)[-high_count:]] = True
            average_budget = low + (high - low) * selected.mean()

            benefit = (
                wide[("delta_ce", low)].to_numpy() - wide[("delta_ce", high)].to_numpy()
            )
            oracle = np.zeros(len(base), dtype=bool)
            oracle[np.argsort(benefit)[-high_count:]] = True
            for metric in ("delta_ce", "top1_changed"):
                low_values = wide[(metric, low)].to_numpy()
                high_values = wide[(metric, high)].to_numpy()
                static_values = wide[(metric, static)].to_numpy()
                routed = np.where(selected, high_values, low_values)
                oracle_values = np.where(oracle, high_values, low_values)
                difference = pd.Series(routed - static_values)
                ci_low, ci_high = _bootstrap_difference(
                    difference,
                    base.document,
                    target_cfg.seed,
                    target_cfg.context.bootstrap_samples,
                )
                rows.append({
                    "router": router,
                    "feature": feature,
                    "type_auc": auc,
                    "low_budget": low,
                    "high_budget": high,
                    "static_budget": static,
                    "average_budget": average_budget,
                    "high_fraction": selected.mean(),
                    "metric": metric,
                    "router_mean": float(routed.mean()),
                    "static_mean": float(static_values.mean()),
                    "router_minus_static": float(difference.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "oracle_mean": float(oracle_values.mean()),
                    "documents": base.document.nunique(),
                    "targets": len(base),
                })

    output = target_cfg.output_dir / "router_evaluation.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
