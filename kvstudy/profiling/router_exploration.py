from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..config import Config
from .router import _load_metrics


def _document_folds(frame: pd.DataFrame, folds: int = 4) -> np.ndarray:
    documents = sorted(frame.document.unique())
    mapping = {document: index % folds for index, document in enumerate(documents)}
    return frame.document.map(mapping).to_numpy()


def _cross_validated_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    row_folds: np.ndarray,
    c: float,
) -> np.ndarray:
    scores = np.zeros(len(labels), dtype=float)
    for fold in sorted(np.unique(row_folds)):
        train = row_folds != fold
        test = ~train
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
        ).fit(scaler.transform(features[train]), labels[train])
        scores[test] = model.predict_proba(scaler.transform(features[test]))[:, 1]
    return scores


def _route_rows(
    name: str,
    scores: np.ndarray,
    frame: pd.DataFrame,
    target_labels: np.ndarray,
    target_name: str,
    overhead: str,
    dense_24k_ms: float,
    recent_24k_ms: float,
) -> list[dict]:
    need = frame.delta_ce.gt(0.1).to_numpy()
    changed = frame.top1_changed.astype(bool).to_numpy()
    rows = []
    for rate in (0.25, 0.4, 0.5, 0.6):
        count = round(rate * len(frame))
        selected = np.zeros(len(frame), dtype=bool)
        selected[np.argsort(scores)[-count:]] = True
        if overhead == "vocab_lut_before_forward":
            estimated_ms = recent_24k_ms + rate * (dense_24k_ms - recent_24k_ms)
        else:
            # Speculative verifier: every token pays recent, rejected tokens
            # additionally pay a complete full-context replay.
            estimated_ms = recent_24k_ms + rate * dense_24k_ms
        rows.append({
            "predictor": name,
            "target": target_name,
            "overhead": overhead,
            "target_auc": roc_auc_score(target_labels, scores),
            "target_average_precision": average_precision_score(target_labels, scores),
            "route_fraction": selected.mean(),
            "context_need_recall": float((selected & need).sum() / need.sum()),
            "top1_change_recall": float((selected & changed).sum() / changed.sum()),
            "residual_delta_ce": float(np.where(selected, 0, frame.delta_ce).mean()),
            "residual_top1_change_rate": float(np.where(selected, False, changed).mean()),
            "estimated_24k_latency_ms": estimated_ms,
            "estimated_24k_speedup": dense_24k_ms / estimated_ms,
        })
    return rows


def explore_lightweight_routers(cfg: Config) -> tuple[Path, Path, Path, Path, Path]:
    """Compare a zero-cost category LUT with a speculative confidence verifier."""
    frame = _load_metrics(cfg.output_dir)
    frame = frame[
        frame.sink_size.eq(0)
        & frame.cache_budget.eq(cfg.context.profile_recent_budget)
    ].sort_values(["document", "target_index"]).reset_index(drop=True)
    folds = _document_folds(frame)
    type_labels = frame.coarse_category.eq("content").astype(int).to_numpy()
    need_labels = frame.delta_ce.gt(cfg.context.long_context_delta_ce).astype(int).to_numpy()
    changed_labels = frame.top1_changed.astype(int).to_numpy()

    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    query_embeddings = embeddings[frame.query_token_id.to_numpy(dtype=int)]
    type_scores = _cross_validated_logistic(
        query_embeddings, type_labels, folds, c=0.001
    )

    # Fit the deployable all-data head, then fold embedding projection into a
    # vocabulary-sized FP16 lookup. Runtime cost is one indexed load.
    scaler = StandardScaler().fit(query_embeddings)
    type_head = LogisticRegression(
        C=0.001,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
    ).fit(scaler.transform(query_embeddings), type_labels)
    normalized_weights = type_head.coef_[0] / scaler.scale_
    bias = type_head.intercept_[0] - np.dot(scaler.mean_, normalized_weights)
    vocabulary_scores = expit(embeddings @ normalized_weights + bias).astype(np.float16)
    del model, embeddings, query_embeddings

    confidence_features = frame[
        ["predicted_token_probability", "prediction_margin", "prediction_entropy"]
    ].to_numpy()
    confidence_scores = _cross_validated_logistic(
        confidence_features, changed_labels, folds, c=0.1
    )
    confidence_scaler = StandardScaler().fit(confidence_features)
    confidence_head = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
    ).fit(confidence_scaler.transform(confidence_features), changed_labels)

    end_to_end = pd.read_csv(cfg.output_dir / "end_to_end_benchmark.csv")
    at_24k = end_to_end[end_to_end.context_length.eq(24576)].groupby("policy").mean(numeric_only=True)
    dense_ms = float(at_24k.loc["dense"].decode_latency_ms_mean)
    recent_ms = float(at_24k.loc["recent"].decode_latency_ms_mean)
    rows = _route_rows(
        "embedding_type_lut",
        type_scores,
        frame,
        type_labels,
        "next_token_is_content",
        "vocab_lut_before_forward",
        dense_ms,
        recent_ms,
    )
    rows += _route_rows(
        "recent_confidence_verifier",
        confidence_scores,
        frame,
        changed_labels,
        "recent_top1_differs_from_full",
        "requires_recent_forward_then_full_replay",
        dense_ms,
        recent_ms,
    )
    result = pd.DataFrame(rows)
    # Expose the mismatch explicitly: a better category predictor can still be
    # nearly random for actual long-context need.
    result.loc[result.predictor.eq("embedding_type_lut"), "context_need_auc"] = (
        roc_auc_score(need_labels, type_scores)
    )
    result.loc[result.predictor.eq("recent_confidence_verifier"), "context_need_auc"] = (
        roc_auc_score(need_labels, confidence_scores)
    )

    csv_path = cfg.output_dir / "router_exploration.csv"
    lut_path = cfg.output_dir / "embedding_type_lut.npy"
    lut_meta_path = cfg.output_dir / "embedding_type_lut_meta.json"
    head_path = cfg.output_dir / "recent_confidence_head.json"
    report_path = cfg.output_dir / "ROUTER_EXPLORATION_RESULTS.md"
    result.to_csv(csv_path, index=False)
    np.save(lut_path, vocabulary_scores)
    training_lut_scores = vocabulary_scores[frame.query_token_id.to_numpy(dtype=int)].astype(float)
    lut_meta_path.write_text(json.dumps({
        "target": "next_token_is_content",
        "dtype": "float16",
        "vocabulary_size": int(len(vocabulary_scores)),
        "thresholds": {
            str(rate): float(np.quantile(training_lut_scores, 1 - rate))
            for rate in (0.25, 0.4, 0.5, 0.6)
        },
    }, indent=2), encoding="utf-8")
    head_path.write_text(json.dumps({
        "feature_order": [
            "predicted_token_probability",
            "prediction_margin",
            "prediction_entropy",
        ],
        "mean": confidence_scaler.mean_.tolist(),
        "scale": confidence_scaler.scale_.tolist(),
        "coefficient": confidence_head.coef_[0].tolist(),
        "intercept": float(confidence_head.intercept_[0]),
        "target": "recent_top1_differs_from_full",
    }, indent=2), encoding="utf-8")

    lut = result[result.predictor.eq("embedding_type_lut")].iloc[0]
    verifier = result[result.predictor.eq("recent_confidence_verifier")]
    verifier_40 = verifier.iloc[(verifier.route_fraction - 0.4).abs().argmin()]
    report_path.write_text(
        "# Ultra-light router exploration\n\n"
        "All accuracy numbers use four-fold out-of-document validation on 1,024 "
        "targets from 16 independent 32K PG-19 documents.\n\n"
        "## Embedding category LUT\n\n"
        f"A linear head over the current input-token embedding predicts whether the next "
        f"token is a content token with ROC AUC {lut.target_auc:.3f}. After training, the "
        "embedding projection is folded into `embedding_type_lut.npy`, so runtime is one "
        f"FP16 vocabulary-table lookup. However, its AUC for actual `ΔCE > "
        f"{cfg.context.long_context_delta_ce:g}` context need is only "
        f"{lut.context_need_auc:.3f}. Better category prediction does not solve routing.\n\n"
        "## Speculative confidence verifier\n\n"
        f"A three-feature logistic head over recent-only probability, top-2 margin, and "
        f"entropy predicts whether full context changes top-1 with ROC AUC "
        f"{verifier_40.target_auc:.3f}. Routing {100*verifier_40.route_fraction:.0f}% to "
        f"full recalls {100*verifier_40.top1_change_recall:.1f}% of top-1 changes and leaves "
        f"a {100*verifier_40.residual_top1_change_rate:.2f}% residual change rate. But the "
        "signal exists only after recent-only inference. Replaying the full model for "
        f"rejected tokens gives an estimated 24K speedup of "
        f"{verifier_40.estimated_24k_speedup:.3f}x, i.e. a slowdown.\n\n"
        "## Decision\n\n"
        "Keep the vocabulary LUT as an essentially free feature, but do not use word class "
        "as the routing decision by itself. The confidence head is useful as a quality "
        "verifier, not as a latency optimization on the current full-replay design. A future "
        "router needs a pre-forward retrieval/surprise signal or a fused partial replay that "
        "avoids repeating MLP and projection work. Full operating points are in "
        "`router_exploration.csv`.\n",
        encoding="utf-8",
    )
    return csv_path, lut_path, lut_meta_path, head_path, report_path
