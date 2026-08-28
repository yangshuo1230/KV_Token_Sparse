from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import Config


BUDGETS = np.array([256, 512, 2048, 8192, 24576], dtype=int)


def _load_labels(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(glob.glob(str(output_dir / "top1_severity_metrics-???-of-004.parquet")))
    if len(paths) != 4:
        raise FileNotFoundError("run the four 24K top1-severity shards first")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    if set(frame.cache_budget.unique()) != {256, 512, 2048, 8192}:
        raise ValueError("expected 256/512/2048/8192 cache-budget labels")
    identity = ["document", "target_index"]
    base = frame[frame.cache_budget.eq(256)].sort_values(identity).reset_index(drop=True)
    delta = frame.pivot(index=identity, columns="cache_budget", values="delta_ce").reindex(
        pd.MultiIndex.from_frame(base[identity])
    )
    changed = frame.pivot(
        index=identity, columns="cache_budget", values="top1_changed"
    ).reindex(delta.index)
    return base, pd.concat({"delta_ce": delta, "top1_changed": changed}, axis=1)


def _required_budget_classes(delta: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    x = np.log2(BUDGETS)
    labels = []
    for row in delta:
        losses = np.r_[np.maximum(row, 0), 0.0]
        monotone = IsotonicRegression(increasing=False).fit_transform(x, losses)
        passing = np.flatnonzero(monotone <= threshold)
        labels.append(int(passing[0]) if len(passing) else len(BUDGETS) - 1)
    return np.asarray(labels, dtype=int)


def _history_features(
    ids: np.ndarray,
    target_index: int,
    embedding: np.ndarray,
    vocabulary_norm_mean: float,
    vocabulary_norm_std: float,
    vocabulary_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    query_index = target_index - 1
    query_id = int(ids[query_index])
    target_id = int(ids[target_index])
    query_vector = embedding[query_id]
    target_vector = embedding[target_id]
    query_unit = query_vector / max(np.linalg.norm(query_vector), 1e-8)
    target_unit = target_vector / max(np.linalg.norm(target_vector), 1e-8)
    history_vectors = []
    scalars = []
    posthoc: dict[str, float] = {}

    for window in (4, 16):
        history_ids = ids[max(0, query_index - window + 1) : query_index + 1]
        history_vectors.append(embedding[history_ids].mean(axis=0))
    for window in (1, 4, 16, 64, 256):
        previous_ids = ids[max(0, query_index - window) : query_index]
        previous_vectors = embedding[previous_ids]
        if len(previous_ids):
            previous_unit = previous_vectors / np.maximum(
                np.linalg.norm(previous_vectors, axis=1, keepdims=True), 1e-8
            )
            query_max_cos = float((previous_unit @ query_unit).max())
            target_max_cos = float((previous_unit @ target_unit).max())
        else:
            query_max_cos = target_max_cos = 0.0
        scalars.extend([
            float(np.any(previous_ids == query_id)),
            1 - query_max_cos,
            len(np.unique(previous_ids)) / max(1, len(previous_ids)),
        ])
        posthoc[f"target_repeat_last_{window}"] = float(np.any(previous_ids == target_id))
        posthoc[f"target_embedding_novelty_last_{window}"] = 1 - target_max_cos

    prefix = ids[:query_index]
    query_hits = np.flatnonzero(prefix == query_id)
    target_hits = np.flatnonzero(prefix == target_id)
    scalars.extend([
        np.log1p(len(query_hits)),
        (query_index - query_hits[-1]) / len(ids) if len(query_hits) else 1.0,
        (np.linalg.norm(query_vector) - vocabulary_norm_mean) / vocabulary_norm_std,
        float(query_unit @ vocabulary_mean),
    ])
    posthoc.update({
        "target_context_log_frequency": float(np.log1p(len(target_hits))),
        "target_last_distance_fraction": (
            (target_index - target_hits[-1]) / len(ids) if len(target_hits) else 1.0
        ),
        "target_embedding_norm_z": float(
            (np.linalg.norm(target_vector) - vocabulary_norm_mean) / vocabulary_norm_std
        ),
        "target_embedding_cosine_to_vocab_mean": float(target_unit @ vocabulary_mean),
        "target_equals_query": float(target_id == query_id),
    })
    dense = np.concatenate([query_vector, *history_vectors]).astype(np.float32)
    return dense, np.asarray(scalars, dtype=np.float32), posthoc


def _document_folds(documents: pd.Series, count: int = 4) -> np.ndarray:
    unique = sorted(documents.unique())
    mapping = {document: index % count for index, document in enumerate(unique)}
    return documents.map(mapping).to_numpy()


def _fit_cross_validated_mlp(
    dense: np.ndarray,
    scalar: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.zeros(len(labels), dtype=int)
    probabilities = np.zeros((len(labels), len(BUDGETS)), dtype=float)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = ~train
        components = min(32, train.sum() - 1)
        pca = PCA(n_components=components, whiten=True, random_state=17).fit(dense[train])
        train_dense = pca.transform(dense[train])
        test_dense = pca.transform(dense[test])
        scaler = StandardScaler().fit(scalar[train])
        train_features = np.concatenate((train_dense, scaler.transform(scalar[train])), axis=1)
        test_features = np.concatenate((test_dense, scaler.transform(scalar[test])), axis=1)
        model = MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            alpha=1.0,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=17 + int(fold),
        ).fit(train_features, labels[train])
        fold_probability = model.predict_proba(test_features)
        test_indices = np.flatnonzero(test)
        class_indices = model.classes_.astype(int)
        probabilities[test_indices[:, None], class_indices[None, :]] = fold_probability
        predictions[test] = probabilities[test].argmax(axis=1)
    return predictions, probabilities


def _fit_cross_validated_ordinal_mlp(
    dense: np.ndarray,
    scalar: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    tail_probability = np.zeros((len(labels), len(BUDGETS) - 1), dtype=float)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = ~train
        components = min(32, train.sum() - 1)
        pca = PCA(n_components=components, whiten=True, random_state=31).fit(dense[train])
        scaler = StandardScaler().fit(scalar[train])
        train_features = np.concatenate(
            (pca.transform(dense[train]), scaler.transform(scalar[train])), axis=1
        )
        test_features = np.concatenate(
            (pca.transform(dense[test]), scaler.transform(scalar[test])), axis=1
        )
        for boundary in range(len(BUDGETS) - 1):
            binary = labels[train] > boundary
            model = MLPClassifier(
                hidden_layer_sizes=(24,),
                activation="relu",
                alpha=1.0,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=31 + 7 * int(fold) + boundary,
            ).fit(train_features, binary)
            tail_probability[test, boundary] = model.predict_proba(test_features)[:, 1]
    # P(required > B) must not increase as B grows.
    return np.minimum.accumulate(tail_probability, axis=1)


def _conservative_class(probabilities: np.ndarray, quantile: float = 0.8) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    return np.argmax(cumulative >= quantile, axis=1)


def profile_and_train_context_mlp(cfg: Config) -> tuple[Path, Path]:
    """Profile 24K token dependency and train a small context-budget MLP."""
    base, wide = _load_labels(cfg.output_dir)
    delta = wide["delta_ce"][[256, 512, 2048, 8192]].to_numpy(dtype=float)
    changed = wide["top1_changed"][[256, 512, 2048, 8192]].to_numpy(dtype=bool)
    required = _required_budget_classes(delta)

    context_path = (cfg.data_dir or cfg.output_dir) / "contexts.jsonl"
    records = {row["id"]: row for row in map(json.loads, context_path.open(encoding="utf-8"))}
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    embedding = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    del model
    norms = np.linalg.norm(embedding, axis=1)
    vocabulary_mean = embedding.mean(axis=0)
    vocabulary_mean /= max(np.linalg.norm(vocabulary_mean), 1e-8)

    dense_rows = []
    scalar_rows = []
    posthoc_rows = []
    tokenized: dict[str, np.ndarray] = {}
    for row in base.itertuples():
        if row.document not in tokenized:
            tokenized[row.document] = np.asarray(
                tokenizer(
                    records[row.document]["text"],
                    add_special_tokens=True,
                    truncation=True,
                    max_length=24576,
                ).input_ids,
                dtype=np.int64,
            )
        dense_features, scalar_features, posthoc = _history_features(
            tokenized[row.document],
            int(row.target_index),
            embedding,
            float(norms.mean()),
            float(norms.std()),
            vocabulary_mean,
        )
        dense_rows.append(dense_features)
        scalar_rows.append(scalar_features)
        posthoc_rows.append(posthoc)
    dense_features = np.stack(dense_rows)
    scalar_features = np.stack(scalar_rows)
    posthoc = pd.DataFrame(posthoc_rows)
    folds = _document_folds(base.document)
    predicted, probabilities = _fit_cross_validated_mlp(
        dense_features, scalar_features, required, folds
    )
    conservative = _conservative_class(probabilities)
    ordinal_probability = _fit_cross_validated_ordinal_mlp(
        dense_features, scalar_features, required, folds
    )
    ordinal_p50 = (ordinal_probability >= 0.5).sum(axis=1)
    ordinal_p80 = (ordinal_probability >= 0.2).sum(axis=1)

    dependency = delta[:, 0] > 0.1
    trait_rows = []
    for feature in posthoc.columns:
        values = posthoc[feature].to_numpy(dtype=float)
        if np.unique(values).size < 2:
            continue
        try:
            auc = roc_auc_score(dependency, values)
            auc = max(auc, 1 - auc)
        except ValueError:
            auc = float("nan")
        trait_rows.append({
            "feature": feature,
            "long_mean": float(values[dependency].mean()),
            "short_mean": float(values[~dependency].mean()),
            "standardized_difference": float(
                (values[dependency].mean() - values[~dependency].mean())
                / max(values.std(), 1e-8)
            ),
            "univariate_auc_best_direction": auc,
        })
    traits = pd.DataFrame(trait_rows).sort_values(
        "univariate_auc_best_direction", ascending=False
    )

    result_rows = []
    for name, prediction in (
        ("multiclass_argmax", predicted),
        ("multiclass_p80", conservative),
        ("ordinal_p50", ordinal_p50),
        ("ordinal_p80", ordinal_p80),
    ):
        chosen_delta = np.array([
            0.0 if choice == 4 else delta[index, choice]
            for index, choice in enumerate(prediction)
        ])
        chosen_changed = np.array([
            False if choice == 4 else changed[index, choice]
            for index, choice in enumerate(prediction)
        ])
        result_rows.append({
            "policy": name,
            "exact_accuracy": accuracy_score(required, prediction),
            "within_one_bucket": float((np.abs(required - prediction) <= 1).mean()),
            "macro_f1": f1_score(required, prediction, average="macro"),
            "under_route_rate": float((prediction < required).mean()),
            "severe_under_route_rate": float((prediction + 1 < required).mean()),
            "mean_predicted_budget": float(BUDGETS[prediction].mean()),
            "mean_routed_delta_ce": float(chosen_delta.mean()),
            "routed_top1_change_rate": float(chosen_changed.mean()),
            "long256_auc": float(
                roc_auc_score(required > 0, ordinal_probability[:, 0])
            ),
        })
    results = pd.DataFrame(result_rows)

    local_path = cfg.output_dir / "context_length_mlp_detailed.csv"
    trait_path = cfg.output_dir / "context_length_token_traits.csv"
    pd.concat(
        [
            base[["document", "target_index"]],
            pd.DataFrame(delta, columns=[f"delta_ce_{budget}" for budget in BUDGETS[:-1]]),
            pd.DataFrame(changed, columns=[f"top1_changed_{budget}" for budget in BUDGETS[:-1]]),
            pd.DataFrame({
                "required_budget": BUDGETS[required],
                "mlp_argmax_budget": BUDGETS[predicted],
                "mlp_p80_budget": BUDGETS[conservative],
                "ordinal_p50_budget": BUDGETS[ordinal_p50],
                "ordinal_p80_budget": BUDGETS[ordinal_p80],
                "ordinal_probability_gt_256": ordinal_probability[:, 0],
            }),
            posthoc,
        ],
        axis=1,
    ).to_csv(local_path, index=False)
    traits.to_csv(trait_path, index=False)

    distribution = pd.Series(BUDGETS[required]).value_counts(normalize=True).sort_index()
    threshold_rates = {
        threshold: float((delta[:, 0] > threshold).mean())
        for threshold in (0.05, 0.1, 0.2, 0.5)
    }
    report_path = cfg.output_dir / "CONTEXT_LENGTH_MLP_SUMMARY.md"
    best_traits = traits.head(8)
    trait_lines = [
        f"| {row.feature} | {row.long_mean:.4f} | {row.short_mean:.4f} | "
        f"{row.standardized_difference:+.3f} | {row.univariate_auc_best_direction:.3f} |"
        for row in best_traits.itertuples()
    ]
    result_lines = [
        f"| {row.policy} | {row.exact_accuracy:.3f} | {row.macro_f1:.3f} | "
        f"{row.under_route_rate:.3f} | {row.mean_predicted_budget:,.0f} | "
        f"{row.mean_routed_delta_ce:.4f} | {row.routed_top1_change_rate:.3f} | "
        f"{row.long256_auc:.3f} |"
        for row in results.itertuples()
    ]
    report_path.write_text(
        "# 24K context-length dependency and small-MLP result\n\n"
        f"The dataset contains {len(base):,} target tokens from "
        f"{base.document.nunique()} held-out-able PG-19 documents. Every compact policy "
        "retains Prefix-1; the 256-token budget therefore means Prefix-1 + Recent-255.\n\n"
        "## Dependency at budget 256\n\n"
        + "\n".join(
            f"- delta CE > {threshold:g}: {100*rate:.2f}%"
            for threshold, rate in threshold_rates.items()
        )
        + f"\n- Top-1 changed: {100*changed[:, 0].mean():.2f}%\n\n"
        "## Monotonicized minimum-budget labels\n\n"
        + "\n".join(
            f"- {int(budget):,}: {100*distribution.get(budget, 0):.2f}%"
            for budget in BUDGETS
        )
        + "\n\n## Strongest token-history/embedding traits\n\n"
        "These `target_*` traits describe the ground-truth next token after the fact and "
        "are diagnostics, not causal router inputs.\n\n"
        "| Feature | Long mean | Short mean | Std. difference | Univariate AUC |\n"
        "|---|---:|---:|---:|---:|\n" + "\n".join(trait_lines)
        + "\n\n## Four-fold document-held-out MLP\n\n"
        "The causal MLP uses 32 PCA components from the current input and preceding "
        "last-4/last-16 token embeddings, "
        "plus repetition, cosine-novelty, frequency, and distance scalars. It has one "
        "32-unit hidden layer.\n\n"
        "| Policy | Exact acc. | Macro F1 | Under-route | Mean budget | Routed delta CE | Top-1 change | >256 AUC |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(result_lines)
        + "\n\nLabels are derived with per-token isotonic regression over 256/512/2K/8K/full "
        "delta CE and threshold 0.1. Detailed files are local-only and ignored by Git.\n",
        encoding="utf-8",
    )
    return report_path, trait_path
