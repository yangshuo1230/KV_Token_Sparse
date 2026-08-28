from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..config import Config
from ..model import decoder_layers, load_model
from .router import _load_metrics
from .router_exploration import _document_folds
from .sparse_experiment import _rotate_half
from .sink_report import _complete_shards


def _cv_linear(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    c: float = 0.01,
) -> np.ndarray:
    scores = np.zeros(len(labels), dtype=float)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
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


def _memory_features(ids: np.ndarray, rows: pd.DataFrame, recent: int) -> np.ndarray:
    result = []
    for row in rows.itertuples():
        target = int(row.target_index)
        query_position = target - 1
        remote_end = max(0, target - recent)
        remote = ids[:remote_end]
        local = ids[remote_end:target]
        features = []
        for token_id in (int(row.query_token_id), int(row.predicted_token_id)):
            remote_hits = np.flatnonzero(remote == token_id)
            local_hits = np.flatnonzero(local == token_id)
            features.extend([
                np.log1p(len(remote_hits)),
                (query_position - remote_hits[-1]) / len(ids) if len(remote_hits) else 1.0,
                np.log1p(len(local_hits)),
                (len(local) - 1 - local_hits[-1]) / recent if len(local_hits) else 1.0,
            ])
        if query_position >= 1:
            pair = ids[query_position - 1 : query_position + 1]
            remote_pairs = np.flatnonzero(
                (remote[:-1] == pair[0]) & (remote[1:] == pair[1])
            ) if len(remote) > 1 else np.array([], dtype=int)
        else:
            remote_pairs = np.array([], dtype=int)
        features.extend([
            np.log1p(len(remote_pairs)),
            (query_position - remote_pairs[-1]) / len(ids) if len(remote_pairs) else 1.0,
        ])
        result.append(features)
    return np.asarray(result, dtype=np.float32)


def _stateful_features(values: np.ndarray, documents: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    state = np.zeros(values.shape[1], dtype=np.float32)
    output = np.zeros((len(values), values.shape[1] * 3), dtype=np.float32)
    previous_document = None
    for index, (row, document) in enumerate(zip(values, documents, strict=True)):
        if document != previous_document:
            state.fill(0)
            previous_document = document
        surprise = np.abs(row - state)
        state = alpha * state + (1 - alpha) * row
        output[index] = np.concatenate((row, state, surprise))
    return output


@torch.inference_mode()
def compare_predictor_mechanisms(cfg: Config) -> tuple[Path, Path]:
    """Evaluate pre-forward retrieval/memory signals and post-forward verifier."""
    base = _load_metrics(cfg.output_dir)
    base = base[
        base.sink_size.eq(0)
        & base.cache_budget.eq(cfg.context.profile_recent_budget)
    ].set_index(["document", "target_index"]).sort_index()
    cached = pd.concat(
        (
            pd.read_parquet(path)
            for path in _complete_shards(cfg.output_dir, "cached_sink_metrics")
        ),
        ignore_index=True,
    )
    cached = cached[
        cached.context_length.eq(cfg.max_length)
        & cached.cache_budget.eq(cfg.context.profile_recent_budget)
        & cached.policy.eq("prefix")
        & cached.remote_count.eq(1)
    ].set_index(["document", "target_index"]).sort_index()
    frame = base.reindex(cached.index).reset_index()
    if frame.query_token_id.isna().any():
        raise ValueError("cached sink targets do not align with token metadata")
    for column in (
        "delta_ce",
        "kl_full_to_compact",
        "top1_changed",
        "predicted_token_id",
        "predicted_token_probability",
        "prediction_margin",
        "prediction_entropy",
    ):
        frame[column] = cached[column].to_numpy()
    path = (cfg.data_dir or cfg.output_dir) / "contexts.jsonl"
    records = {row["id"]: row for row in map(json.loads, path.open(encoding="utf-8"))}
    model, tokenizer = load_model(cfg.model, cfg.device, cfg.dtype)
    base, _ = decoder_layers(model)
    page_size = cfg.context.block_size
    recent_pages = cfg.context.profile_recent_budget // page_size

    embedding_parts = []
    early_parts = []
    retrieval_parts = []
    memory_parts = []
    ordered_documents = []
    for document, rows in frame.groupby("document", sort=True):
        encoded = tokenizer(
            records[document]["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        ).input_ids.to(cfg.device)
        ids_np = encoded[0].cpu().numpy()
        positions = torch.arange(cfg.max_length, device=cfg.device)[None]
        hidden = base.embed_tokens(encoded)
        query_indices = torch.tensor(
            rows.target_index.to_numpy(dtype=int) - 1, device=cfg.device
        )
        layer = base.layers[0].self_attn
        q_hidden = hidden[:, query_indices]
        q = layer.q_proj(q_hidden).view(
            1, len(rows), layer.config.num_attention_heads, layer.head_dim
        ).transpose(1, 2)
        k = layer.k_proj(hidden).view(
            1, cfg.max_length, layer.config.num_key_value_heads, layer.head_dim
        ).transpose(1, 2)
        cos, sin = base.rotary_emb(hidden, positions)
        q_cos = cos[:, query_indices].unsqueeze(1)
        q_sin = sin[:, query_indices].unsqueeze(1)
        q = q * q_cos + _rotate_half(q) * q_sin
        k = k * cos.unsqueeze(1) + _rotate_half(k) * sin.unsqueeze(1)
        pages = k[0].transpose(0, 1).reshape(
            cfg.max_length // page_size,
            page_size,
            layer.config.num_key_value_heads,
            layer.head_dim,
        ).float().mean(dim=1)
        groups = layer.config.num_attention_heads // layer.config.num_key_value_heads
        grouped_q = q[0].transpose(0, 1).reshape(
            len(rows), layer.config.num_key_value_heads, groups, layer.head_dim
        ).float()
        page_scores = torch.einsum("qhgd,phd->qphg", grouped_q, pages).mean(dim=(2, 3))
        page_scores = page_scores * layer.scaling
        remote = page_scores[:, :-recent_pages]
        recent_score = page_scores[:, -recent_pages:]
        top2 = remote.topk(2, dim=-1).values
        probabilities = torch.softmax(remote, dim=-1)
        retrieval = torch.stack((
            remote.max(dim=-1).values,
            remote.mean(dim=-1),
            remote.std(dim=-1),
            top2[:, 0] - top2[:, 1],
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1),
            recent_score.max(dim=-1).values,
            remote.max(dim=-1).values - recent_score.max(dim=-1).values,
        ), dim=-1).cpu().numpy()

        recent_indices = torch.arange(
            cfg.max_length - cfg.context.profile_recent_budget,
            cfg.max_length,
            device=cfg.device,
        )
        compact = base(
            input_ids=encoded[:, recent_indices],
            position_ids=positions[:, recent_indices],
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        embedding_parts.append(compact.hidden_states[0][0, -(len(rows) + 1) : -1].float().cpu().numpy())
        early_parts.append(compact.hidden_states[1][0, -(len(rows) + 1) : -1].float().cpu().numpy())
        retrieval_parts.append(retrieval)
        memory_parts.append(_memory_features(ids_np, rows, cfg.context.profile_recent_budget))
        ordered_documents.extend([document] * len(rows))
        del hidden, q, k, compact
        torch.cuda.empty_cache()

    del model
    embedding = np.concatenate(embedding_parts)
    early = np.concatenate(early_parts)
    retrieval = np.concatenate(retrieval_parts)
    memory = np.concatenate(memory_parts)
    documents = np.asarray(ordered_documents)
    confidence = frame[
        ["predicted_token_probability", "prediction_margin", "prediction_entropy"]
    ].to_numpy(dtype=np.float32)
    stateful = _stateful_features(np.concatenate((retrieval, memory), axis=1), documents)
    feature_sets = {
        "embedding_linear": (embedding, "pre_forward_vocab_lut_capable", 0.01),
        "early_layer1_linear": (early, "after_one_recent_attention_layer", 0.01),
        "token_bigram_memory": (memory, "pre_forward_o1_hash_state", 0.1),
        "page_retrieval": (retrieval, "pre_forward_page_landmark_scan", 0.1),
        "stateful_retrieval_surprise": (stateful, "pre_forward_stateful_page_and_hash", 0.1),
        "retrieval_plus_memory": (
            np.concatenate((retrieval, memory), axis=1),
            "pre_forward_page_and_hash",
            0.1,
        ),
        "speculative_confidence": (confidence, "post_sink_recent_requires_replay", 0.1),
    }
    labels = {
        "delta_ce_gt_0.1": frame.delta_ce.gt(cfg.context.long_context_delta_ce).astype(int).to_numpy(),
        "top1_changed": frame.top1_changed.astype(int).to_numpy(),
    }
    folds = _document_folds(frame)
    result_rows = []
    for mechanism, (features, availability, c) in feature_sets.items():
        for target, target_labels in labels.items():
            scores = _cv_linear(features, target_labels, folds, c=c)
            for rate in (0.25, 0.4, 0.5):
                selected = np.zeros(len(frame), dtype=bool)
                selected[np.argsort(scores)[-round(rate * len(frame)) :]] = True
                result_rows.append({
                    "mechanism": mechanism,
                    "availability": availability,
                    "target": target,
                    "feature_dimensions": features.shape[1],
                    "auc": roc_auc_score(target_labels, scores),
                    "average_precision": average_precision_score(target_labels, scores),
                    "route_fraction": selected.mean(),
                    "recall": float((selected & target_labels.astype(bool)).sum() / target_labels.sum()),
                    "precision": float(target_labels[selected].mean()),
                    "residual_delta_ce": float(np.where(selected, 0, frame.delta_ce).mean()),
                    "residual_top1_change": float(
                        np.where(selected, False, frame.top1_changed.astype(bool)).mean()
                    ),
                })
    result = pd.DataFrame(result_rows)
    csv_path = cfg.output_dir / "predictor_mechanism_comparison.csv"
    report_path = cfg.output_dir / "PREDICTOR_MECHANISM_RESULTS.md"
    result.to_csv(csv_path, index=False)

    best_need = result[
        result.target.eq("delta_ce_gt_0.1") & result.route_fraction.eq(0.25)
    ].sort_values("auc", ascending=False)
    best_top = result[
        result.target.eq("top1_changed") & result.route_fraction.sub(0.4).abs().lt(0.01)
    ].sort_values("auc", ascending=False)
    need_lines = [
        f"| {row.mechanism} | {row.availability} | {row.auc:.3f} | "
        f"{100*row.recall:.1f}% | {row.residual_delta_ce:.4f} |"
        for row in best_need.itertuples()
    ]
    top_lines = [
        f"| {row.mechanism} | {row.auc:.3f} | {100*row.recall:.1f}% | "
        f"{100*row.residual_top1_change:.2f}% |"
        for row in best_top.itertuples()
    ]
    report_path.write_text(
        "# Predictor mechanism comparison\n\n"
        f"All mechanisms use the same {len(frame)} targets from "
        f"{frame.document.nunique()} documents and a four-fold out-of-document split. "
        "The routed baseline is real 32K cached decode with one prefix sink token and "
        f"{cfg.context.profile_recent_budget - 1:,} recent tokens. "
        "The label and operating route fraction are held fixed across mechanisms.\n\n"
        "## Direct context need at 25% full-route rate\n\n"
        "| Mechanism | Availability/cost class | AUC | Recall | Residual ΔCE |\n"
        "|---|---|---:|---:|---:|\n" + "\n".join(need_lines)
        + "\n\n## Top-1 change at 40% full-route rate\n\n"
        "| Mechanism | AUC | Recall | Residual top-1 change |\n"
        "|---|---:|---:|---:|\n" + "\n".join(top_lines)
        + "\n\nEmbedding and early hidden are linear heads; token memory uses incremental "
        "count/last-position hashes; page retrieval uses layer-0 query-to-page-landmark "
        "statistics; stateful surprise adds an EWMA and deviation; speculative confidence "
        "uses sink-plus-recent probability, margin, and entropy and therefore requires replay.\n",
        encoding="utf-8",
    )
    return csv_path, report_path
