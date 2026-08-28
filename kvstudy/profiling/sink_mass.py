from __future__ import annotations

import torch


class AttentionSinkMassRecorder:
    """Record full-context prefix mass and value contribution for decode queries."""

    def __init__(
        self,
        layers: list[int],
        prefix_sizes: list[int],
        document: str,
        context_length: int,
    ) -> None:
        self.layers = set(layers)
        self.prefix_sizes = prefix_sizes
        self.document = document
        self.context_length = context_length
        self.decode_step = -1
        self.rows: list[dict] = []

    @torch.inference_mode()
    def __call__(
        self,
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scaling: float,
    ) -> None:
        if module.layer_idx not in self.layers:
            return
        groups = query.shape[1] // key.shape[1]
        q = query[0, :, 0].float()
        k = key[0].repeat_interleave(groups, dim=0).float()
        v = value[0].repeat_interleave(groups, dim=0).float()
        scores = torch.einsum("hd,hld->hl", q, k) * scaling
        log_denominator = torch.logsumexp(scores, dim=-1)
        weights = torch.softmax(scores, dim=-1)
        full_output_norm = torch.einsum("hl,hld->hd", weights, v).norm(dim=-1)
        length = key.shape[-2]
        for size in self.prefix_sizes:
            if size >= length:
                continue
            mass = torch.exp(torch.logsumexp(scores[:, :size], dim=-1) - log_denominator)
            contribution = torch.einsum(
                "hl,hld->hd", weights[:, :size], v[:, :size]
            ).norm(dim=-1)
            uniform_mass = size / length
            concentration = mass / uniform_mass
            self.rows.append({
                "document": self.document,
                "context_length": self.context_length,
                "decode_step": self.decode_step,
                "key_length": length,
                "layer": module.layer_idx,
                "prefix_size": size,
                "uniform_mass": uniform_mass,
                "attention_mass_mean": float(mass.mean().cpu()),
                "attention_mass_median": float(mass.median().cpu()),
                "attention_mass_max": float(mass.max().cpu()),
                "concentration_mean": float(concentration.mean().cpu()),
                "heads_above_10x_uniform": float(concentration.gt(10).float().mean().cpu()),
                "prefix_contribution_norm_mean": float(contribution.mean().cpu()),
                "full_output_norm_mean": float(full_output_norm.mean().cpu()),
                "contribution_norm_ratio_mean": float(
                    (contribution / full_output_norm.clamp_min(1e-8)).mean().cpu()
                ),
            })
