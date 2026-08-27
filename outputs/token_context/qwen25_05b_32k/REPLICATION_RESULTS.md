# Qwen2.5-0.5B replication

The sink-aware target-token experiment was repeated with Qwen2.5-0.5B on the
same 16 PG-19 documents, target positions, cache budgets, and sink allocations
as the Qwen2.5-7B run. This controls text composition while testing whether the
category relationship transfers across model scale.

Results below use four sink tokens and document-bootstrap 95% confidence
intervals.

| Total KV budget | Content ΔCE | Function ΔCE | Content − function (95% CI) |
|---:|---:|---:|---:|
| 128 | 0.5886 | 0.1032 | +0.4853 `[+0.2986,+0.6962]` |
| 512 | 0.2685 | 0.0222 | +0.2464 `[+0.0939,+0.4263]` |
| 2,048 | 0.1212 | -0.0205 | +0.1417 `[+0.0393,+0.2715]` |
| 8,192 | -0.0173 | 0.0021 | -0.0193 `[-0.0711,+0.0283]` |

The pattern closely replicates Qwen2.5-7B: content targets require more omitted
context at 128, 512, and 2,048 positions, while the distinction disappears by
8,192 positions. Effect sizes are also similar, not merely directionally
consistent. The result is therefore unlikely to be unique to the 7B model.

This replication supports moving to a deployable predictor. Since the true
target type is unavailable before generation, that predictor must use features
from the preceding query position, such as the full model's candidate-token
distribution or a small learned projection of the hidden state.

Aggregate outputs are retained alongside this report. Per-target Parquet data
is regenerable and not versioned.
