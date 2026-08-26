# Long-context ablation: function versus content tokens

For each document, a normal teacher-forcing forward pass was compared with a
second pass where sampled function/content query rows could attend only to the
first 4 sink tokens plus the most recent 128 tokens. Other rows remained causal.
Metrics are measured on the next-token distribution at the selected position.

This is an exploratory group intervention: sampled rows of both types are
masked in one pass, so later selected rows can see earlier modified states. It
does not yet isolate one token with a separate forward pass.

| Token type | Δ cross-entropy | KL(full || masked) | top-1 changed |
|---|---:|---:|---:|
| Content | 0.2502 | 0.2696 | 16.1% |
| Function | 0.2298 | 0.2502 | 17.6% |

Document-paired function-minus-content differences:

- Δ cross-entropy: `+0.1454`, 95% CI `[-0.0925,+0.4061]`
- KL: `+0.1321`, 95% CI `[-0.1045,+0.4018]`
- top-1 change rate: `+0.0971`, 95% CI `[-0.0177,+0.2323]`

Thus the pilot does not establish a reliable overall difference in context-length
requirement. The distributions are highly heavy-tailed: median ΔCE is nearly
zero for both types, while a small number of positions show large degradation.

By corpus, Wikipedia has much larger effects (content/function ΔCE 0.744/0.873)
than HotpotQA (0.010/0.028) or PG-19 (0.015/0.205). This likely reflects
document length/truncation and entity-dense text, so corpus and absolute position
must be controlled in the formal experiment.

The implementation is `kvstudy/context_ablation.py`, invoked by:

```bash
python -m kvstudy diagnose-context-length --config configs/qwen25_7b_pilot.yaml
```

