# Generating content versus function tokens at 32K context

This is the target-token version of the long-context ablation. For target token
`x_i`, logits come from query position `i-1`, and the POS/category label comes
from `x_i`. The target token is never included in its own input context.

Four PG-19 documents were evaluated at exactly 32,768 tokens. Full-context
prediction was compared with suffix-only contexts of 128, 512, and 2,048 tokens,
using the original absolute position IDs and scoring only the final 127 targets.

## Target-token results

| Visible suffix | Content ΔCE | Function ΔCE | Function − Content |
|---:|---:|---:|---:|
| 128 | 1.2443 | 0.4815 | -0.7628 |
| 512 | 0.2351 | 0.0246 | -0.2105 |
| 2,048 | 0.0701 | -0.0096 | -0.0797 |

Document-macro paired contrasts differ slightly from token-macro values:

- 128: `-0.7961`, bootstrap 95% CI `[-1.0186,-0.5906]`
- 512: `-0.2428`, bootstrap 95% CI `[-0.4825,-0.0312]`
- 2,048: `-0.0982`, bootstrap 95% CI `[-0.1745,-0.0169]`

All four documents show larger content-token degradation for the 128-token
suffix. Three of four do so at 512 and 2,048 tokens. With only four documents,
the bootstrap intervals are exploratory rather than confirmatory.

At a 128-token suffix, content/function KL is 1.167/0.617 and top-1 changes are
45.0%/21.7%. Thus the difference is not limited to target probability: the full
prediction distribution and most likely target are also more sensitive for
content words.

The fraction with ΔCE > 0.1 is content/function 59.2%/50.7% at 128,
35.6%/28.3% at 512, and 25.7%/19.6% at 2,048 tokens.

Run with:

```bash
python -m kvstudy run-target-long-context --config configs/qwen25_7b_32k.yaml
```

