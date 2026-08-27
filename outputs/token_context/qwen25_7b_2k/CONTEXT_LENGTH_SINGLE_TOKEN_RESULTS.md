# Single-token long-context ablation

Each sampled token was ablated in a separate forward pass. Only that query row
was prevented from reading history older than 128 tokens; no other selected token
was masked. This removes the cross-token intervention contamination in the prior
experiment.

## Main results

| Token category | ΔCE | KL(full‖masked) | Top-1 changed |
|---|---:|---:|---:|
| Content | 0.3135 | 0.2862 | 18.75% |
| Function | 0.2723 | 0.2600 | 20.61% |
| Negation | -0.1803 | 0.2551 | 17.65% |
| Number | -0.0581 | 0.1707 | 16.76% |
| Pronoun | -0.0537 | 0.0871 | 15.38% |
| Punctuation | 0.0665 | 0.1367 | 10.47% |
| Question word | 0.0480 | 0.0965 | 10.61% |

Document-paired function-minus-content contrast:

- ΔCE: `-0.0467`, 95% CI `[-0.1511,+0.0432]`
- KL: `-0.0223`, 95% CI `[-0.1926,+0.1412]`
- Top-1 change rate: `+0.0315`, 95% CI `[-0.0951,+0.1636]`

Thus there is no reliable overall function/content difference in this pilot.

## Interpretation by category

- Content and function words are both high-context-sensitivity categories.
- Pronouns and numbers have low next-token CE changes, but this should not be
  interpreted as no contextual representation: their full-to-masked KL remains
  nonzero.
- Negation has a negative mean ΔCE but high KL and only five documents contain
  sampled negation tokens; this category needs a larger controlled sample.
- Punctuation and question words have lower average dependence than content or
  function words, especially in top-1 changes.

Wikipedia is much more long-context-sensitive than PG-19 and HotpotQA. For
content/function ΔCE the corpus means are 0.784/0.973 on Wikipedia,
0.109/0.053 on PG-19, and 0.048/0.087 on HotpotQA. Formal testing must match
absolute position, document length, corpus, and syntactic role.

Run with:

```bash
python -m kvstudy context-diagnose-2k --config configs/token_context/qwen25_7b_2k.yaml
```
