# 32K-context final-window ablation

Four PG-19 documents were extended to exactly 32,768 Qwen2.5-7B tokens. A full
32K forward was compared with suffix-only forwards of 128, 512, and 2,048
tokens. The suffix forwards retained the original absolute `position_ids`, and
only the final 127 next-token positions were scored. Thus this tests how much
information is lost when the model can see only a short suffix, rather than
re-basing a short document at position zero.

## Content versus function words

| Visible suffix | Content ΔCE | Function ΔCE | Function − Content |
|---:|---:|---:|---:|
| 128 | 0.8685 | 0.6726 | -0.1959 |
| 512 | 0.1343 | 0.0609 | -0.0734 |
| 2,048 | 0.0594 | 0.0040 | -0.0554 |

Document-paired bootstrap contrasts (function minus content):

- 128-token suffix: `-0.2604`, 95% CI `[-0.5651,+0.0438]`
- 512-token suffix: `-0.0858`, 95% CI `[-0.1259,-0.0569]`
- 2,048-token suffix: `-0.0615`, 95% CI `[-0.1296,-0.0173]`

At 32K context, information words show consistently larger next-token CE
degradation when long history is removed. The difference is strongest for a
128-token suffix. With only four documents, this is an exploratory result, but
it is more consistent than the earlier 2K position-bin analysis.

KL and top-1 changes are less decisive. For example, at a 128-token suffix the
content/function KL means are 0.921/0.785, while at 512 they are 0.113/0.108.
This means information words mainly lose probability on the correct next token;
the whole-distribution and top-1 effects are not always proportionally larger.

Other categories at the 128-token suffix show substantial heterogeneity:
`other` ΔCE 1.259, question 1.135, negation 1.194, content 0.869, function
0.673, pronoun 0.558, punctuation 0.515. Rare categories have too few examples
for reliable ranking.

This suffix-only implementation has been superseded by the fixed-budget,
sink-aware experiment in `kvstudy/token_context/experiment.py`. This report is
retained only as historical baseline evidence.
