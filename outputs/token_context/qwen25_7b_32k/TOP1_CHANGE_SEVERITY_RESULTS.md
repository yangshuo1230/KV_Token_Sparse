# What kind of Top-1 changes occur?

This analysis compares dense and prefix-1 + recent cached decode at 16K. `Definite harm` means dense Top-1 equals the ground-truth token but compact Top-1 does not; `definite benefit` is the reverse. Surface/punctuation changes normalize whitespace and case. `Embedding-near` uses Qwen input-embedding cosine >= 0.8. The remaining bucket is a potential semantic change, not a human semantic-error judgment. Percentages after the first column are fractions of changed tokens.

| Recent | Top-1 change rate | Definite harm | Definite benefit | Surface/punct | Embedding-near | Potential semantic |
|---:|---:|---:|---:|---:|---:|---:|
| 127 | 21.7% | 30.6% | 7.2% | 15.3% | 0.0% | 46.8% |
| 511 | 16.2% | 18.1% | 7.2% | 24.1% | 0.0% | 50.6% |
| 2,047 | 9.4% | 18.8% | 2.1% | 20.8% | 0.0% | 58.3% |
| 8,191 | 6.1% | 19.4% | 3.2% | 45.2% | 0.0% | 32.3% |

## Content versus function outcomes

Rates below use all tokens in that category, rather than only changed tokens.

| Recent | Content changed | Content definite harm | Function changed | Function definite harm |
|---:|---:|---:|---:|---:|
| 127 | 24.3% | 7.4% | 16.0% | 3.1% |
| 2,047 | 11.7% | 1.3% | 8.4% | 3.1% |
| 8,191 | 7.4% | 1.3% | 6.1% | 1.5% |

## Severity diagnostics among changed tokens

| Recent | ΔCE > 0.1 | ΔCE > 0.5 | Embedding cosine < 0.5 |
|---:|---:|---:|---:|
| 127 | 59.5% | 40.5% | 83.8% |
| 511 | 54.2% | 36.1% | 78.3% |
| 2,047 | 60.4% | 20.8% | 79.2% |
| 8,191 | 45.2% | 9.7% | 51.6% |

Ground-truth correctness transitions are the strongest evidence. Embedding similarity is only a diagnostic proxy; single-token substitutions can change meaning even at high cosine, while two different tokens can be equivalent in a larger phrase. Severity buckets are assigned in priority order: correctness transition, surface/punctuation, then embedding similarity. CSV files include target ranks, probabilities, ΔCE, category, cosine-threshold sensitivity, and all changed token triples for independent inspection.
