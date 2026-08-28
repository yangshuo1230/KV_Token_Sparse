# Does attention sink have functional value?

The experiment uses 8 PG-19 documents, context lengths 16K, 24K, 32K, and 1,536 teacher-forced target tokens. Every policy starts from the same dense prefill cache and processes the same teacher-forced queries. Prefix, random-remote, strided-remote, zero-value-prefix, and recent-only policies read exactly the same number of KV positions. Confidence intervals resample whole documents.

## Functional prefix value

Negative values below mean that replacing recent KV with prefix sink KV improves cross-entropy relative to recent-only.

| Context | KV budget | Prefix-4 ΔCE difference | Prefix-16 ΔCE difference |
|---:|---:|---:|---:|
| 16,384 | 128 | -1.6444 [-2.1000,-1.2592] | -1.6231 [-2.0666,-1.2379] |
| 16,384 | 512 | -0.7752 [-1.2032,-0.4374] | -0.7679 [-1.1988,-0.4258] |
| 16,384 | 2,048 | -0.4441 [-0.6882,-0.2624] | -0.4362 [-0.6988,-0.2481] |
| 16,384 | 8,192 | -0.3319 [-0.5692,-0.1617] | -0.3311 [-0.5716,-0.1594] |
| 24,576 | 128 | -1.8797 [-2.1616,-1.6038] | -1.8369 [-2.1478,-1.5282] |
| 24,576 | 512 | -1.3073 [-1.7529,-0.9088] | -1.3103 [-1.7692,-0.8852] |
| 24,576 | 2,048 | -0.6528 [-1.1360,-0.3652] | -0.6537 [-1.1406,-0.3672] |
| 24,576 | 8,192 | -0.4188 [-0.7526,-0.2024] | -0.4178 [-0.7575,-0.2008] |
| 32,768 | 128 | -2.0434 [-2.4273,-1.6388] | -2.0390 [-2.4244,-1.6540] |
| 32,768 | 512 | -1.3341 [-1.6990,-0.9713] | -1.3296 [-1.6984,-0.9593] |
| 32,768 | 2,048 | -0.6322 [-0.9551,-0.3638] | -0.6324 [-0.9498,-0.3683] |
| 32,768 | 8,192 | -0.3305 [-0.5787,-0.1430] | -0.3275 [-0.5815,-0.1404] |

## Strongest full-context sink mass

Concentration is observed mass divided by uniform expected mass.

| Context | Layer | First-4 mass | Concentration | Heads >10x uniform |
|---:|---:|---:|---:|---:|
| 16,384 | 4 | 58.90% | 2407.9x | 99.9% |
| 24,576 | 4 | 58.43% | 3585.2x | 100.0% |
| 32,768 | 4 | 57.13% | 4675.4x | 99.9% |
| 16,384 | 24 | 46.55% | 1902.9x | 99.6% |
| 24,576 | 24 | 45.25% | 2776.6x | 99.7% |
| 16,384 | 8 | 44.63% | 1824.6x | 99.9% |
| 24,576 | 8 | 44.17% | 2710.5x | 100.0% |
| 32,768 | 8 | 43.06% | 3523.9x | 99.9% |

Matched random/strided controls and zero-value ablations are reported in `cached_sink_contrasts.csv`. A prefix is only considered a useful sink when it beats recent-only and matched remote controls, while zeroing its values removes the benefit. This separates functional cached state from attention mass alone.

## Conclusion

For this Qwen2.5-7B cached-decode setting, attention sink is both statistically visible and functionally valuable. A single prefix token captures most of the benefit; allocating 16–64 prefix positions rarely improves over prefix-1 and can waste recent capacity. The benefit persists through all four 16-token decode quartiles (`cached_sink_step_summary.csv`), beats non-prefix remote controls, and depends on the retained values rather than keys acting only as a softmax sink. This result applies to eviction after a dense prefill followed by 64 streaming decode steps; much longer generations remain a separate extrapolation risk.
