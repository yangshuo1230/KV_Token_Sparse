# Adaptive block-KV inference: measured results

## Profile before optimization

The profile covers 1,024 target tokens in 16 PG-19 documents of 32,768 tokens. With a 2,048-token recent window and a 0.1-nat ΔCE criterion, 24.80% need long context and 75.20% do not.

With 128-token pages, the ideal KV-read model predicts 3.39x for V1 and 10.69x for V2. These are attention-only bandwidth bounds, not end-to-end claims.

## Implementation

V1 keeps the full DynamicCache and chooses a zero-copy recent tensor view or the full KV for each query. V2 always runs recent attention, reads query-selected non-contiguous remote pages with FlashInfer's paged decode kernel, and merges the two softmax states using their log-sum-exp values. Page selection uses a first-layer query and one mean-key landmark per 128-token page. The page table is reusable for a block of decode steps.

## Attention-kernel timing

The mixture uses the measured 24.80% long-token rate; page-selection cost is amortized over 128 tokens.

| Context | V1 mixture | V2 mixture |
|---:|---:|---:|
| 16,384 | 1.75x | 2.25x |
| 24,576 | 1.92x | 2.65x |
| 32,768 | 2.28x | 2.80x |

## Real Qwen2.5-7B decode

Each policy starts from an identical prefill cache and decodes 128 tokens. The routing schedules use the oracle *rate* only to measure compute; they do not use oracle labels and are not deployable quality results. Mean wall-clock latency is reported because the mixed distribution is bimodal.

| Context | Dense ms/token | V1 ms/token (speedup) | V2 ms/token (speedup) |
|---:|---:|---:|---:|
| 16,384 | 15.373 | 15.514 (0.991x) | 16.138 (0.953x) |
| 24,576 | 16.999 | 16.118 (1.055x) | 16.971 (1.002x) |

V1 produces a real end-to-end gain at 24K, but not at 16K. V2's two-kernel execution and LSE merge erase nearly all 24K gain and regress at 16K; a fused recent-plus-paged kernel is the next operator target.

## V2 quality at 32K

The sparse policy attends 6,144 tokens (2,048 recent + 4,096 remote). Its mean ΔCE is 0.0253, improving over recent-2,048 by 0.0361 (95% CI [0.0020, 0.0936]). It is statistically indistinguishable from static recent-4,096 in ΔCE (difference +0.0077, 95% CI [-0.0121, +0.0328]). Top-1 changes drop by 2.83 percentage points versus recent-2,048.

## Lightweight-router status

The one-table-lookup input-token predictor reaches type AUC 0.632; the 0.5B draft-token lookup reaches 0.733. Neither is Pareto competitive at the small plans. At the 2,048/8,192 plan the draft router matches static-4,096 ΔCE within uncertainty (difference +0.0010, 95% CI [-0.0138, +0.0174]), but its auxiliary-model cost is not included. Therefore the repository demonstrates an engineering speedup opportunity and a working kernel path, not yet a deployable quality-preserving learned router.
