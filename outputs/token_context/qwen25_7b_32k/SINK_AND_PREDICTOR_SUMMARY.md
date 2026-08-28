# Attention sink and router mechanisms: consolidated result

## What was measured

Eight PG-19 documents were evaluated at 16K, 24K, and 32K. Each policy starts from the same dense prefill KV cache, then processes 64 teacher-forced decode queries. KV budgets are fixed at 128, 512, 2,048, or 8,192. Prefix sink policies are paired against recent-only, equal-count random remote, equal-count strided remote, and zero-value prefix controls. Confidence intervals bootstrap documents.

## Sink is real and useful

At 32K with a 2,048-position budget, recent-only has mean ΔCE 0.6495; prefix-1 + recent-2,047 reduces it to 0.0178. The paired improvement is -0.6317 (95% CI [-0.9585, -0.3646]). Prefix-16 also beats equal random remote by -0.5969 and equal strided remote by -0.6315. Zeroing prefix values makes ΔCE worse by +0.4707, so retained values carry functional state; keys are not merely absorbing softmax probability.

The strongest observed case is 16,384-token layer 4: its first four tokens receive 58.90% mean attention mass, 2408x the uniform expectation. Prefix-1 captures nearly all functional benefit, so the recommended cache allocation is one sink token rather than a 16–128-token sink block. This differs from compact-sequence recomputation: cached recent K/V retain representations built during dense prefill, whereas recomputation rebuilds every retained token under the compact context. The cached experiment is the relevant one for post-prefill KV eviction.

## Current implementation cost

| Context | Recent-only | Two-kernel sink-1 | Copy-then-one-kernel sink-1 |
|---:|---:|---:|---:|
| 16,384 | 1.059x | 0.708x | 0.808x |
| 24,576 | 1.093x | 0.737x | 0.828x |

Both generic implementations are too slow. A deployable path needs one fused kernel that streams one prefix KV and the contiguous recent window through the same online softmax without concatenation, a second attention launch, or LSE merge.

## Word category after adding sink

Sink removes the dominant structural failure, but it does not make lexical class irrelevant. At 16K with prefix-1, content-minus-function ΔCE is +0.2437 when only 127 recent tokens remain (95% CI [+0.0245, +0.5057]), versus +0.0015 with 8,191 recent tokens. The tight-minus-wide interaction is +0.2422 (95% CI [+0.0239, +0.4899]). Thus content words become significantly more context-sensitive as recent KV is tightened. Full-KV position curves nevertheless overlap strongly across categories: the effect is selective information/value sensitivity, not a universal increase in total remote attention mass. See `SINK_CATEGORY_16K_RESULTS.md` and the accompanying all-KV figures.

## Predictor mechanisms on the corrected sink-aware baseline

The target is whether 32K cached decode with prefix-1 + recent-2,047 still has ΔCE > 0.1 versus dense. Results below use a 25% full-route rate and held-out documents.

| Mechanism | Availability | AUC | Recall |
|---|---|---:|---:|
| speculative_confidence | post_sink_recent_requires_replay | 0.658 | 32.3% |
| stateful_retrieval_surprise | pre_forward_stateful_page_and_hash | 0.514 | 27.8% |
| token_bigram_memory | pre_forward_o1_hash_state | 0.504 | 24.8% |
| retrieval_plus_memory | pre_forward_page_and_hash | 0.501 | 24.8% |
| early_layer1_linear | after_one_recent_attention_layer | 0.497 | 24.8% |
| embedding_linear | pre_forward_vocab_lut_capable | 0.472 | 23.3% |
| page_retrieval | pre_forward_page_landmark_scan | 0.472 | 22.6% |

No pre-forward mechanism is reliable: page retrieval, token/bigram memory, stateful surprise, embedding, and early hidden all remain near random. The best post-forward verifier is speculative confidence (top-1-change AUC 0.808, 40% route recall 82.4%), but it requires a full replay and is therefore a quality safeguard rather than a speed optimization.

## Engineering decision

Always retain at least the first KV token. Do not spend a full 128-token page on sink if the kernel can represent a one-token segment. Keep long-context routing conservative until a materially better pre-forward signal is found; the immediate high-confidence optimization target is the fused sink+recent(+sparse-remote) decode kernel.

## Limits

The causal evidence covers Qwen2.5-7B, PG-19, BF16, dense prefill followed by 64 decode steps, and one accelerator family. It does not yet prove the same magnitude for chat/code data, other model families, quantized KV, or thousands of streaming steps. Those are replication targets, not assumptions hidden in the conclusion.
