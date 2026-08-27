# V2 sparse-context quality

Qwen2.5-7B was evaluated on 16 PG-19 documents of 32,768 tokens (1,024 target tokens). Every query keeps the most recent 2,048 tokens and attends 4,096 remote tokens selected in 128-token pages. Page relevance is the first-layer query/key-landmark score, refreshed once for each 64-token evaluation block.

Mean sparse ΔCE is 0.0253, full-to-sparse KL is 0.0425, and top-1 changes on 10.16% of targets. Paired comparisons and document-level bootstrap confidence intervals are in `sparse_context_summary.csv`.

The quality run recomputes the selected compact sequence with an explicit causal mask and original RoPE positions. Kernel timing is measured independently with the paged FlashInfer path in `decode_attention_benchmark.csv`.
