# Long-context need profile and theoretical opportunity

This profile uses 1,024 target tokens from 16 PG-19 documents of exactly 32,768 tokens. A token is labelled as needing long context when recent-only decoding increases target cross-entropy by more than 0.1 nat relative to the full context. This is an oracle analysis label, not a deployable feature.

At a 2,048-token recent window, 254/1,024 tokens (24.80%) need long context; 75.20% do not.

## Ideal attention-traffic bound

KV is paged in 128-token blocks. V1 always reads the recent window and reads all remote blocks only for long-context tokens. Its mean KV reads are 9,668 tokens/query, a 70.50% reduction and 3.39x decode-attention upper bound. V2 gives long-context tokens 4,096 selected remote tokens; it reads 3,064 tokens/query, a 90.65% reduction and 10.69x upper bound.

These are bandwidth-only upper bounds. They exclude QKV/output projections, MLPs, router cost, block selection, kernel launches, and prediction errors; measured end-to-end results must be reported separately.
