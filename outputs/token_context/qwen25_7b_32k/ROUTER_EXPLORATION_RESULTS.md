# Ultra-light router exploration

All accuracy numbers use four-fold out-of-document validation on 1,024 targets from 16 independent 32K PG-19 documents.

## Embedding category LUT

A linear head over the current input-token embedding predicts whether the next token is a content token with ROC AUC 0.721. After training, the embedding projection is folded into `embedding_type_lut.npy`, so runtime is one FP16 vocabulary-table lookup. However, its AUC for actual `ΔCE > 0.1` context need is only 0.538. Better category prediction does not solve routing.

## Speculative confidence verifier

A three-feature logistic head over recent-only probability, top-2 margin, and entropy predicts whether full context changes top-1 with ROC AUC 0.833. Routing 40% to full recalls 87.2% of top-1 changes and leaves a 1.66% residual change rate. But the signal exists only after recent-only inference. Replaying the full model for rejected tokens gives an estimated 24K speedup of 0.760x, i.e. a slowdown.

## Decision

Keep the vocabulary LUT as an essentially free feature, but do not use word class as the routing decision by itself. The confidence head is useful as a quality verifier, not as a latency optimization on the current full-replay design. A future router needs a pre-forward retrieval/surprise signal or a fused partial replay that avoids repeating MLP and projection work. Full operating points are in `router_exploration.csv`.
