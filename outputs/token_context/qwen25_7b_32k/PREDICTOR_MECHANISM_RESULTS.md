# Predictor mechanism comparison

All mechanisms use the same 512 targets from 8 documents and a four-fold out-of-document split. The routed baseline is real 32K cached decode with one prefix sink token and 2,047 recent tokens. The label and operating route fraction are held fixed across mechanisms.

## Direct context need at 25% full-route rate

| Mechanism | Availability/cost class | AUC | Recall | Residual ΔCE |
|---|---|---:|---:|---:|
| speculative_confidence | post_sink_recent_requires_replay | 0.658 | 32.3% | 0.0131 |
| stateful_retrieval_surprise | pre_forward_stateful_page_and_hash | 0.514 | 27.8% | 0.0111 |
| token_bigram_memory | pre_forward_o1_hash_state | 0.504 | 24.8% | 0.0101 |
| retrieval_plus_memory | pre_forward_page_and_hash | 0.501 | 24.8% | 0.0142 |
| early_layer1_linear | after_one_recent_attention_layer | 0.497 | 24.8% | 0.0105 |
| embedding_linear | pre_forward_vocab_lut_capable | 0.472 | 23.3% | 0.0093 |
| page_retrieval | pre_forward_page_landmark_scan | 0.472 | 22.6% | 0.0255 |

## Top-1 change at 40% full-route rate

| Mechanism | AUC | Recall | Residual top-1 change |
|---|---:|---:|---:|
| speculative_confidence | 0.808 | 82.4% | 2.34% |
| page_retrieval | 0.539 | 45.6% | 7.23% |
| token_bigram_memory | 0.526 | 42.6% | 7.62% |
| retrieval_plus_memory | 0.514 | 42.6% | 7.62% |
| stateful_retrieval_surprise | 0.510 | 39.7% | 8.01% |
| embedding_linear | 0.490 | 38.2% | 8.20% |
| early_layer1_linear | 0.449 | 36.8% | 8.40% |

Embedding and early hidden are linear heads; token memory uses incremental count/last-position hashes; page retrieval uses layer-0 query-to-page-landmark statistics; stateful surprise adds an EWMA and deviation; speculative confidence uses sink-plus-recent probability, margin, and entropy and therefore requires replay.
