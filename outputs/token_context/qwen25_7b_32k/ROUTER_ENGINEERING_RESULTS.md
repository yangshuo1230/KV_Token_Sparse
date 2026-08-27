# Lightweight router and KV-cache engineering results

## Causal predictors

Two predictors were evaluated without using the true target token or its
post-hoc POS label:

1. `input_token_lookup` maps the currently known input token ID to a smoothed
   probability that the next target is a content token. It requires one table
   lookup before the forward pass.
2. `draft_token_lookup` runs Qwen2.5-0.5B with a 128-position sink-aware context,
   then maps its top candidate ID to the same probability. It is more accurate
   but has auxiliary-model cost.

All predictions are out-of-document: four-fold cross-validation trains each
lookup on 12 documents and evaluates it on four held-out documents. The input
lookup obtains target-type ROC AUC 0.6322; the draft lookup obtains 0.7493.

## Equal-average-budget routing

Each router chooses between a low and high budget. The fraction assigned high
budget is fixed so its mean KV count matches a measured static intermediate
budget. Results use four sink tokens.

| Router | Low/high | Mean/static KV | Router ΔCE | Static ΔCE | Difference (95% CI) |
|---|---:|---:|---:|---:|---:|
| Input lookup | 128/2,048 | 512.4/512 | 0.2674 | 0.1534 | +0.1140 `[+0.0750,+0.1574]` |
| Draft lookup | 128/2,048 | 512.4/512 | 0.2517 | 0.1534 | +0.0982 `[+0.0427,+0.1541]` |
| Input lookup | 512/2,048 | 1,023.5/1,024 | 0.1248 | 0.0831 | +0.0417 `[+0.0059,+0.0822]` |
| Draft lookup | 512/2,048 | 1,023.5/1,024 | 0.1066 | 0.0831 | +0.0236 `[-0.0041,+0.0536]` |
| Input lookup | 2,048/8,192 | 4,094/4,096 | 0.0416 | 0.0184 | +0.0232 `[-0.0116,+0.0713]` |
| Draft lookup | 2,048/8,192 | 4,094/4,096 | 0.0183 | 0.0184 | -0.0001 `[-0.0139,+0.0127]` |

The lightweight input lookup is not Pareto competitive. The auxiliary draft
lookup matches static quality only at the largest plan, but the difference is
not reliable and the auxiliary model's compute is not included in the nominal
KV budget. This is not a demonstrated speedup.

The oracle upper bound is large—for example, oracle ΔCE is -0.0579 at mean KV
4,094—so token-specific routing could still work with a substantially better
predictor of *per-token benefit*. The present results show that predicting the
broad content/function type is not sufficient, even though type has a strong
average association with context need.

## Real `past_key_values` prototype

`kvstudy/token_context/kv_cache.py` implements fixed-budget pruning for legacy
and `DynamicCache` objects. It preserves a configurable prefix sink and the
most recent positions. The code was exercised through real one-token cached
decoding, not only tensor-shape unit tests.

On Qwen2.5-7B with an 8,192-token prefill, the measured BF16 KV storage is:

| Retained KV | KV storage | Median decode latency |
|---:|---:|---:|
| 128 | 7.0 MiB | 13.30 ms |
| 512 | 28.0 MiB | 13.39 ms |
| 1,024 | 56.0 MiB | 13.42 ms |
| 2,048 | 112.0 MiB | 13.63 ms |
| 4,096 | 224.0 MiB | 13.37 ms |
| 8,192 | 448.0 MiB | 13.66 ms |

KV memory scales exactly with the retained budget, but batch-one decode latency
on the available PPU is essentially flat over this range; non-attention work
and kernel overhead dominate. The prototype currently copies tensors when it
prunes: median copy time ranges from 2.4 ms at 128 KV to 22.0 ms at 4,096 KV.
Applying that copy every step would make latency worse. A production version
needs a preallocated ring buffer or paged-cache index update instead of tensor
copying.

## Engineering decision

The code establishes the integration path and the memory benefit, but the
current router does not deliver a verified end-to-end performance improvement.
The next useful iteration is a low-cost predictor trained directly on marginal
context benefit, ideally from an early hidden layer of the main model, followed
by integration with a zero-copy paged KV backend. Broad POS prediction alone
should not be shipped as a cache controller.

Exact router results are in `router_evaluation.csv`; hardware measurements are
in `cache_benchmark.csv`.
