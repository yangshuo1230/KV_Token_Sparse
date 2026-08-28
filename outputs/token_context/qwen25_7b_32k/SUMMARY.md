# KV routing study: consolidated trends

## Scope

Results use Qwen2.5-7B BF16 on PG-19 with 16K, 24K, or 32K contexts. The main
cached-decode study uses eight documents and 64 teacher-forced decode steps per
document. Confidence intervals resample whole documents. These results do not
yet establish the same effect for other models, chat/code data, quantized KV,
or thousands of streaming steps.

## Main findings

### 1. Attention sink is mandatory for cached eviction

After dense prefill, recent-only cache eviction causes a large structural
distribution shift. At 32K with a fixed 2,048-position read budget:

- Recent-only mean delta CE: 0.6495.
- Prefix-1 + Recent-2,047 mean delta CE: 0.0178.
- Paired improvement: -0.6317, 95% CI [-0.9585, -0.3646].

One prefix token captures nearly all sink benefit. Random or strided remote
tokens do not substitute for it, and zeroing its value removes the benefit.
The sink is therefore not only a softmax denominator slot; its cached value is
functionally important.

### 2. Content words become more context-sensitive as recent KV tightens

With Prefix-1 always retained at 16K, content-minus-function delta CE is:

| Recent tokens | Content - function delta CE | 95% CI |
|---:|---:|---:|
| 8,191 | +0.0015 | [-0.0376, +0.0434] |
| 2,047 | +0.0373 | [-0.0063, +0.0733] |
| 511 | +0.0638 | [-0.0320, +0.1478] |
| 127 | +0.2437 | [+0.0245, +0.5057] |

The tight-minus-wide interaction is +0.2422, 95% CI [+0.0239, +0.4899].
Thus word category is useful as a risk prior when KV is tight, but not as a
standalone routing decision.

### 3. Layer structure dominates average attention patterns

Marginal eta-squared for regional attention mass is 0.159-0.496 for layer and
only 0.0003-0.0015 for coarse word category. No individual layer/block or
layer/head/region cell survives global BH-FDR correction with eight documents.

At a lower-dimensional layer aggregate, five middle-remote effects survive
FDR: content has about 0.8-1.0 percentage points more middle-remote mass in
layers 0/12/16 and 2.0-3.7 points less in layers 24/27. The category signal is
distributed and reverses across depth rather than belonging to one stable
"content head". All 840 layer/head/region features classify content versus
function with held-out AUC 0.778, but this is not a causal router result.

### 4. Top-1 changes range from harmless to clearly harmful

For Prefix-1 cached decode at 16K:

| Recent | Change rate | Dense-correct lost among changes | Surface/punctuation among changes | Potential semantic among changes |
|---:|---:|---:|---:|---:|
| 127 | 21.7% | 30.6% | 15.3% | 46.8% |
| 511 | 16.2% | 18.1% | 24.1% | 50.6% |
| 2,047 | 9.4% | 18.8% | 20.8% | 58.3% |
| 8,191 | 6.1% | 19.4% | 45.2% | 32.3% |

"Potential semantic" is a diagnostic bucket, not a human judgment. The
strongest evidence is the exact correctness transition: dense predicts the
ground-truth token and compact does not. At Recent-127, 40.5% of changed tokens
also have delta CE above 0.5, so changes are not mostly formatting noise.

### 5. Adaptive inference performance

With the original recent/full routing schedule, V1 gives roughly 1.05x at 16K
and 1.06x at 24K. Sparse V2 is near break-even at 16K and about 1.04x at 24K.
Attention-only kernels show larger gains, but projections, MLPs, launches, and
cache management dominate end-to-end latency.

The current generic sink implementations are too slow:

- Two attention kernels plus LSE merge: about 0.71x/0.74x at 16K/24K.
- Copy prefix+recent then run one kernel: about 0.81x/0.83x.

A deployable implementation needs a fused online-softmax kernel that reads one
sink token, contiguous recent KV, and optional sparse remote pages in one launch.

### 6. Lightweight context-need predictors remain weak

On the corrected Prefix-1 + Recent-2,047 cached baseline, direct context-need
AUC is 0.658 for the post-forward confidence verifier and approximately
0.47-0.51 for pre-forward embedding, early-hidden, token-memory, page-retrieval,
and stateful-surprise features. The confidence verifier predicts Top-1 changes
better (AUC about 0.81) but requires full replay, so it is a quality safeguard
rather than a latency optimization.

## Engineering decision

1. Always retain at least Prefix-1.
2. Treat word category as a soft budget prior, especially below 2K recent KV.
3. Do not route from coarse category alone.
4. Prioritize a fused sink + recent + optional sparse-remote decode kernel.
5. Before training a small context-length network, expand sink-aware labels to
   at least 10K document-independent examples and verify an oracle dynamic
   policy beats equal-average-budget static windows.

## 24K dependency at a 256-token budget

This follow-up removes token-category conditioning. It uses 2,048 targets from
16 documents, Prefix-1 + Recent-255, and a dense 24K reference.

- Delta CE > 0.05: 48.00%.
- Delta CE > 0.10: 43.26%.
- Delta CE > 0.20: 35.40%.
- Delta CE > 0.50: 20.21%.
- Top-1 changed: 24.76%.

After per-token isotonic monotonicization, minimum-budget labels are 51.03% at
256, 8.64% at 512, 10.99% at 2K, 11.87% at 8K, and 17.48% at full 24K.

After the fact, long-dependent targets are less likely to repeat within the previous 256
tokens (43.1% versus 55.5%) and have higher local embedding novelty (0.437
versus 0.339), but the strongest individual trait reaches only AUC 0.574. These
target traits are diagnostic and unavailable to a causal router.

A 32-unit MLP over PCA-reduced current/last-4/last-16 embeddings and causal
history scalars achieves 46.9% exact budget accuracy in document-held-out
validation. Argmax under-routes 44.3% of tokens. The ordinal head has only
0.524 AUC for `required > 256`. A conservative ordinal P80 decision reduces
under-routing to 9.3%, but spends 15.1K KV on average. These features are
insufficient for an efficient learned router.

Detailed outputs are intentionally not versioned here. Every experiment and
report remains reproducible from the commands documented in the repository.
