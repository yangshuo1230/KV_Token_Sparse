# Adaptive KV Routing Study

This repository studies token-dependent KV budgets for long-context causal
language models. The current reference model is Qwen2.5-7B on 16K–32K PG-19
contexts.

The deployable design is:

```text
Prefix-1 sink (always retained)
        +
contiguous recent KV
        +
optional sparse remote pages or full fallback
```

The consolidated results are in
`outputs/token_context/qwen25_7b_32k/SUMMARY.md`; core numeric values are in
`KEY_RESULTS.csv`. Detailed intermediate outputs are local-only and ignored.

## Code layout

```text
kvstudy/runtime/
  engine.py              shared autoregressive DecodeEngine
  kv_cache.py            cache indexing, cloning, and pruning

kvstudy/backends/
  policies.py            dense, recent, sink+recent, and sparse backends
  routed_inference.py    optimized FlashInfer attention integration
  block_attention.py     page landmarks and sparse page selection

kvstudy/profiling/
  context_length_mlp.py  24K dependency labels, token traits, and small MLP
  experiment.py          compact-context ablations
  sink_cached_experiment.py
  sparse_experiment.py
  full_kv_distribution.py
  ...                    offline profiling, statistics, and reports

kvstudy/semantic/        independent semantic-segment routing study
kvstudy/cli/main.py      thin CLI dispatch
```

Runtime code does not depend on profiling code. Profiling experiments use the
same `DecodeEngine` and plug in policy backends.

## Core API

```python
from kvstudy.backends import DenseBackend, SinkRecentBackend, SparseBackend
from kvstudy.runtime import DecodeEngine

engine = DecodeEngine(model)

dense_logits = engine.decode(cache, query_ids, start_position, DenseBackend())

compact_logits = engine.decode(
    cache,
    query_ids,
    start_position,
    SinkRecentBackend(total_budget=256, sink_tokens=1),
)
```

All backends implement the same two methods:

```python
backend.install(model)
backend.before_step(model, step)
```

## Main findings

- Prefix-1 is essential after dense prefill: at 32K and a 2,048-position
  budget, it reduces mean delta CE from 0.6495 to 0.0178.
- Content words become more context-sensitive than function words as recent KV
  is tightened, but category is only a weak routing prior.
- Layer identity dominates average attention layout; category information is
  distributed across heads and reverses direction across depth.
- Existing two-kernel and copy-based sink implementations are too slow. A
  fused sink + recent + sparse-remote online-softmax kernel is required.
- Current pre-forward context-need predictors remain weak.

## 24K / 256-token dependency experiment

The new unclassified-token experiment uses 16 documents, 2,048 target tokens,
24K context, and Prefix-1 + Recent-255 as the smallest budget.

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -m kvstudy context-run-top1-severity \
    --config configs/token_context/qwen25_7b_32k.yaml \
    --context-length 24576 --documents 16 --eval-tokens 128 \
    --budgets 256 512 2048 8192 --shard-index $i --num-shards 4 &
done
wait

python -m kvstudy context-profile-24k-mlp \
  --config configs/token_context/qwen25_7b_32k.yaml
```

At budget 256, 43.26% of tokens have delta CE above 0.1 and 24.76% change
Top-1. After the fact, long-dependent target tokens are less likely to repeat
within the previous 256 tokens and are modestly more novel in embedding space,
but the best individual trait AUC is only 0.574. These target traits are
diagnostic and are not exposed to the causal MLP.

The document-held-out 32-unit multiclass MLP reaches 46.9% exact budget
accuracy but under-routes 44.3% of tokens. An ordinal MLP has only 0.524 AUC
for deciding whether more than 256 KV is needed. A conservative ordinal P80
decision reduces under-routing to 9.3% while increasing mean budget to 15.1K.
This is not a deployable router.

## Setup

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Install optimized decode dependencies separately:

```bash
python -m pip install -r requirements-kernels.txt
```

List all profiling and benchmark commands:

```bash
python -m kvstudy --help
```

## Tests

```bash
pytest -q
```
