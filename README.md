# KV Routing and Token Context Studies

This repository contains two related but independent research tracks for causal
language models. They share data loading, token annotation, and model adapters,
but have separate code, configurations, and results.

## Research tracks

### 1. Semantic-segment KV routing

`kvstudy/semantic/` tests whether tokens in a semantic segment can share a
remote KV-block route without losing much attention mass. It compares semantic
segments with length-matched and random controls, and includes hidden-state,
repeated-word, evidence-retrieval, and raw-QK diagnostics.

- Configs: `configs/semantic/`
- Results: `outputs/semantic/`
- Main commands: `semantic-run`, `semantic-summarize`, and
  `semantic-diagnose-representations`

### 2. Prediction-token context requirements

`kvstudy/token_context/` measures how removing distant context changes the
next-token distribution for different target-token categories. This track is
about token-specific context/KV retention and does not use semantic segments as
the experimental treatment.

- Configs: `configs/token_context/`
- Results: `outputs/token_context/`
- Main commands: `context-prepare`, `context-run`, `context-summarize`,
  `context-evaluate-router`, and `context-benchmark-cache`

The reports named `LONG_CONTEXT_RESULTS.md` and `TARGET_TOKEN_RESULTS.md`
predate the sink-aware experiment. They compare a full context with suffix-only
inputs and are kept as baseline evidence, not as a deployable KV-cache policy.
The current experiment compares recent-only with sink-plus-recent policies at
the same total KV budget.

The checked-in systematic reports cover Qwen2.5-7B and a same-document
Qwen2.5-0.5B replication. Both find that content targets need more context than
function targets at 128, 512, and 2,048 retained positions, with the difference
disappearing at 8,192 positions.

## Shared infrastructure

```text
kvstudy/data.py       PG-19, Wikipedia, and HotpotQA preparation
kvstudy/segments.py   token/POS categories and semantic segmentation
kvstudy/model.py      model loading and Q/K projection
artifacts/            regenerable corpora and IDF data (gitignored)
```

Large prepared corpora, per-document Parquet shards, model caches, Q/K tensors,
and attention matrices are not versioned. Only condensed reports and selected
aggregate tables are checked in.

## Setup

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Prepare shared pilot data and run the semantic smoke test:

```bash
python -m kvstudy prepare --config configs/semantic/smoke.yaml
python -m kvstudy semantic-run --config configs/semantic/smoke.yaml
python -m kvstudy semantic-summarize --config configs/semantic/smoke.yaml
```

Prepare and run the sink-aware 32K target-token experiment with four workers:

```bash
python -m kvstudy context-prepare \
  --config configs/token_context/qwen25_7b_32k.yaml
CUDA_VISIBLE_DEVICES=0 python -m kvstudy context-run --config configs/token_context/qwen25_7b_32k.yaml --shard-index 0 --num-shards 4
CUDA_VISIBLE_DEVICES=1 python -m kvstudy context-run --config configs/token_context/qwen25_7b_32k.yaml --shard-index 1 --num-shards 4
CUDA_VISIBLE_DEVICES=2 python -m kvstudy context-run --config configs/token_context/qwen25_7b_32k.yaml --shard-index 2 --num-shards 4
CUDA_VISIBLE_DEVICES=3 python -m kvstudy context-run --config configs/token_context/qwen25_7b_32k.yaml --shard-index 3 --num-shards 4
python -m kvstudy context-summarize --config configs/token_context/qwen25_7b_32k.yaml
```

Evaluate the causal lightweight router and exercise real `DynamicCache`
pruning:

```bash
python -m kvstudy context-evaluate-router \
  --config configs/token_context/qwen25_7b_32k.yaml \
  --draft-config configs/token_context/qwen25_05b_32k.yaml
python -m kvstudy context-benchmark-cache \
  --config configs/token_context/qwen25_7b_32k.yaml \
  --prefill-tokens 8192 --repeats 20
```

Use `python -m kvstudy --help` to list commands. Set `HF_HOME` to relocate
model and dataset downloads.

## Full semantic run

`configs/semantic/qwen25_7b.yaml` specifies the 1,000-document run over PG-19,
Wikipedia, and HotpotQA. Run one process per GPU; each process sees its assigned
physical GPU as `cuda:0`.

```bash
CUDA_VISIBLE_DEVICES=0 python -m kvstudy semantic-run --config configs/semantic/qwen25_7b.yaml --shard-index 0 --num-shards 4
CUDA_VISIBLE_DEVICES=1 python -m kvstudy semantic-run --config configs/semantic/qwen25_7b.yaml --shard-index 1 --num-shards 4
CUDA_VISIBLE_DEVICES=2 python -m kvstudy semantic-run --config configs/semantic/qwen25_7b.yaml --shard-index 2 --num-shards 4
CUDA_VISIBLE_DEVICES=3 python -m kvstudy semantic-run --config configs/semantic/qwen25_7b.yaml --shard-index 3 --num-shards 4
```

`semantic-summarize` automatically combines matching shard files. Setting
`position_mode: raw` skips RoPE only in the final QK score diagnostic; it does
not create a no-position language model.

## Tests

```bash
pytest -q
```
