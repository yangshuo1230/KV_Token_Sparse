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
- Main commands: `context-diagnose-2k`, `context-prepare-32k`,
  `context-run-32k`, and `context-run-target-32k`

The retained 32K reports predate the sink-aware experiment. They compare a full
context with suffix-only inputs and are kept as baseline evidence, not as a
deployable KV-cache policy. New cache-policy experiments must retain both the
configured attention-sink prefix and the recent window.

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

Run the existing target-token 32K baseline:

```bash
python -m kvstudy context-prepare-32k \
  --config configs/token_context/qwen25_7b_32k.yaml
python -m kvstudy context-run-target-32k \
  --config configs/token_context/qwen25_7b_32k.yaml
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
