# Semantic KV Routing Study

This project tests whether content tokens in a semantic segment share remote
KV-block routes, and whether low-information tokens degrade a shared route.
The pipeline discards Q/K tensors and token attention matrices after each
document; only bounded document statistics are written during a run.

The repository contains the experiment code, configurations, and condensed
result reports. Large prepared corpora, per-document Parquet shards, caches,
and other regenerable intermediates are intentionally not versioned.

## Layout

```text
kvstudy/       data preparation, model hooks, metrics, experiments, reports
configs/       smoke, pilot, full, raw-QK, and 32K experiment configurations
outputs/       retained Markdown reports and aggregate CSV tables
tests/         unit tests for routing metrics
```

## Quick start

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m kvstudy prepare --config configs/smoke.yaml
python -m kvstudy run --config configs/smoke.yaml
python -m kvstudy summarize --config configs/smoke.yaml
```

Use `configs/qwen25_7b.yaml` for the preregistered run. The default data mix is
PG-19, Wikipedia, and HotpotQA. HotpotQA is used for external evidence labels:
LongBench exposes answers but generally does not expose evidence spans, so
answer-string matches must not be presented as human evidence.

Run one process per GPU for the full configuration:

```bash
CUDA_VISIBLE_DEVICES=0 python -m kvstudy run --config configs/qwen25_7b.yaml --shard-index 0 --num-shards 4
CUDA_VISIBLE_DEVICES=1 python -m kvstudy run --config configs/qwen25_7b.yaml --shard-index 1 --num-shards 4
CUDA_VISIBLE_DEVICES=2 python -m kvstudy run --config configs/qwen25_7b.yaml --shard-index 2 --num-shards 4
CUDA_VISIBLE_DEVICES=3 python -m kvstudy run --config configs/qwen25_7b.yaml --shard-index 3 --num-shards 4
```

The config uses `cuda:0` intentionally: each process sees its assigned physical
GPU as device zero. `summarize` automatically combines all matching shard files.

Set `position_mode: raw` to run the diagnostic ablation that skips RoPE only in
the final QK score computation. This is not a newly trained no-position model.

Outputs are written under the configured `output_dir`:

- `contexts.jsonl`: sampled text and evidence character spans
- `idf.json`: document-frequency-based word weights
- `document_metrics.parquet`: document-level sufficient statistics
- `summary.csv`: corpus-level means and document-bootstrap 95% CIs
- `contrasts.csv`: paired document-bootstrap differences for preregistered comparisons

Run `python -m kvstudy --help` for commands. Set `HF_HOME` to relocate model and
dataset downloads. No model activations or attention matrices are persisted.
The checked-in `outputs/` directory keeps only final reports and aggregate
tables; rerunning `prepare`/`run` recreates the deleted intermediates.

## Tests

```bash
pytest -q
```
