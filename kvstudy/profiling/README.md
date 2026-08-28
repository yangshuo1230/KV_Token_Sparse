# Profiling modules

Everything in this package is offline experimentation. Production-style decode
logic lives in `kvstudy.runtime`; attention implementations live in
`kvstudy.backends`.

## Primary experiments

- `top1_severity.py`: dense-versus-budget label collection.
- `context_length_mlp.py`: 24K dependency profile, token traits, and small MLP.
- `sink_cached_experiment.py`: fixed-budget sink controls.
- `sparse_experiment.py`: query-selected remote-page quality.
- `inference_benchmark.py`: end-to-end policy timing.

## Attention diagnostics

- `full_kv_distribution.py`: block/head-resolved attention collection.
- `fine_attention_analysis.py`: category and locality statistics.
- `sink_mass.py`: prefix attention and value contribution.

## Legacy/replication analyses

The remaining modules retain earlier ablations, router baselines, summaries,
and report generators. They are intentionally isolated here so they do not
expand the runtime API.
