# 16K sink-aware category analysis

Queries use real cached decode with one prefix sink token and a recent window. Categories describe the next ground-truth token and are used for analysis only; they are not available to a causal router before generation.

## Content versus function as recent KV tightens

| Recent tokens | Content − function ΔCE (95% CI) | Need-rate difference (95% CI) |
|---:|---:|---:|
| 127 | +0.2437 [+0.0245,+0.5057] | +6.7 pp [-5.5,+19.1] |
| 511 | +0.0638 [-0.0320,+0.1478] | +10.9 pp [+3.0,+18.9] |
| 2,047 | +0.0373 [-0.0063,+0.0733] | +11.9 pp [+6.7,+17.1] |
| 8,191 | +0.0015 [-0.0376,+0.0434] | +6.1 pp [-0.1,+13.4] |

The difference-of-differences between 127 and 8,191 recent tokens is +0.2422 (95% CI [+0.0239, +0.4899]). A positive value means tightening recent KV increases the content/function gap.

## Distribution over the complete KV axis

The content and function position curves overlap strongly. Content-minus-function mass on the middle remote region is not consistently positive across layers; it ranges from -5.90 to +1.02 percentage points. Therefore the larger content-word quality loss under tight recent KV is not explained by a simple global rule that content queries always put more total mass on remote positions. Selective blocks, head specialization, and value sensitivity are more plausible explanations.

## Figures

PNG figures are generated locally and intentionally excluded from Git.

- `sink_full_kv_coarse_distributions_16k.png`: all 128 KV blocks, coarse categories, and all profiled layers.
- `sink_full_kv_fine_heatmaps_16k.png`: all KV blocks × fine lexical categories for each layer.
- `sink_full_kv_regions_by_category_16k.png`: sink block, middle remote, and recent-2K mass by layer.
- `sink_attention_fine_heatmap_16k.png`: sink-prefix detail by fine category.
- `sink_attention_coarse_distribution_16k.png`: sink-prefix per-query distributions at layers 4, 8, and 24.
- `sink_recent_tightening_categories_16k.png`: ΔCE, need rate, and top-1 change versus recent size.

All confidence intervals resample the eight documents, not individual adjacent tokens.
