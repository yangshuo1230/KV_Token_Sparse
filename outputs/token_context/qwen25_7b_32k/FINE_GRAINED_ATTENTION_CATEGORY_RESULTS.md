# Fine-grained attention analysis by token category

This follow-up avoids the strongest averaging artifacts in the earlier plot. Block differences retain document-level confidence intervals; region analysis is resolved by all 28 heads; remote blocks are also rank-aligned per query/head. PNG figures are generated locally and excluded from Git.

## Block-level content − function differences

Out of 1,280 layer/block cells, 418 have document-bootstrap 95% intervals excluding zero and 0 survive paired sign-flip tests with Benjamini–Hochberg FDR q<=0.05. The largest absolute mean difference is layer 27, block 0 (0–128), +6.78 percentage points (95% CI [+5.35, +8.24]).

## Head-resolved differences

Out of 840 layer/head/region cells, 343 exclude zero and 0 survive FDR q<=0.05. The strongest cell is layer 24, head 7, sink_block_mass: +20.54 pp (95% CI [+13.99, +28.14]).

## Rank-aligned and layer-aggregate remote results

Five of 60 tests survive FDR q<=0.05:

| Layer | Metric | Content − function | 95% CI | FDR q |
|---:|---|---:|---:|---:|
| 0 | remote_middle_mass | +0.78 pp | [+0.34, +1.17] | 0.0469 |
| 12 | remote_middle_mass | +0.98 pp | [+0.44, +1.44] | 0.0469 |
| 16 | remote_middle_mass | +0.98 pp | [+0.26, +1.81] | 0.0469 |
| 24 | remote_middle_mass | -2.02 pp | [-2.67, -1.38] | 0.0391 |
| 27 | remote_middle_mass | -3.74 pp | [-4.50, -3.08] | 0.0391 |

All surviving tests concern total middle-remote mass. Top-1/Top-4 remote concentration and effective-block-count differences do not survive FDR, so there is no corrected evidence that content queries consistently concentrate attention into fewer remote blocks. The sign reversal across depth is robust: content has slightly more middle-remote mass in layers 0/12/16 and less in layers 24/27.

## Marginal variance decomposition

These eta-squared values are one-factor marginal effects, not an additive ANOVA.

| metric             |   coarse_category |   document |   head |   layer |
|:-------------------|------------------:|-----------:|-------:|--------:|
| recent_mass        |            0.0007 |     0.0040 | 0.0195 |  0.2133 |
| remote_middle_mass |            0.0003 |     0.0300 | 0.0371 |  0.1587 |
| sink_block_mass    |            0.0015 |     0.0258 | 0.0125 |  0.4964 |

A document-held-out linear classifier using all 840 layer/head/region features predicts content versus function with ROC AUC 0.778. This measures whether category information exists in head-resolved patterns, not whether the features are a safe causal context-length router.

## Interpretation

Layer-level structure remains dominant. Uncorrected plots contain many localized category differences, but none of the individual block or head cells survive global FDR with only eight documents. The reliable signal is lower-dimensional and changes sign across depth. The AUC result suggests distributed category information, but it is exploratory and does not identify a safe routing head. See the CSV files for every effect, interval, permutation p-value, and FDR q-value.
