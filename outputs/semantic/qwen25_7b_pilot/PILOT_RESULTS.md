# Qwen2.5-7B pilot results

This is a 12-document engineering pilot (4 documents per corpus), not a test of
the preregistered success criteria. It uses layers 0, 13, and 27 with 2,048-token
contexts. The full configuration remains in `configs/semantic/qwen25_7b.yaml`;
prepared documents are regenerable artifacts and are not versioned.

Remote-KV macro means across the pilot strata:

| Measure | Result |
|---|---:|
| Semantic Jaccard | 0.5763 |
| Length-matched fixed Jaccard | 0.5726 |
| Random Jaccard | 0.2290 |
| Semantic shared retained ratio | 0.9341 |
| Semantic + 2 blocks retained ratio | 0.9825 |
| Mean content-vs-all delta pull | -0.0163 |
| All-token evidence recall | 0.2433 |
| Content-token evidence recall | 0.2373 |

The pilot shows a small H1 effect over the length-matched control and no support
for H2. These values must not be treated as confirmatory because the sample is
small and was examined during implementation.
