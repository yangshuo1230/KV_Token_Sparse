# Raw-QK positional ablation

This diagnostic uses the same 12 documents, layers, heads, masks, and sampling
as the Qwen2.5-7B pilot. Model hidden states are still produced by the original
RoPE model; only the final Q/K vectors used for analysis are left unrotated.
It is not a no-position language model.

Remote-KV document-macro results:

| Measure | RoPE | Raw Q/K | Raw - RoPE |
|---|---:|---:|---:|
| Content-to-content-route Jaccard | 0.6150 | 0.6699 | +0.0549 |
| Function-to-content-route Jaccard | 0.5921 | 0.6429 | +0.0508 |
| Content JS similarity | 0.7262 | 0.7550 | +0.0289 |
| Function JS similarity | 0.7141 | 0.7414 | +0.0273 |
| Content top-mass overlap | 0.4663 | 0.4251 | -0.0413 |
| Function top-mass overlap | 0.4403 | 0.3892 | -0.0511 |

Function-minus-content Jaccard changes from +0.0024 to -0.0038 at the document
macro level; the difference-in-differences is -0.0062 with a document-bootstrap
95% CI of [-0.0163, 0.0033]. Thus the final RoPE rotation affects overall route
overlap but does not explain a stable function-specific gap.

By layer, raw-QK function/content Jaccard is 0.9613/0.9697 at layer 0,
0.5500/0.5735 at layer 13, and 0.4174/0.4663 at layer 27. The deep-layer gap
remains after removing the final position rotation.
