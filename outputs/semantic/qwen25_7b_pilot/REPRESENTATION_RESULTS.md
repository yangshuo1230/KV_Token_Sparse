# Hidden-state and repeated-word diagnostics

The diagnostic uses 12 documents and layers 0, 13, and 27. Function-content
pairs and content-content controls are matched by distance within the same
semantic segment. Repeated content lemmas are compared with different content
words near the second occurrence. Retrieval pairs use exactly the same remote
KV candidate set.

## Function versus content

| Layer | Function-content h cosine | Content-content h cosine | Function token top-10 Jaccard | Content token top-10 Jaccard |
|---:|---:|---:|---:|---:|
| 0 | 0.0870 | 0.1110 | 0.2706 | 0.2314 |
| 13 | 0.5490 | 0.5361 | 0.2477 | 0.2338 |
| 27 | 0.6865 | 0.6911 | 0.1593 | 0.1757 |

After distance matching, no stable function-specific hidden-state similarity is
observed. Same-segment token representations become generally similar in middle
and deep layers. Exact token top-10 overlap is modest; the higher overlap in the
main experiment is primarily a block-aggregation effect.

## Repeated content lemmas

| Layer | Repeated h cosine | Different-word control | Repeated top-10 Jaccard | Control top-10 Jaccard |
|---:|---:|---:|---:|---:|
| 0 | 0.8883 | 0.1159 | 0.0854 | 0.0899 |
| 13 | 0.7384 | 0.4022 | 0.2240 | 0.1705 |
| 27 | 0.7365 | 0.6112 | 0.2042 | 0.1629 |

Repeated lemmas retain much more similar hidden states. Their retrieval overlap
is higher only in middle/deep layers: +0.0535 at layer 13 and +0.0413 at layer
27, with document-bootstrap intervals excluding zero. Absolute token overlap
remains near 20%, so repeated words do not generally have identical routes.
