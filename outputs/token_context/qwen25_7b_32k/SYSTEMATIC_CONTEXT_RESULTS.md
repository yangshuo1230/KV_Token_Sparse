# Sink-aware target-token context requirements

## Design

This experiment evaluates Qwen2.5-7B on 16 PG-19 documents of exactly 32,768
tokens. The final 64 target tokens in each document are scored, giving 1,024
targets and 12,288 target-policy observations.

Each compact policy has a fixed total KV budget of 128, 512, 2,048, or 8,192
positions. Within each budget, the policy retains either 0, 4, or 16 prefix
tokens and spends the remainder on the most recent tokens. Thus the sink-aware
policies do not receive extra KV capacity. All retained tokens keep their
original absolute position IDs.

Target labels include a broad category, fine lexical class, universal POS,
dependency relation, and first/continuation-subtoken status. Means and 95%
confidence intervals use documents—not tokens—as the bootstrap unit.

## Content versus function targets

Results below use a four-token sink. `ΔCE` is compact-context cross-entropy
minus full-context cross-entropy; larger values mean a greater need for omitted
context.

| Total KV budget | Content ΔCE | Function ΔCE | Content − function (95% CI) |
|---:|---:|---:|---:|
| 128 | 0.6013 | 0.1276 | +0.4737 `[+0.3070,+0.6469]` |
| 512 | 0.2924 | 0.0505 | +0.2419 `[+0.1011,+0.4185]` |
| 2,048 | 0.1164 | -0.0013 | +0.1177 `[+0.0323,+0.2193]` |
| 8,192 | 0.0056 | 0.0135 | -0.0079 `[-0.0427,+0.0276]` |

The association is not limited to correct-target probability. Content-minus-
function differences in full-to-compact KL are +0.2462, +0.1499, and +0.0881
at budgets 128, 512, and 2,048, respectively; their document-bootstrap
intervals exclude zero. Top-1 change-rate differences are +11.8, +7.4, and
+7.7 percentage points, also with intervals excluding zero.

The result is robust to sink allocation. At each of the first three budgets,
the content/function ΔCE contrast remains positive with 0, 4, and 16 sink
tokens. At 8,192 positions, it disappears under all three allocations.

## Fine-grained categories

At a 128-position budget with four sink tokens, the largest sufficiently
represented category means are:

| Target class | ΔCE | Documents | Targets |
|---|---:|---:|---:|
| Proper noun | 2.1758 | 13 | 41 |
| Adverb | 0.6809 | 11 | 31 |
| Common noun | 0.5620 | 16 | 168 |
| Verb | 0.4344 | 16 | 104 |
| Adjective | 0.3289 | 14 | 66 |
| Punctuation | 0.2468 | 16 | 105 |
| Auxiliary | 0.2270 | 14 | 52 |
| Determiner | 0.1995 | 16 | 65 |
| Pronoun | 0.1883 | 15 | 101 |
| Adposition | 0.0943 | 16 | 94 |

After subtracting each document's overall mean, proper nouns remain +1.8453
above the document baseline (95% CI `[+0.5077,+3.5749]`) and common nouns
remain +0.2281 above it (`[+0.0606,+0.3877]`). Adpositions, pronouns,
coordinating conjunctions, particles, and negations are below their within-
document baselines. Rare classes should not be ranked literally: for example,
there are only 14 negation and 19 question-word targets.

First subtokens are also more context-sensitive than continuation subtokens at
the 128-position budget: ΔCE difference +0.1721, 95% CI
`[+0.0320,+0.3066]`. The difference is not reliable at larger budgets.

## Attention-sink result

Replacing recent positions with four or sixteen prefix positions does not
improve average target prediction at a fixed total budget. Most sink-minus-
recent-only contrasts are close to zero. At budget 128, a four-token sink
slightly *increases* function-token ΔCE by +0.0264 (`[+0.0095,+0.0464]`).

This does not imply that attention sinks can be discarded in a streaming
deployment. The present experiment recomputes a compact sequence with original
position IDs; it does not prune an already-built `past_key_values` cache. The
engineering policy should retain a configurable sink by default and benchmark
real cached decoding separately.

## Conclusion and engineering implication

The expanded experiment supports a graded relationship between target type and
required context: nouns—especially proper nouns—and other content targets lose
substantially more predictive information under small KV budgets, while many
function and special-token classes tolerate aggressive context reduction. A
single static cache budget therefore spends unnecessary KV on many targets.

The next engineering step is to predict a target's context-sensitivity *before*
the target is generated, using cheap features available at the preceding query
position (for example top-logit token classes, entropy, and a small classifier),
then select among a few sink-plus-recent budgets. Oracle target labels are useful
for establishing the opportunity but are not themselves a deployable router.

Aggregate statistics are retained in `category_summary.csv`,
`category_contrasts.csv`, `category_pair_contrasts.csv`, and
`sink_contrasts.csv`. Per-target Parquet data remains regenerable and is not
versioned.
