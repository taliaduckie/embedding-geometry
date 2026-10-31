# Embedding geometry

Is a language model's residual-stream geometry good enough to steer along an
intensity scale (good < great < excellent), without assuming the space is a
smooth manifold?

Smoothness can't be confirmed from finite samples, and the network is smooth by
construction anyway, so "is the space smooth?" is the wrong question. Following
Levenstein et al., *The Problem-Ladenness of Theory* (2024), the idealization is
judged by whether it is accurate enough for the problem. Steering is the problem,
and it needs something weaker than a manifold: order preserved along a path.

## What counts as success

Steering along an intensity direction works if four things hold:

| Criterion | Success looks like | Covered by |
|---|---|---|
| Monotone | bigger coefficient, more intense output, no reversals | direct steering sweep: E[tier] of the next token vs lambda; probe readout along the interpolation path |
| Graded | intermediate coefficient, intermediate intensity | steering sweep: E[tier] through intermediate values; step-KL resolution ratio |
| Near the data | steered activations stay close to real ones, output stays fluent | steering sweep: scale-word mass, entropy, KL vs lambda; chord distance to nearest other word |
| General | the direction works across contexts and on unseen words | leave-one-tier-out probe across frames (leave-one-word-out is inflated by synonym tiers) |

## The probe

`src/smoothness_probe.py` runs three experiments on a small open model
(GPT-2 small by default) and works as a notebook (`# %%` cells) or top to bottom.

**Experiment 1, geometry.** Per layer: PC1 ordering with a permutation p-value,
tortuosity of the ordered path as a percentile against shuffled orderings,
anisotropy, and a leave-one-word-out ridge probe that must rank a word it never
trained on.

**Experiment 2, interpolation.** Norm-matched mix of two words' activations at a
layer, patched into the last position, run to the end. Readouts: a probe trained
without the endpoint words, step KL over the next-token distribution (full and
alphanumeric tokens only) at two resolutions, and the nearest real word at each
step. A max-KL ratio near 16 between 41 and 161 steps means smooth at that
resolution; near 1 means a transition narrower than the fine step.

**Experiment 3, steering.** Add lambda times a unit direction at the last
position of each frame (the adjective slot) and read the next-token
distribution: expected tier over the scale words, scale-word mass, entropy and
KL from the clean run, against lambda in units of the scale's span. Directions:
probe (ridge weights), difference of tier means, and random.

Guards: the scalar word is always the last token so patching can't leak the
source word through lower layers; an assert that alpha = 0 reproduces the clean
run; a `RANDOM_BASELINE` switch; unprocessed weights so geometry isn't skewed by
TransformerLens's centering.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd figures && python ../src/smoothness_probe.py
```

All settings are in the config block at the top of the script. Figures are
written to the working directory as `fig_*.png`.

## Status

Run on GPT-2 small (trained and random-weights baseline) on 2026-10-31, 19
words in six entailment tiers. Headline numbers, with the random baseline in
parentheses:

- Order: a linear direction ranks held-out tiers at Spearman 0.3 to 0.6, best in
  late layers, permutation p 0.01 to 0.07 (random: ~0). Held-out single words
  score 0.8, but that is mostly synonym matching within tiers.
- Arrangement: not supported. With the end tiers fixed, the true interior order
  is at the 71st to 87th tortuosity percentile, i.e. no straighter than a random
  interior ordering. PC1 ordering does beat its permutation null (p <= 0.01).
- Interpolation: smooth everywhere, including the good -> purple control; every
  resolution ratio is 15 to 16. The chord stays 0.6 to 1.0 endpoint-distances
  from any other word.
- Steering: monotone with no reversals along both learned directions at every
  layer. The difference-of-means direction moves the expected next-token tier
  by about 1.4 tiers across +/-1 span at layer 6; the probe direction moves it
  by 0.3. Scale-word mass holds to |lambda| = 1 and collapses by lambda = 3 at
  layers 3 to 6 (0.078 -> 0.01) with entropy rising; layers 9 to 12 degrade
  more gracefully. Random direction: flat.

Logs are in `results/`, figures in `figures/gpt2-small*`. See `docs/ROADMAP.md`
for the record of what changed and why, and `docs/project_notes_2026-10-27.pdf`
for the full project notes, prior work, and the manifold-test design.

## Layout

```
embedding-geometry/
├── src/smoothness_probe.py   # experiments 1, 2 and 3
├── docs/                     # roadmap and project notes
├── figures/                  # generated plots (ignored)
├── results/                  # saved per-model results (ignored)
└── requirements.txt
```
