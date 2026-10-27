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
| Monotone | bigger coefficient, more intense output, no reversals | probe readout along the path; direct steering sweep (not built) |
| Graded | intermediate coefficient, intermediate intensity | probe readout; step-KL resolution ratio |
| Near the data | steered activations stay close to real ones, output stays fluent | nearest real word (distances not built) |
| General | the direction works across contexts and on unseen words | leave-one-word-out probe across frames |

## The probe

`src/smoothness_probe.py` runs two experiments on a small open model
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

The script has only been run end to end on a tiny randomly initialized stand-in
model. No real-model numbers exist yet. See `docs/ROADMAP.md` for what's next and
`docs/project_notes_2026-10-27.pdf` for the full project notes, prior work, and
the manifold-test design.

## Layout

```
embedding-geometry/
├── src/smoothness_probe.py   # experiments 1 and 2
├── docs/                     # roadmap and project notes
├── figures/                  # generated plots (ignored)
├── results/                  # saved per-model results (ignored)
└── requirements.txt
```
