# Roadmap

Order matters: debug the harness on cheap models before paying for GPU time.

## Before any real-model claims

- [ ] Run v2 on GPT-2 small as a smoke test. Check the random baseline's
      leave-one-word-out p-value and that fine-resolution max KL is well above 1e-9.
- [ ] Check the cosine between the one-pole intensity direction and a
      Tigges-style valence direction. If near 1, the honest framing is
      "steering along valence magnitude".
- [ ] Order the word list by Horn-scale entailment, not ratings. Drop one of any
      pair entailment doesn't order (amazing / fantastic). Add single-token words
      if the random baseline isn't clearly beaten: fine, nice, solid, superb,
      wonderful, terrific, awesome, incredible, brilliant, outstanding, flawless.

## Probe improvements

- [ ] Direct steering sweep: add lambda * v to neutral sentences, plot output
      intensity and fluency against lambda. This is the headline experiment; the
      interpolation is a bounded special case. A suffix readout ("out of ten,
      I'd give it a") is legitimate here because there is no source word to leak.
- [ ] Replace nearest-word with distance from a bank of real activations,
      relative to typical spacing.
- [ ] Fluency readout: next-token entropy and continuation perplexity vs lambda.
- [ ] Exact permutation p-value for the leave-one-word-out Spearman (n = 7 gives
      only 5040 orderings).
- [ ] Report the resolution ratio as an exponent, log(ratio) / log(4), and add a
      third resolution to confirm the scaling.
- [ ] Garí Soler-style difference-of-means baseline direction.
- [ ] Whitening / cosine switch for rogue dimensions; report geometry stats with
      massive-activation dimensions zeroed.

## Post 1: does it hold across models?

- [ ] Multi-model harness: model list, saved results per model, relative-depth
      plots, `hf_model=` loading.
- [ ] Pythia 160M / 410M / 1.4B (and 2.8B). Use step-0 checkpoints as the random
      baseline instead of re-initialized weights.
- [ ] Llama-3.2-1B (gated), Qwen2.5-1.5B or Gemma-2-2B.

## Post 2: does fine-tuning move it?

- [ ] Llama-3.1-8B base vs Instruct vs DeepSeek-R1-Distill-Llama-8B on a rented
      GPU. Verify the scalar words map to the same token IDs on all three first.
- [ ] Transfer test: cosine between base and fine-tune intensity directions;
      does the base steering vector still work on each fine-tune?

## Before publishing

- [ ] One hour of literature search: "scalar adjective" with "steering" or
      "activation patching", 2024-26.
- [ ] Ask someone with geometry or topology background to read the draft.
