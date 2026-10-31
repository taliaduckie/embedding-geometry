# Roadmap

Order matters: debug the harness on cheap models before paying for GPU time.

## Before any real-model claims

- [x] Run v2 on GPT-2 small as a smoke test. Check the random baseline's
      leave-one-word-out p-value and that fine-resolution max KL is well above 1e-9.
      Done 2026-10-29. Fine max KL 2e-5 to 4e-4 (trained), 2e-6 to 7e-6 (random).
      Trained LOWO permutation p 0.04 to 0.05 at layers 3 to 7; random p 0.2 to 0.9.
      Every resolution ratio is 15.4 to 16.1, including the good -> purple control:
      no snaps at this step size. Logs in results/, figures in figures/gpt2-small*.
- [ ] Check the cosine between the one-pole intensity direction and a
      Tigges-style valence direction. If near 1, the honest framing is
      "steering along valence magnitude".
- [x] Order the word list by Horn-scale entailment, not ratings. Done 2026-10-29
      as six entailment TIERS with tied ranks (okay/fine < decent/solid < good/nice
      < great/... < excellent/... < perfect/flawless), 19 single-token words.
      Words entailment can't order tie instead of being dropped. Result: trained
      LOWO rho 0.8, permutation p 0.003 at every layer; random baseline rho ~0,
      p 0.2 to 0.6. Layer 0 (raw embeddings) already scores rho 0.82. Tier
      membership is a judgment call; revisit against a ratings source.
      Follow-up 2026-10-29: leave-one-TIER-out added. It is the honest
      generalization test with synonym tiers, and it is much weaker: 6-tier
      layout rho 0.24 to 0.58 (p 0.01 to 0.07, best late), 7-tier layout
      (amazing / incredible / fantastic split above excellent) rho 0.12 to 0.44,
      significant only at layers 10 to 12. Random baseline near 0 for both.
      So LOWO's 0.8 was mostly synonym matching. GPT-2 places amazing /
      incredible / fantastic below excellent / superb / brilliant, against
      the entailment split. End tiers compress to the training mean (ridge
      can't extrapolate): "perfect" held out is predicted mid-scale, so LOTO
      says nothing about pushing past the endpoints. Default is the 6-tier
      layout (tie when entailment intuition is uncertain); the split is
      WORD_SET = "intensifier_7tier".
- [x] Fixed-endpoint tortuosity null. Done 2026-10-31. The all-label shuffle
      put the true order in the bottom 1 to 13 percent at every layer; with the
      end tiers held fixed and only interior labels shuffled, the true order
      sits at the 71st to 87th percentile. The "straighter than random" result
      was entirely the endpoints being the farthest pair. The interior path
      (decent .. excellent) is, if anything, wigglier than a random interior
      ordering. Random baseline: 11 to 52 percent, chance. So the Arrangement
      criterion is NOT supported in GPT-2 small; PC1 ordering and the held-out
      probes carry the order claim, tortuosity does not.

## Probe improvements

- [ ] Direct steering sweep: add lambda * v to neutral sentences, plot output
      intensity and fluency against lambda. This is the headline experiment; the
      interpolation is a bounded special case. A suffix readout ("out of ten,
      I'd give it a") is legitimate here because there is no source word to leak.
- [ ] Replace nearest-word with distance from a bank of real activations,
      relative to typical spacing. Partly done: endpoints are now excluded from
      the reference set and the chord's distance to the nearest other word is
      reported as a fraction of |A-B|. Still relative to the endpoint distance,
      not to typical spacing, and GPT-2 small gives 0.6 to 1.0 everywhere.
- [ ] Fluency readout: next-token entropy and continuation perplexity vs lambda.
- [x] Permutation p-value for the leave-one-word-out Spearman. Done as a
      sampled null (N_SHUFFLE draws) over the whole procedure, which also
      handles the leave-one-out negative bias: an untrained model scores
      rho near -0.8 from that bias alone, so scipy's zero-centred p was wrong.
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
