# %% [markdown]
# # smoothness probe v2: does the model's geometry preserve a scale?
#
# two experiments on a small open model (GPT-2 small by default):
#   1. GEOMETRY. collect residual-stream activations for scalar words across
#      sentence frames, per layer. do they line up in order? is the ordered path
#      straighter than a random ordering of the same points? does a probe find a
#      scale direction that generalizes to words it never saw?
#   2. INTERPOLATION. mix the activations for word A and word B at layer L,
#      patch the mix in, run the rest of the model, and watch the readouts.
#      is the change gradual, or a jump narrower than our step size?
#
# design choices, and why:
#   - the scalar word is always the LAST token. with causal attention, nothing
#     downstream can see the unpatched lower layers at that position, so patching
#     at layer L doesn't leak word A's identity. (this is also why there's no
#     "out of ten, I'd give it a ___" readout: any suffix token reads the
#     unpatched lower layers at the word position and leaks word A.)
#   - LEAVE-ONE-WORD-OUT probing. a probe held out over frames can just memorize
#     token identity -> rank (a random-weights model scores ~0.9 that way). a
#     probe that ranks a word it never trained on has found a scale direction.
#   - SHUFFLE NULLS for PC1 ordering and tortuosity. with a few points in 768
#     dimensions, raw numbers are uninterpretable; percentiles against random
#     orderings of the same points are not.
#   - RESOLUTION SCALING for snaps. on a smooth path, per-step KL shrinks with
#     the square of the step size, so 4x more alphas -> ~16x smaller max KL.
#     a transition narrower than the fine step keeps its max KL (ratio ~1).
#     note: the network is smooth, so every snap resolves at SOME resolution;
#     the ratio tells you whether the transition is narrower than your steps.
#   - CONTENT-TOKEN KL. the full next-token distribution is dominated by
#     punctuation, so KL is also computed after renormalizing over tokens that
#     contain a letter or digit.
#   - ON-CHORD CHECK. a straight line between two activations can leave the
#     region real tokens occupy. at each alpha we record the nearest real word;
#     if good -> excellent passes through "great", the chord tracks the data.
#   - from_pretrained_no_processing: the default weight processing (centering
#     writing weights) subtracts the per-vector mean along d_model, which changes
#     anisotropy and norms. geometry claims should use unprocessed weights.
#   - norm-matched interpolation by default; a plain linear mix dips in norm
#     mid-path, and that dip alone can fake a snap. NORM_MATCH = False shows it.
#   - RANDOM_BASELINE switch: same architecture and tokenizer, untrained weights.
#
# runs as a notebook (VS Code / Jupyter read the # %% cells) or top to bottom.
# pip install "transformer_lens<4" scikit-learn matplotlib scipy
# (TransformerLens 4.0 removed HookedTransformer; this uses the 3.x API.)

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

# ---------------- config (everything you'd change lives here) ----------------
MODEL_NAME = "gpt2-small"   # also try "pythia-410m", "gpt2-medium"
RANDOM_BASELINE = False     # True = untrained weights, same tokenizer
WORD_SET = "intensifier"    # "intensifier" (one pole) or "valence" (two poles)
N_SHUFFLE = 1000            # random orderings for the null distributions
ALPHAS_COARSE = np.linspace(0, 1, 41)
ALPHAS_FINE = np.linspace(0, 1, 161)   # 4x resolution -> smooth predicts ~16x
NORM_MATCH = True
INTERP_FRAME_IDX = 0        # frame used for interpolation; probes never train on it
SEED = 0

# intensifiers: one pole, weakest to strongest. this ordering IS the claim being
# tested, so ground it in human ratings (de Melo & Bansal 2013; Garí Soler &
# Apidianaki 2020) rather than intuition. near-ties (excellent / amazing) add
# noise to every rank statistic; drop one if the ratings say they tie.
WORD_SETS = {
    "intensifier": ["okay", "decent", "good", "great", "excellent", "amazing", "perfect"],
    # valence spans two poles. if the model codes valence and extremeness
    # separately, this path folds into a V and inflates tortuosity. if you run
    # it, call the result a sentiment scale, not an intensifier scale.
    "valence": ["terrible", "awful", "bad", "poor", "mediocre", "okay",
                "decent", "good", "great", "excellent", "amazing", "fantastic", "perfect"],
}
# (A, B) pairs to interpolate. the last is a control across the semantic gap.
PAIRS = [("good", "excellent"), ("okay", "perfect"), ("good", "purple")]

FRAMES = [
    "The movie was",
    "Honestly, the food was",
    "I thought the concert was",
    "Overall, the hotel was",
    "The service tonight was",
    "My first impression was that the book was",
    "Everyone agreed the performance was",
    "The new album is",
    "To be fair, the coffee was",
    "Her presentation was",
    "In my opinion, the game was",
    "The ending of the show was",
]
# ------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=device)
if RANDOM_BASELINE:
    torch.manual_seed(SEED)
    model.init_weights()
    # TransformerLens init_weights only redraws parameters named W_*. biases
    # (b_Q, b_K, b_V, b_O, b_in, b_out, b_U) and LayerNorm scale/shift keep
    # their PRETRAINED values, so the "untrained" model isn't. reset them.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "W_" not in n:
                p.fill_(1.0) if n.endswith(".w") else p.zero_()
    n_bias = sum(1 for n, _ in model.named_parameters() if "W_" not in n)
    print(f"RANDOM BASELINE: weights re-initialized, {n_bias} bias/LN params reset")

N_LAYERS = model.cfg.n_layers
D = model.cfg.d_model
N_POINTS = N_LAYERS + 1  # index 0 = embeddings, index k = output of block k-1
SHOW_LAYERS = sorted({0, N_LAYERS // 4, N_LAYERS // 2, 3 * N_LAYERS // 4, N_LAYERS})
INTERP_FRAME = FRAMES[INTERP_FRAME_IDX]
rng = np.random.default_rng(SEED)


def is_single_token(word):
    return model.to_tokens(" " + word, prepend_bos=False).shape[1] == 1


WORDS = [w for w in WORD_SETS[WORD_SET] if is_single_token(w)]
print("kept:", WORDS)
print("dropped (multi-token, can't patch cleanly):",
      [w for w in WORD_SETS[WORD_SET] if w not in WORDS])
RANK = np.arange(len(WORDS))
for a_w, b_w in PAIRS:
    for w in (a_w, b_w):
        assert is_single_token(w), f"pair word {w!r} is multi-token; pick another"


def hook_name(idx):
    return "blocks.0.hook_resid_pre" if idx == 0 else f"blocks.{idx - 1}.hook_resid_post"


HOOKS = [hook_name(i) for i in range(N_POINTS)]


def build_tokens(frame, words):
    """frame tokens + one word token per row. building them by hand guarantees alignment."""
    base = model.to_tokens(frame)  # includes BOS
    wt = torch.tensor([model.to_single_token(" " + w) for w in words], device=device)
    return torch.cat([base.repeat(len(words), 1), wt[:, None]], dim=1)


def collect(frames, words):
    """returns acts[layer_idx, frame, word, d_model] at the word position."""
    acts = torch.zeros(N_POINTS, len(frames), len(words), D)
    for fi, frame in enumerate(frames):
        _, cache = model.run_with_cache(build_tokens(frame, words),
                                        names_filter=lambda n: n in HOOKS)
        for li, name in enumerate(HOOKS):
            acts[li, fi] = cache[name][:, -1].float().cpu()
    return acts


acts = collect(FRAMES, WORDS)
print("acts:", tuple(acts.shape))

# %% [markdown]
# ## experiment 1: geometry, with nulls
# per layer:
# - **ordering**: PC1 of frame-centered activations vs the hypothesized rank.
#   reported as |spearman| plus a permutation p-value (shuffled rank labels).
# - **tortuosity**: path length through the frame-averaged words in order,
#   divided by endpoint distance. reported with its percentile among random
#   orderings of the same points. low percentile = the true order is
#   unusually straight.
# - **anisotropy**: mean pairwise cosine of raw activations (the narrow cone).
# - **leave-one-word-out probe**: ridge trained on all-but-one word (all frames
#   except the interpolation frame), predicting the held-out word's rank.
#   signed spearman across held-out predictions. this is the real
#   "is there a scale direction" test.


def make_probe():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20)))


def tortuosity(m):
    return np.linalg.norm(np.diff(m, axis=0), axis=1).sum() / np.linalg.norm(m[-1] - m[0])


def layer_stats(acts):
    W = acts.shape[2]
    perms = [rng.permutation(W) for _ in range(N_SHUFFLE)]
    train_frames = [f for f in range(acts.shape[1]) if f != INTERP_FRAME_IDX]
    rows = []
    for li in range(N_POINTS):
        X = acts[li].numpy()                    # [F, W, D]
        F = X.shape[0]
        Xc = X - X.mean(axis=1, keepdims=True)  # remove per-frame mean
        pc1 = PCA(n_components=1).fit_transform(Xc.reshape(-1, D))[:, 0]

        rho = abs(spearmanr(pc1, np.tile(RANK, F))[0])
        rho_null = np.array([abs(spearmanr(pc1, np.tile(RANK[p], F))[0]) for p in perms])

        m = Xc.mean(axis=0)  # [W, D], frame-averaged
        tort = tortuosity(m)
        tort_null = np.array([tortuosity(m[p]) for p in perms])

        raw = X.reshape(-1, D)
        rn = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        cos = rn @ rn.T
        aniso = cos[np.triu_indices_from(cos, k=1)].mean()

        Xt = acts[li, train_frames].numpy()  # [F-1, W, D]

        def lowo_rho(labels):
            """train on W-1 words, predict the held-out word's label, for each word."""
            preds = []
            for w in range(W):
                keep = np.array([i for i in range(W) if i != w])
                p = make_probe().fit(Xt[:, keep].reshape(-1, D),
                                     np.tile(labels[keep], Xt.shape[0]))
                preds.append(p.predict(Xt[:, w]).mean())
            return spearmanr(preds, labels)[0]

        lowo = lowo_rho(RANK)
        # permutation null over the WHOLE procedure, not scipy's p-value. leave-
        # one-out has a negative bias: hold out the top word and the training
        # mean drops, so its prediction regresses below it. the null is centered
        # below zero (an untrained GPT-2 scores rho ~ -0.8 from this alone), so
        # the question is whether the observed rho beats shuffled-label runs.
        lowo_null = np.array([lowo_rho(RANK[p]) for p in perms])
        lowo_p = ((lowo_null >= lowo).sum() + 1) / (len(perms) + 1)

        rows.append(dict(rho=rho, rho_p=(rho_null >= rho).mean(),
                         tort=tort, tort_pct=(tort_null <= tort).mean(),
                         aniso=aniso, lowo=lowo, lowo_p=lowo_p,
                         lowo_null=lowo_null.mean()))
    return rows


stats = layer_stats(acts)
print(f"{'layer':>5} {'|rho|':>6} {'p':>6} {'tort':>6} {'pct':>5} {'cos':>5} "
      f"{'LOWO rho':>9} {'null':>6} {'p':>6}")
for li, s in enumerate(stats):
    print(f"{li:5d} {s['rho']:6.2f} {s['rho_p']:6.3f} {s['tort']:6.2f} "
          f"{s['tort_pct']:5.2f} {s['aniso']:5.2f} {s['lowo']:9.2f} "
          f"{s['lowo_null']:6.2f} {s['lowo_p']:6.3f}")
print("LOWO p is a permutation p-value (shuffled labels, whole procedure); "
      "null = mean shuffled rho, which sits below 0 by leave-one-out bias.")

fig, ax = plt.subplots(1, 4, figsize=(17, 3.2))
ax[0].plot([s["rho"] for s in stats], marker="o", label="|spearman|")
ax[0].plot([s["rho_p"] for s in stats], marker="x", ls="--", label="permutation p")
ax[0].set_title("ordering along PC1")
ax[0].legend(fontsize=7)
ax[1].plot([s["tort_pct"] for s in stats], marker="o")
ax[1].axhline(0.05, color="gray", ls=":")
ax[1].set_title("tortuosity percentile vs shuffles\n(low = unusually straight)")
ax[2].plot([s["aniso"] for s in stats], marker="o")
ax[2].set_title("anisotropy (mean cosine)")
ax[3].plot([s["lowo"] for s in stats], marker="o", label="observed")
ax[3].plot([s["lowo_null"] for s in stats], marker="x", ls="--", label="shuffled-label mean")
ax[3].axhline(0, color="gray", ls=":")
ax[3].set_title("leave-one-word-out probe (spearman)")
ax[3].legend(fontsize=7)
for a in ax:
    a.set_xlabel("layer (0 = embeddings)")
plt.tight_layout()
plt.savefig("fig_layer_stats.png", dpi=160)
plt.show()

# %%
# PCA pictures at a few depths. this is the figure for the post.
fig, axes = plt.subplots(1, len(SHOW_LAYERS), figsize=(3.4 * len(SHOW_LAYERS), 3.4))
for a, li in zip(axes, SHOW_LAYERS):
    X = acts[li].numpy()
    m = (X - X.mean(axis=1, keepdims=True)).mean(axis=0)
    p = PCA(n_components=2).fit_transform(m)
    a.plot(p[:, 0], p[:, 1], "-", color="lightgray", zorder=0)
    a.scatter(p[:, 0], p[:, 1], c=RANK, cmap="viridis", zorder=1)
    for (x, y), w in zip(p, WORDS):
        a.annotate(w, (x, y), fontsize=7)
    a.set_title(f"layer {li}")
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.savefig("fig_pca_layers.png", dpi=160)
plt.show()

# %% [markdown]
# ## experiment 2: interpolation
# readouts at each alpha:
# - **probe**: final-layer ridge probe trained WITHOUT the pair's endpoint
#   words and without the interpolation frame, so it can't just recognize A or B.
#   when the patch layer is the final layer, this readout is linear in alpha by
#   construction: keep it as a sanity line, don't count it as evidence.
# - **step KL**, full distribution and content tokens only, at two resolutions.
# - **nearest real word** to the mixed vector (the on-chord check).


def content_token_mask():
    tok = getattr(model, "tokenizer", None)
    V = model.cfg.d_vocab
    if tok is None:
        return torch.ones(V, dtype=torch.bool)
    mask = torch.ones(V, dtype=torch.bool)
    for i in range(min(V, len(tok))):
        mask[i] = any(c.isalnum() for c in tok.decode([i]))
    return mask


CONTENT = content_token_mask()
print(f"content tokens: {int(CONTENT.sum())} / {len(CONTENT)}")


def endpoint_acts(frame, word):
    """all-layer activations at the word position for one (frame, word)."""
    _, cache = model.run_with_cache(build_tokens(frame, [word]),
                                    names_filter=lambda n: n in HOOKS)
    return torch.stack([cache[n][0, -1].float().cpu() for n in HOOKS])  # [N_POINTS, D]


def clean_logprobs(frame, word):
    # .cpu() before .double(): MPS has no float64
    return model(build_tokens(frame, [word]))[0, -1].cpu().double().log_softmax(-1)


def mix(a, b, alphas, norm_match=NORM_MATCH):
    out = []
    for t in alphas:
        v = (1 - t) * a + t * b
        if norm_match:
            v = v / v.norm() * ((1 - t) * a.norm() + t * b.norm())
        out.append(v)
    return torch.stack(out)


def run_patched(frame, li, vecs):
    """overwrite the last position at hook li with vecs (one per row), run to the end."""
    # the word token here is irrelevant: it gets overwritten, and nothing
    # downstream sees the lower layers at the last position.
    toks = build_tokens(frame, [WORDS[0]]).repeat(len(vecs), 1)
    vecs = vecs.to(device=device, dtype=model.cfg.dtype)
    store = {}

    def patch(resid, hook):
        resid[:, -1] = vecs
        return resid

    def grab(resid, hook):
        store["final"] = resid[:, -1].float().cpu()

    logits = model.run_with_hooks(
        toks, fwd_hooks=[(hook_name(li), patch), (hook_name(N_LAYERS), grab)])
    # float64 log-softmax keeps tiny fine-resolution KLs above the noise floor
    return store["final"], logits[:, -1].cpu().double().log_softmax(-1)


def step_kl(logp):
    """KL(step t || step t+1) for consecutive rows of a log-prob matrix."""
    p = logp.exp()
    return (p[:-1] * (logp[:-1] - logp[1:])).sum(-1).numpy()


def renorm(logp, mask):
    sub = logp[:, mask]
    return sub - torch.logsumexp(sub, dim=-1, keepdim=True)


def runs(seq):
    """run-length encode: ['good','good','great'] -> 'good×2 → great×1'."""
    out, prev, n = [], None, 0
    for s in seq + [None]:
        if s == prev:
            n += 1
        else:
            if prev is not None:
                out.append(f"{prev}×{n}")
            prev, n = s, 1
    return " → ".join(out)


def readout_probe(exclude):
    """final-layer probe trained without the given words and the interp frame."""
    keep = np.array([i for i, w in enumerate(WORDS) if w not in exclude])
    frames = [f for f in range(len(FRAMES)) if f != INTERP_FRAME_IDX]
    X = acts[N_LAYERS][frames][:, keep].reshape(-1, D).numpy()
    return make_probe().fit(X, np.tile(RANK[keep], len(frames)))


def sweep(word_a, word_b, alphas, layers=SHOW_LAYERS, frame=INTERP_FRAME,
          norm_match=NORM_MATCH):
    A, B = endpoint_acts(frame, word_a), endpoint_acts(frame, word_b)
    clean = clean_logprobs(frame, word_a)
    probe = readout_probe({word_a, word_b})
    ref = acts[:, INTERP_FRAME_IDX]  # [N_POINTS, W, D] real word activations
    # the endpoints are in the reference set, so "nearest word" would trivially
    # read A then B. exclude them: which OTHER word is nearest at each alpha, and
    # how far is the chord from it, relative to the A-B distance?
    others = [i for i, w in enumerate(WORDS) if w not in (word_a, word_b)]
    res = {}
    for li in layers:
        vecs = mix(A[li], B[li], alphas, norm_match)
        final, logp = run_patched(frame, li, vecs)
        # free bug check: alpha=0 must reproduce the clean run for word A
        err = (logp[0] - clean).abs().max().item()
        assert err < 1e-3, f"alpha=0 doesn't match clean run at layer {li} (max diff {err:.2e})"
        d = torch.cdist(vecs, ref[li][others])  # [n_alpha, n_others]
        res[li] = dict(
            readout=probe.predict(final.numpy()),
            kl_full=step_kl(logp),
            kl_content=step_kl(renorm(logp, CONTENT)),
            nearest=[WORDS[others[i]] for i in d.argmin(-1)],
            rel_dist=(d.min(-1).values / (A[li] - B[li]).norm()).numpy(),
            trivial=(li == N_LAYERS),
        )
    return res


def summarize(word_a, word_b, coarse, fine):
    print(f"\n{word_a} → {word_b}")
    print(f"{'layer':>5} {'maxKL coarse':>13} {'maxKL fine':>11} {'ratio':>6} "
          f"{'dist/|A-B|':>11}   nearest OTHER word along the chord")
    for li in fine:
        c, f = coarse[li]["kl_content"].max(), fine[li]["kl_content"].max()
        r = fine[li]["rel_dist"]
        tag = "  (final layer: readout trivially linear)" if fine[li]["trivial"] else ""
        print(f"{li:5d} {c:13.2e} {f:11.2e} {c / f:6.1f} {r.min():5.2f}-{r.max():4.2f}   "
              f"{runs(fine[li]['nearest'])}{tag}")
    print("dist/|A-B|: distance from the chord to the nearest non-endpoint word, "
          "as a fraction of the endpoint distance (min-max over alpha).")
    print("ratio ≈16: smooth at this resolution. ≈1: transition narrower than the fine step.")


def plot_sweep(res, alphas, word_a, word_b, tag):
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.4))
    for li, r in res.items():
        lab = f"layer {li}" + (" (trivial)" if r["trivial"] else "")
        ls = ":" if r["trivial"] else "-"
        ax[0].plot(alphas, r["readout"], ls=ls, label=lab)
        ax[1].plot(alphas[1:], r["kl_full"], label=lab)
        ax[2].plot(alphas[1:], r["kl_content"], label=lab)
    ax[0].set_title(f"probe readout (endpoints held out): {word_a} → {word_b}")
    ax[0].set_ylabel("predicted rank")
    ax[1].set_title("step KL, full distribution")
    ax[2].set_title("step KL, content tokens only")
    for a in ax:
        a.set_xlabel("mix α")
        a.legend(fontsize=7)
    ax[1].set_yscale("log")
    ax[2].set_yscale("log")
    plt.tight_layout()
    plt.savefig(f"fig_interp_{tag}.png", dpi=160)
    plt.show()


# %%
results = {}
for a_w, b_w in PAIRS:
    coarse = sweep(a_w, b_w, ALPHAS_COARSE)
    fine = sweep(a_w, b_w, ALPHAS_FINE)
    results[(a_w, b_w)] = (coarse, fine)
    summarize(a_w, b_w, coarse, fine)
    plot_sweep(fine, ALPHAS_FINE, a_w, b_w, f"{a_w}_{b_w}")

# %%
# the norm-dip artifact: same pair, plain linear mixing.
a_w, b_w = PAIRS[0]
raw = sweep(a_w, b_w, ALPHAS_FINE, norm_match=False)
plot_sweep(raw, ALPHAS_FINE, a_w, b_w, f"{a_w}_{b_w}_nonormmatch")
