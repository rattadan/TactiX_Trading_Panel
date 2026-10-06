#!/usr/bin/env python3
"""Word-mutation Fourier map.

Replaces tokens of source A one by one (cumulative, random positions) with
tokens drawn from source B — the text drifts from A toward B. Each mutant is
embedded, and the per-dimension contribution signal  v_A[i] * v_mutant[i]
is FFT'd.

Default sources: the birria recipe (A) drifting toward churros (B).
With --pine: two .pine files from indicator-scripts/ — comments are stripped
and embedding goes through the same pipeline as `embed_indicators.py build`.

Plot 1 (<tag>.jpg):  top — cosine vs A and vs B across the sweep
       bottom — x: % of tokens replaced, y: embedding dims ranked by their
                contribution to the A×A similarity, colour: contribution
                value at that step. You watch the dominant dims erode as
                tokens get swapped.

Plot 2 (trajectory_<tag>.jpg): 2D PCA projection of every mutant vector —
the path the text walks in semantic space from A to B, rainbow-coloured by
% replaced.

Usage:  .venv/bin/python word_sweep_fourier.py [n_steps]
        .venv/bin/python word_sweep_fourier.py --pine A.pine B.pine [n_steps]
"""

import sys
from pathlib import Path

import numpy as np

import embed_indicators as ei
import compare_texts_app as app  # BIRRIA / CHURROS + embed helper

MODEL = 'qwen3-embedding:0.6b'
STEPS = 21      # 0%, 5%, ..., 100% of tokens replaced
SEED = 7

DEFAULT_PINE_A = '08_rsi_divergence.pine'  # RSI-based
DEFAULT_PINE_B = 'macd_ema.pine'           # EMA-based


def embed_plain(text):
    return app.embed_text(text, MODEL)[0]


def embed_pine(text):
    ei.MODEL = MODEL
    return ei.embed_indicator(text)[0]


def run_sweep(text_a, text_b, embed, cache, n_steps, label_a, label_b):
    if cache.exists():
        z = np.load(cache)
        if z['S'].shape[1] == n_steps and 'V' in z.files:
            print(f'loaded cached sweep from {cache}')
            return z['S'], z['sim_a'].tolist(), z['sim_b'].tolist(), z['V']

    words = text_a.split()
    vocab = [w for w in text_b.split() if w.strip('.,;:')]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(words))          # which positions get replaced, in order
    repl = rng.choice(vocab, size=len(words))    # replacement token per position

    v0 = embed(text_a)
    vc = embed(text_b)
    print(f'{len(words)} tokens, drifting {label_a} -> {label_b} '
          f'over {n_steps} steps\n')
    print(f'{"%words":>6}  {"cos vs A":>9} {"cos vs B":>9}')
    print('-' * 30)

    spectra, sim_a, sim_b, vecs = [], [], [], []
    for step in range(n_steps):
        f = step / (n_steps - 1)
        k = int(round(f * len(words)))
        w = words.copy()
        for i in order[:k]:
            w[i] = repl[i]
        v = embed(' '.join(w))
        vecs.append(v)
        spectra.append(np.abs(np.fft.rfft(v0 * v)))
        sim_a.append(ei.cosine(v0, v))
        sim_b.append(ei.cosine(vc, v))
        print(f'{f * 100:>5.0f}%  {sim_a[-1]:>9.4f} {sim_b[-1]:>9.4f}', flush=True)

    S = np.asarray(spectra).T  # (freq bins) x (steps)
    V = np.asarray(vecs)       # (steps) x (dim)
    np.savez(cache, S=S, sim_a=sim_a, sim_b=sim_b, V=V)
    return S, sim_a, sim_b, V


def generate(text_a, text_b, label_a, label_b, embed, tag, n_steps=STEPS,
             show=True):
    """Run the token-mutation sweep for one A/B pair; returns output paths."""
    cache = Path(f'word_sweep_cache_{tag}.npz')
    S, sim_a, sim_b, V = run_sweep(text_a, text_b, embed, cache, n_steps,
                                   label_a, label_b)

    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = [s / (n_steps - 1) * 100 for s in range(n_steps)]

    # ---- fig 1: similarity curve + per-dim contribution heatmap -----------
    # contrib[k, i] = v0[i] * V[k, i] — which dims carry the remaining A-ness
    contrib = V * embed(text_a)  # (steps, dim); embed A once for the product
    order = np.argsort(-np.abs(contrib[0]))  # rank by contribution at step 0
    top_n = min(128, contrib.shape[1])
    C = contrib[:, order[:top_n]].T          # (top dims) x (steps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 2]})
    ax1.plot(x, sim_a, 'o-', lw=2, color='tab:red', label=f'cos vs {label_a} (original)')
    ax1.plot(x, sim_b, 'o-', lw=2, color='tab:blue', label=f'cos vs {label_b} (donor)')
    ax1.set_ylabel('cosine similarity')
    ax1.legend()
    ax1.grid(alpha=0.3)

    lim = np.abs(C).max()
    im = ax2.imshow(C, aspect='auto', origin='lower', cmap='turbo',
                    vmin=-lim, vmax=lim,
                    extent=[x[0], x[-1], 0, top_n - 1])
    ax2.set_xlabel(f'% of {label_a} tokens replaced by {label_b} tokens')
    ax2.set_ylabel(f'top {top_n} dims by contribution at 0% (rank)')
    ax2.set_title(f'contribution erosion map — {MODEL}  '
                  f'(hot = dim still carrying A-similarity)')
    fig.colorbar(im, ax=ax2, label='a[i]*mutant[i] contribution')
    fig.tight_layout()
    out = Path(f'word_sweep_fourier_{tag}.jpg')
    fig.savefig(out, format='jpg', dpi=110, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f'saved {out}')

    # ---- fig 2: 2D PCA trajectory of the mutants ---------------------------
    # project A, B, and all mutants onto the top-2 principal components of
    # the mutant cloud — shows the path walked through semantic space
    pts = np.vstack([V, embed(text_a)[None, :], embed(text_b)[None, :]])
    centered_pts = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered_pts, full_matrices=False)
    xy = centered_pts @ vt[:2].T
    mut_xy, a_xy, b_xy = xy[:-2], xy[-2], xy[-1]

    fig2, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(mut_xy[:, 0], mut_xy[:, 1], c=x, cmap='rainbow',
                    s=80, edgecolor='k', linewidth=0.5, zorder=3)
    ax.plot(mut_xy[:, 0], mut_xy[:, 1], '-', color='gray', alpha=0.4, zorder=2)
    ax.scatter(*a_xy, marker='*', s=400, color='limegreen', edgecolor='k',
               zorder=4, label=f'{label_a} (original)')
    ax.scatter(*b_xy, marker='X', s=200, color='red', edgecolor='k',
               zorder=4, label=f'{label_b} (donor)')
    ax.annotate('0%', mut_xy[0], textcoords='offset points', xytext=(8, 6))
    ax.annotate('100%', mut_xy[-1], textcoords='offset points', xytext=(8, 6))
    fig2.colorbar(sc, ax=ax, label='% tokens replaced')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'mutation trajectory in semantic space — {label_a} -> {label_b}')
    ax.legend()
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = Path(f'word_sweep_trajectory_{tag}.jpg')
    fig2.savefig(out2, format='jpg', dpi=110, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig2)
    print(f'saved {out2}')
    return out, out2


def sweep_pine(a_name=DEFAULT_PINE_A, b_name=DEFAULT_PINE_B, n_steps=STEPS,
               show=True):
    """Token-mutation sweep between two .pine files (comments stripped)."""
    text_a = ei.strip_comments(
        (ei.INDICATOR_DIR / a_name).read_text(encoding='utf-8', errors='replace'))
    text_b = ei.strip_comments(
        (ei.INDICATOR_DIR / b_name).read_text(encoding='utf-8', errors='replace'))
    la, lb = Path(a_name).stem, Path(b_name).stem
    return generate(text_a, text_b, la, lb, embed_pine, f'{la}_vs_{lb}',
                    n_steps, show)


def sweep_recipes(n_steps=STEPS, show=True):
    """The birria -> churros demo pair."""
    return generate(app.BIRRIA, app.CHURROS, 'birria', 'churros', embed_plain,
                    'birria_vs_churros', n_steps, show)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--pine' in sys.argv:
        a = args[0] if len(args) > 0 else DEFAULT_PINE_A
        b = args[1] if len(args) > 1 else DEFAULT_PINE_B
        sweep_pine(a, b, int(args[2]) if len(args) > 2 else STEPS, show=False)
    else:
        sweep_recipes(int(args[0]) if args else STEPS, show=False)


if __name__ == '__main__':
    main()
