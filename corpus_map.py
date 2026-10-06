#!/usr/bin/env python3
"""Corpus map: every indicator vs every other, visually.

Fig 1 — pairwise centered-cosine heatmap over all scripts in the DB,
        greedily ordered by nearest-neighbour so same-family scripts
        clump into blocks along the diagonal.
Fig 2 — 2D PCA of all vectors; points labelled by script name, coloured
        by family cluster, and pairs above the REMIX/RELATED thresholds
        connected by lines.

Usage:  .venv/bin/python corpus_map.py                # all DBs found
        .venv/bin/python corpus_map.py --db <path>    # one specific DB
"""

import json
import sys
from pathlib import Path

import numpy as np

import embed_indicators as ei


def find_dbs():
    """label -> path, for every indicator_embeddings*.json in the project."""
    dbs = {}
    for p in sorted(Path(__file__).parent.glob('indicator_embeddings*.json')):
        tag = p.stem.replace('indicator_embeddings', '').lstrip('_')
        dbs[tag or 'qwen3-embedding_0.6b'] = p
    return dbs


MODELS = find_dbs()


def greedy_order(names, sim):
    """Order scripts as a nearest-neighbour chain so families form blocks.

    Seed with the strongest pair in the corpus, then extend the chain in
    both directions by the most-similar remaining script to either end.
    """
    remaining = set(range(len(names)))
    i, j = np.unravel_index(np.argmax(sim + np.eye(len(sim)) * -9), sim.shape)
    chain = [i, j]
    remaining -= {i, j}
    while remaining:
        # best (chain-end, candidate) over both ends
        cand_l = max(remaining, key=lambda c: sim[chain[0], c])
        cand_r = max(remaining, key=lambda c: sim[chain[-1], c])
        if sim[chain[0], cand_l] > sim[chain[-1], cand_r]:
            chain.insert(0, cand_l)
            remaining.discard(cand_l)
        else:
            chain.append(cand_r)
            remaining.discard(cand_r)
    return chain


def cluster_chain(names, sim, chain):
    """Split the greedy chain wherever consecutive sim < RELATED — each
    contiguous run is one family cluster."""
    clusters, cur = [], [chain[0]]
    for a, b in zip(chain, chain[1:]):
        if sim[a, b] >= ei.RELATED_THRESHOLD:
            cur.append(b)
        else:
            clusters.append(cur)
            cur = [b]
    clusters.append(cur)
    return clusters


def run(label='0.6b', show=True):
    """Build the corpus heatmap + PCA map for one model DB."""
    db = json.loads(MODELS[label].read_text())
    names = sorted(db['indicators'])
    raw = np.asarray([db['indicators'][n]['vector'] for n in names])
    if 'reference_mean' in db:
        mu = np.asarray(db['reference_mean'])
    else:
        # no frozen mean in this DB — recompute for visualization;
        # absolute scores shift slightly vs the calibrated thresholds
        mu = raw.mean(axis=0)
        print('note: DB has no reference_mean — recomputed corpus mean on the fly')
    V = np.asarray([ei.center(v, mu) for v in raw])
    sim = V @ V.T

    # display names: prefer the `// @title:` metadata captured at build time
    title = {n: db['indicators'][n].get('title') or Path(n).stem
             for n in names}

    order = greedy_order(names, sim)
    clusters = cluster_chain(names, sim, order)
    olabels = [title[names[i]] for i in order]
    S = sim[np.ix_(order, order)]

    print(f'{db["model"]}  —  {len(names)} scripts, '
          f'{len(clusters)} clusters')
    # strongest pairs
    pairs = sorted(((sim[i, j], names[i], names[j])
                    for i in range(len(names)) for j in range(i + 1, len(names))),
                   reverse=True)
    print('\nstrongest pairs:')
    for s, a, b in pairs[:12]:
        tag = ('REMIX' if s >= ei.REMIX_THRESHOLD
               else 'related' if s >= ei.RELATED_THRESHOLD else '')
        print(f'  {s:>7.4f}  {tag:<8} {title[a]}  <>  {title[b]}  ({a}, {b})')

    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ---- fig 1: ordered similarity heatmap --------------------------------
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(S, cmap='turbo', vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(olabels)), olabels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(olabels)), olabels, fontsize=7)
    for c in clusters[:-1]:
        b = len([i for cl in clusters[:clusters.index(c) + 1] for i in cl]) - 0.5
        ax.axhline(b, color='k', lw=0.6)
        ax.axvline(b, color='k', lw=0.6)
    fig.colorbar(im, ax=ax, label='centered cosine', shrink=0.8)
    ax.set_title(f'pairwise similarity — {db["model"]} '
                 f'(remix >= {ei.REMIX_THRESHOLD}, related >= {ei.RELATED_THRESHOLD})')
    fig.tight_layout()
    safe = label.replace(':', '_')
    out1 = Path(f'corpus_heatmap_{safe}.jpg')
    fig.savefig(out1, format='jpg', dpi=120, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f'\nsaved {out1}')

    # ---- fig 2: 2D PCA scatter --------------------------------------------
    P = V - V.mean(axis=0)
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    xy = P @ vt[:2].T
    cluster_of = {i: ci for ci, cl in enumerate(clusters) for i in cl}

    fig, ax = plt.subplots(figsize=(12, 9))
    # link strong pairs
    for s, a, b in pairs:
        if s < ei.RELATED_THRESHOLD:
            break
        i, j = names.index(a), names.index(b)
        remix = s >= ei.REMIX_THRESHOLD
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                color='crimson' if remix else 'gray',
                lw=1.6 if remix else 0.7, alpha=0.8 if remix else 0.4, zorder=1)
    sc = ax.scatter(xy[:, 0], xy[:, 1],
                    c=[cluster_of[i] for i in range(len(names))],
                    cmap='rainbow', s=70, edgecolor='k', linewidth=0.5, zorder=3)
    for i, n in enumerate(names):
        ax.annotate(title[n], (xy[i, 0], xy[i, 1]),
                    fontsize=7, textcoords='offset points', xytext=(5, 4))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'indicator corpus in semantic space — {db["model"]}  '
                 f'(red link = REMIX >= {ei.REMIX_THRESHOLD}, '
                 f'gray = related >= {ei.RELATED_THRESHOLD})')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out2 = Path(f'corpus_map_{safe}.jpg')
    fig.savefig(out2, format='jpg', dpi=120, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f'saved {out2}')
    return out1, out2


def main():
    args = sys.argv[1:]
    if '--db' in args:
        # run for a single DB path: label = its stem tag
        p = Path(args[args.index('--db') + 1])
        label = p.stem.replace('indicator_embeddings', '').lstrip('_')
        MODELS.clear()
        MODELS[label or 'qwen3-embedding_0.6b'] = p
        run(next(iter(MODELS)), show=False)
        return
    for label in MODELS:
        run(label, show=False)


if __name__ == '__main__':
    main()
