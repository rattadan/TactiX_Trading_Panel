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


def classical_mds(sim, dims=2):
    """Layout in `dims` dims that approximates true pairwise cosine distance.

    Double-centers the squared-distance matrix and takes the top-k
    eigendecomposition — points end up as far apart in the map as the
    embeddings say they are, unlike PCA which only maximizes variance.
    """
    n = len(sim)
    d2 = np.maximum(2.0 - 2.0 * sim, 0.0)   # ||u-v||^2 = 2-2cos for unit vecs
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ d2 @ J
    evals, evecs = np.linalg.eigh(B)
    top = np.argsort(evals)[::-1][:dims]
    return evecs[:, top] * np.sqrt(np.maximum(evals[top], 0))


def _scatter3d_html(xy, names, title, clusters, pairs, model, mode, center,
                    method, out):
    """Interactive 3D scatter → standalone HTML (rotate/zoom/hover)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print('plotly not installed — skipping 3D html '
              '(.venv/bin/pip install plotly)')
        return None
    cluster_of = {i: ci for ci, cl in enumerate(clusters) for i in cl}
    cmap = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
            '#42d4f4', '#f032e6', '#bfef45', '#469990', '#9a6324',
            '#800000', '#aaffc3', '#808000', '#000075', '#a9a9a9']

    traces = []
    # pairwise links (centered mode only — thresholds calibrated there)
    if center:
        for s, a, b in pairs:
            if s < ei.RELATED_THRESHOLD:
                break
            i, j = names.index(a), names.index(b)
            remix = s >= ei.REMIX_THRESHOLD
            traces.append(go.Scatter3d(
                x=[xy[i, 0], xy[j, 0], None],
                y=[xy[i, 1], xy[j, 1], None],
                z=[xy[i, 2], xy[j, 2], None],
                mode='lines',
                line=dict(color='crimson' if remix else '#999999',
                          width=5 if remix else 2),
                opacity=0.9 if remix else 0.4,
                hoverinfo='skip', showlegend=False))

    for ci, cl in enumerate(clusters):
        idx = cl
        traces.append(go.Scatter3d(
            x=xy[idx, 0], y=xy[idx, 1], z=xy[idx, 2],
            mode='markers+text',
            marker=dict(size=7, color=cmap[ci % len(cmap)],
                        line=dict(width=1, color='black')),
            text=[title[names[i]] for i in idx],
            textposition='top center', textfont=dict(size=9),
            hovertemplate='%{text}<extra></extra>',
            name=f'cluster {ci + 1}'))
    fig = go.Figure(traces)
    fig.update_layout(
        title=f'indicator corpus — {method} 3D map ({mode}) — {model}',
        scene=dict(xaxis_title='dim 1', yaxis_title='dim 2',
                   zaxis_title='dim 3'),
        legend=dict(font=dict(size=10)), width=1100, height=850)
    fig.write_html(str(out), include_plotlyjs=True)  # inline — works offline
    print(f'saved {out}')
    return out


def _scatter2d(xy, names, title, clusters, pairs, model, mode, center,
               method, out, show):
    """Shared renderer for the PCA and MDS 2D corpus maps."""
    import matplotlib.pyplot as plt
    cluster_of = {i: ci for ci, cl in enumerate(clusters) for i in cl}

    fig, ax = plt.subplots(figsize=(12, 9))
    # link strong pairs — thresholds are calibrated for the centered scale
    if center:
        for s, a, b in pairs:
            if s < ei.RELATED_THRESHOLD:
                break
            i, j = names.index(a), names.index(b)
            remix = s >= ei.REMIX_THRESHOLD
            ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                    color='crimson' if remix else 'gray',
                    lw=1.6 if remix else 0.7, alpha=0.8 if remix else 0.4, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1],
               c=[cluster_of[i] for i in range(len(names))],
               cmap='rainbow', s=70, edgecolor='k', linewidth=0.5, zorder=3)
    for i, n in enumerate(names):
        ax.annotate(title[n], (xy[i, 0], xy[i, 1]),
                    fontsize=7, textcoords='offset points', xytext=(5, 4))
    ax.set_xlabel(f'{method} dim 1')
    ax.set_ylabel(f'{method} dim 2')
    ax.set_title(f'indicator corpus — {method} map ({mode}) — {model}'
                 + (f'  (red link = REMIX >= {ei.REMIX_THRESHOLD}, '
                    f'gray = related >= {ei.RELATED_THRESHOLD})' if center else ''))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, format='jpg', dpi=120, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f'saved {out}')


def run(label='0.6b', center=True, show=True):
    """Build the corpus heatmap + PCA map for one model DB.

    center=True  — subtract the frozen reference_mean (calibrated scale)
    center=False — raw cosine on the stored unit vectors
    """
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
    if center:
        V = np.asarray([ei.center(v, mu) for v in raw])
    else:
        V = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    sim = V @ V.T
    mode = 'centered' if center else 'raw'

    # display names: prefer the `// @title:` metadata captured at build time
    title = {n: db['indicators'][n].get('title') or Path(n).stem
             for n in names}

    order = greedy_order(names, sim)
    clusters = cluster_chain(names, sim, order)
    olabels = [title[names[i]] for i in order]
    S = sim[np.ix_(order, order)]

    print(f'{db["model"]} ({mode})  —  {len(names)} scripts, '
          f'{len(clusters)} clusters')
    # strongest pairs
    pairs = sorted(((sim[i, j], names[i], names[j])
                    for i in range(len(names)) for j in range(i + 1, len(names))),
                   reverse=True)
    print('\nstrongest pairs:')
    for s, a, b in pairs[:12]:
        tag = ('REMIX' if s >= ei.REMIX_THRESHOLD
               else 'related' if s >= ei.RELATED_THRESHOLD else '')
        if not center:
            tag = ''  # thresholds are calibrated on the centered scale
        print(f'  {s:>7.4f}  {tag:<8} {title[a]}  <>  {title[b]}  ({a}, {b})')

    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ---- fig 1: ordered similarity heatmap --------------------------------
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(S, cmap='turbo',
                   vmin=-0.2 if center else 0.4, vmax=1.0)
    ax.set_xticks(range(len(olabels)), olabels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(olabels)), olabels, fontsize=7)
    for c in clusters[:-1]:
        b = len([i for cl in clusters[:clusters.index(c) + 1] for i in cl]) - 0.5
        ax.axhline(b, color='k', lw=0.6)
        ax.axvline(b, color='k', lw=0.6)
    fig.colorbar(im, ax=ax, label=f'{mode} cosine', shrink=0.8)
    ax.set_title(f'pairwise similarity ({mode}) — {db["model"]}'
                 + (f' (remix >= {ei.REMIX_THRESHOLD}, related >= {ei.RELATED_THRESHOLD})'
                    if center else ''))
    fig.tight_layout()
    safe = label.replace(':', '_')
    out1 = Path(f'corpus_heatmap_{safe}_{mode}.jpg')
    fig.savefig(out1, format='jpg', dpi=120, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f'\nsaved {out1}')

    # ---- figs 2+3: 2D maps -------------------------------------------------
    # PCA: maximizes variance — the big-picture shape of the corpus
    P = V - V.mean(axis=0)
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    xy_pca = P @ vt[:2].T
    out2 = Path(f'corpus_map_{safe}_{mode}.jpg')
    _scatter2d(xy_pca, names, title, clusters, pairs, db['model'], mode,
               center, 'PCA', out2, show)

    # classical MDS: preserves pairwise cosine distances — read it as
    # "how far apart the scripts really are", not just variance spread
    xy_mds = classical_mds(sim)
    out3 = Path(f'corpus_mds_{safe}_{mode}.jpg')
    _scatter2d(xy_mds, names, title, clusters, pairs, db['model'], mode,
               center, 'MDS', out3, show)

    # interactive 3D MDS — same distance-preserving layout, one more dim
    xy3 = classical_mds(sim, dims=3)
    out4 = Path(f'corpus_mds3d_{safe}_{mode}.html')
    _scatter3d_html(xy3, names, title, clusters, pairs, db['model'], mode,
                    center, 'MDS', out4)
    return out1, out2, out3, out4


def main():
    args = sys.argv[1:]
    raw_only = '--raw' in args and '--centered' not in args
    ctr_only = '--centered' in args and '--raw' not in args
    modes = [False] if raw_only else ([True] if ctr_only else [True, False])
    # default (no flags) and --both: both modes
    if '--db' in args:
        # run for a single DB path: label = its stem tag
        p = Path(args[args.index('--db') + 1])
        label = p.stem.replace('indicator_embeddings', '').lstrip('_')
        MODELS.clear()
        MODELS[label or 'qwen3-embedding_0.6b'] = p
    for label in MODELS:
        for center in modes:
            run(label, center=center, show=False)


if __name__ == '__main__':
    main()
