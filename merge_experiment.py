#!/usr/bin/env python3
"""Merge-sweep: how much modification moves a script's similarity score?

Splices script A (RSI-based) and script B (EMA-based) at increasing
fractions of B, embeds every hybrid with the same pipeline as `build`
(strip comments -> chunk -> mean-pool), and plots cosine vs each parent.

Two merge modes:
  concat      hybrid = first (1-f) of A's lines + last f of B's lines
              (models "indicator A with strategy/block B appended")
  interleave  line i comes from B at ~f of the positions, else from A
              (models a line-by-line rewrite blend)

Usage:
    .venv/bin/python merge_experiment.py                 # default pair, concat
    .venv/bin/python merge_experiment.py --interleave
    .venv/bin/python merge_experiment.py A.pine B.pine [--interleave]
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import embed_indicators as ei

MODEL = 'qwen3-embedding:0.6b'
DB = Path(__file__).parent / 'indicator_embeddings.json'
STEPS = 21  # 0%, 5%, ..., 100% of B

DEFAULT_A = '08_rsi_divergence.pine'  # RSI-based
DEFAULT_B = 'macd_ema.pine'           # EMA-based


def merge_concat(src_a, src_b, frac_b):
    la, lb = src_a.splitlines(), src_b.splitlines()
    return '\n'.join(la[:int((1 - frac_b) * len(la))]
                     + lb[int((1 - frac_b) * len(lb)):])


def merge_interleave(src_a, src_b, frac_b):
    la, lb = src_a.splitlines(), src_b.splitlines()
    n = max(len(la), len(lb))
    from_b = {round(i / frac_b) for i in range(1, int(frac_b * n) + 1)} if frac_b else set()
    return '\n'.join((lb[i] if i < len(lb) else '') if i in from_b
                     else (la[i] if i < len(la) else '')
                     for i in range(n))


def embed_pine(source):
    """Same as build: strip comments -> chunk -> embed -> mean-pool -> unit norm."""
    vec, _ = ei.embed_indicator(source)
    return vec


def run(a_name=DEFAULT_A, b_name=DEFAULT_B, model=MODEL, mode='concat',
        n_steps=STEPS, save=True, show=True):
    """Run the merge sweep for one pair. Returns (rows, output_path)."""
    merge = merge_interleave if mode == 'interleave' else merge_concat

    ei.MODEL = model
    db_path = (DB if model == 'qwen3-embedding:0.6b'
               else ei.default_db_path(model))
    db = json.loads(db_path.read_text())
    mu = np.asarray(db['reference_mean']) if 'reference_mean' in db else None
    va_db = np.asarray(db['indicators'][a_name]['vector'])
    vb_db = np.asarray(db['indicators'][b_name]['vector'])

    src_a = (ei.INDICATOR_DIR / a_name).read_text(encoding='utf-8', errors='replace')
    src_b = (ei.INDICATOR_DIR / b_name).read_text(encoding='utf-8', errors='replace')

    print(f'{a_name} ({src_a.count(chr(10)) + 1} lines)  vs  '
          f'{b_name} ({src_b.count(chr(10)) + 1} lines)   mode={mode}\n')
    print(f'{"%B":>4}  {"cos vs A":>9} {"cos vs B":>9}   '
          f'{"ctr vs A":>9} {"ctr vs B":>9}   verdict')
    print('-' * 72)

    rows = []
    for step in range(n_steps):
        f = step / (n_steps - 1)
        vh = embed_pine(merge(src_a, src_b, f))
        ra, rb = ei.cosine(vh, va_db), ei.cosine(vh, vb_db)
        if mu is not None:
            ch = ei.center(vh, mu)
            ca = ei.cosine(ch, ei.center(va_db, mu))
            cb = ei.cosine(ch, ei.center(vb_db, mu))
        else:
            ca = cb = float('nan')
        if ca >= ei.REMIX_THRESHOLD:
            verdict = f'REMIX of A (>= {ei.REMIX_THRESHOLD})'
        elif cb >= ei.REMIX_THRESHOLD:
            verdict = f'REMIX of B (>= {ei.REMIX_THRESHOLD})'
        elif ca >= ei.RELATED_THRESHOLD or cb >= ei.RELATED_THRESHOLD:
            verdict = f'related (>= {ei.RELATED_THRESHOLD})'
        else:
            verdict = 'distinct'
        rows.append((f, ra, rb, ca, cb))
        print(f'{f * 100:>3.0f}%  {ra:>9.4f} {rb:>9.4f}   {ca:>9.4f} {cb:>9.4f}   {verdict}',
              flush=True)

    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = [r[0] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, [r[1] for r in rows], 'o-', color='tab:blue', alpha=0.45,
            label='raw cos vs A')
    ax.plot(x, [r[2] for r in rows], 'o-', color='tab:red', alpha=0.45,
            label='raw cos vs B')
    ax.plot(x, [r[3] for r in rows], 'o-', lw=2.5, color='tab:blue',
            label='centered cos vs A')
    ax.plot(x, [r[4] for r in rows], 'o-', lw=2.5, color='tab:red',
            label='centered cos vs B')
    if mu is not None:
        ax.axhline(ei.REMIX_THRESHOLD, color='k', ls='--', lw=1)
        ax.axhline(ei.RELATED_THRESHOLD, color='k', ls=':', lw=1)
        ax.text(1, ei.REMIX_THRESHOLD + 0.02, f'REMIX {ei.REMIX_THRESHOLD}', fontsize=8)
        ax.text(1, ei.RELATED_THRESHOLD + 0.02, f'RELATED {ei.RELATED_THRESHOLD}', fontsize=8)
    ax.set_xlabel('% of hybrid taken from B')
    ax.set_ylabel('cosine similarity')
    ax.set_title(f'merge sweep ({mode}): {a_name} <-> {b_name} — {model}')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(f'merge_sweep_{mode}_{Path(a_name).stem}_vs_{Path(b_name).stem}.jpg')
    fig.savefig(out, format='jpg', dpi=110, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f'\nsaved {out}')
    return rows, out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    interleave = '--interleave' in sys.argv
    run(a_name=args[0] if len(args) > 0 else DEFAULT_A,
        b_name=args[1] if len(args) > 1 else DEFAULT_B,
        mode='interleave' if interleave else 'concat',
        show=False)


if __name__ == '__main__':
    main()
