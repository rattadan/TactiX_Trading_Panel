#!/usr/bin/env python3
"""Browser UI for comparing two texts semantically — no ipywidgets needed.

Same pipeline as compare_texts.ipynb: chunk -> Ollama embed -> mean-pool ->
normalize -> cosine/euclidean + per-dim contribution & disagreement bar charts.

Run:   .venv/bin/python compare_texts_app.py
Open:  http://localhost:8000
"""

import base64
import io
import json
import re
import sys
import textwrap
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import embed_indicators as ei  # reuse chunking + Ollama client

MODELS = ['qwen3-embedding:0.6b', 'qwen3-embedding:4b']
PORT = 8000
OUT_DIR = Path(__file__).parent / 'comparisons'  # one JPG overview per pair

BIRRIA = """Method
Heat a small, dry frying pan over a low-medium heat. Toast the paprika, cumin, oregano, thyme and cinnamon for 30 seconds until fragrant, then set aside.

Heat a little oil in a large frying pan and sear the beef until browned on all sides (depending on the size of your pan, you may need to be this in batches). Remove from the pan and set aside.

Add the onion and garlic to the pan and cook until soft and lightly browned. Add the blended tomatoes, stock, vinegar, chipotle paste and toasted spices. Season generously with salt and pepper. Transfer to a blender or food processor and blend until smooth.

Return the blended sauce to the pan or transfer to a slow cooker. Add the beef and bay leaves to the pan or slow cooker. Cook on the hob over a very low heat for 2-3 hours, or in the slow cooker for 4-5 hours on low. When cooked the beef will be very tender and shred easily.

Transfer the beef to a plate and shred it using two forks. Return to the pan and stir the beef through the sauce.

At this point, the birria can be cooled and frozen, or served immediately with the sides of your choice. If serving straight away, top with finely chopped onion and fresh coriander. See the recipe tips, below, for how to reheat."""

BIRRIA_REMIX = (BIRRIA
                .replace('beef', 'lamb')
                .replace('chipotle paste', 'smoked chilli paste')
                .replace('2-3 hours', '3-4 hours')
                .replace('slow cooker for 4-5 hours', 'slow cooker for 6-7 hours'))

CHURROS = """Method
Place the crumpet fingers in a bowl, drizzle over the melted butter and stir to coat the crumpets.

Mix the sugar and cinnamon together in a separate bowl, and then sprinkle onto the crumpet fingers.

Mix everything together then place in the air fryer for 5-6 minutes at 180C.

To make the sauce, whisk all of the ingredients in a pan and heat until it thickens, around 5 minutes.

Add extra sugar and cinnamon to the churros to taste, and serve alongside the chocolate sauce for dipping."""


def _load_templates():
    """Preset A/B pairs selectable from the page."""
    templates = [
        {'name': 'identical: birria recipe x2',
         'a': BIRRIA, 'b': BIRRIA},
        {'name': 'remixed: birria beef -> lamb',
         'a': BIRRIA, 'b': BIRRIA_REMIX},
        {'name': 'different: birria vs churros',
         'a': BIRRIA, 'b': CHURROS},
        {'name': 'paraphrase: MA crossover',
         'a': 'a moving average crossover strategy',
         'b': 'a strategy that trades MA crossovers'},
        {'name': 'unrelated: trading vs cake',
         'a': 'a moving average crossover strategy',
         'b': 'the recipe for chocolate cake'},
    ]
    pa = ei.INDICATOR_DIR / 'macd_ema.pine'
    pb = ei.INDICATOR_DIR / 'mtf_ema.pine'
    if pa.exists() and pb.exists():
        templates.append({
            'name': 'pine: macd_ema vs mtf_ema (same family)',
            'a': pa.read_text(encoding='utf-8', errors='replace'),
            'b': pb.read_text(encoding='utf-8', errors='replace'),
        })
    return templates


TEMPLATES = _load_templates()


def db_path_for(model):
    if model == 'qwen3-embedding:0.6b':
        return Path(__file__).parent / 'indicator_embeddings.json'
    return ei.default_db_path(model)


def load_reference_mean(model):
    path = db_path_for(model)
    if path.exists():
        db = json.loads(path.read_text())
        if db.get('model') == model and db.get('reference_mean'):
            return np.asarray(db['reference_mean'])
    return None


def embed_text(text, model):
    ei.MODEL = model
    chunks = ei.chunk_source(text)
    vecs = ei.embed_texts(chunks)
    pooled = vecs.mean(axis=0)
    return pooled / (np.linalg.norm(pooled) + 1e-12), len(chunks)


def contrib_png(va, vb, centered, model):
    """Render the two bar charts to a base64 PNG for the page."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    _plot_bars(ax1, ax2, va, vb, centered, model)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _plot_bars(ax1, ax2, va, vb, centered, model):
    contrib = va * vb
    diff2 = (va - vb) ** 2
    order = np.argsort(-np.abs(contrib))
    ax1.bar(np.arange(len(contrib)), contrib[order], width=1.0)
    ax1.axhline(0, color='k', lw=0.5)
    ax1.set_title(f'per-dim contribution a[i]*b[i] '
                  f'({"centered" if centered else "raw"}, sorted by |contrib|, {model})')
    ax1.set_ylabel('contribution to cosine')
    ax2.bar(np.arange(len(diff2)), diff2[order], width=1.0, color='tomato')
    ax2.set_title('per-dim disagreement (a[i]-b[i])^2 (same dim order)')
    ax2.set_ylabel('squared diff')
    ax2.set_xlabel('dimension rank (not dimension id)')


def save_overview(r, va, vb, centered, text_a, text_b):
    """Write a one-file JPG overview: metrics header + the two bar charts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(exist_ok=True)
    fname = f'compare_{datetime.now():%Y%m%d_%H%M%S}_cos{r["cosine_similarity"]:.3f}.jpg'

    lines = [
        f'model: {r["model"]}   dim={r["dim"]}   '
        f'chunks A/B: {r["chunks_a"]}/{r["chunks_b"]}',
        f'cosine similarity: {r["cosine_similarity"]:.4f}   '
        f'cosine distance: {r["cosine_distance"]:.4f}   '
        f'euclidean distance: {r["euclidean_distance"]:.4f}',
    ]
    if 'centered_cosine' in r:
        c = r['centered_cosine']
        lines.append(f'centered cosine: {"n/a (no reference_mean)" if c is None else f"{c:.4f}"}')
    for tag, text in (('A', text_a), ('B', text_b)):
        wrapped = textwrap.wrap(' '.join(text.split()), 110)
        lines.append(f'{tag}: ' + '\n    '.join(wrapped[:3]) + (' ...' if len(wrapped) > 3 else ''))

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 2])
    ax0 = fig.add_subplot(gs[0])
    ax0.axis('off')
    ax0.text(0, 0.95, '\n'.join(lines), va='top', fontsize=9, family='monospace')
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    _plot_bars(ax1, ax2, va, vb, centered, r['model'])
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, format='jpg', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return fname


def compare(a, b, model, center=False):
    va, na = embed_text(a, model)
    vb, nb = embed_text(b, model)
    sim = ei.cosine(va, vb)
    result = {
        'model': model, 'dim': int(va.size), 'chunks_a': na, 'chunks_b': nb,
        'cosine_similarity': sim,
        'cosine_distance': 1.0 - sim,
        'euclidean_distance': float(np.linalg.norm(va - vb)),
    }
    centered = False
    if center:
        mu = load_reference_mean(model)
        if mu is not None:
            va, vb = ei.center(va, mu), ei.center(vb, mu)
            result['centered_cosine'] = ei.cosine(va, vb)
            centered = True
        else:
            result['centered_cosine'] = None
    result['chart'] = contrib_png(va, vb, centered, model)
    result['file'] = save_overview(result, va, vb, centered, a, b)
    return result


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Compare texts</title>
<style>
 body { font-family: system-ui, sans-serif; max-width: 980px; margin: 24px auto; }
 textarea { width: 100%; height: 160px; font-family: monospace; }
 .row { display: flex; gap: 16px; align-items: center; margin: 12px 0; }
 pre { background: #f4f4f4; padding: 12px; }
 img { max-width: 100%; }
</style></head><body>
<h2>Compare two texts semantically</h2>
<div class="row">
  <label>model <select id="model">MODEL_OPTIONS</select></label>
  <label>template <select id="tpl" onchange="loadTpl()">
    <option value="">— pick one to fill A and B —</option>TEMPLATE_OPTIONS
  </select></label>
  <label><input type="checkbox" id="center"> center by corpus mean (Pine only)</label>
  <button onclick="run()">Compare</button>
</div>
<b>Text A</b><textarea id="a" placeholder="paste anything — code, prose, prompt"></textarea>
<b>Text B</b><textarea id="b"></textarea>
<div id="status"></div>
<div id="file"></div>
<pre id="res" style="display:none"></pre>
<img id="chart" style="display:none">
<script>
async function loadTpl() {
  const i = document.getElementById('tpl').value;
  if (i === '') return;
  const r = await fetch('/api/template?i=' + i);
  const d = await r.json();
  document.getElementById('a').value = d.a;
  document.getElementById('b').value = d.b;
}
async function run() {
  const s = document.getElementById('status');
  s.textContent = 'embedding...';
  document.getElementById('res').style.display = 'none';
  document.getElementById('chart').style.display = 'none';
  document.getElementById('file').textContent = '';
  const q = new URLSearchParams({
    a: document.getElementById('a').value,
    b: document.getElementById('b').value,
    model: document.getElementById('model').value,
    center: document.getElementById('center').checked ? '1' : '0',
  });
  const r = await fetch('/api/compare?' + q);
  const d = await r.json();
  if (d.error) { s.textContent = 'error: ' + d.error; return; }
  s.textContent = '';
  let txt = `model             ${d.model}   (dim=${d.dim}, chunks: A=${d.chunks_a}, B=${d.chunks_b})\\n` +
            `--------------------------------------------------------\\n` +
            `cosine similarity   ${d.cosine_similarity.toFixed(4)}\\n` +
            `cosine distance     ${d.cosine_distance.toFixed(4)}\\n` +
            `euclidean distance  ${d.euclidean_distance.toFixed(4)}\\n`;
  if ('centered_cosine' in d)
    txt += d.centered_cosine === null
      ? 'centered cosine      n/a (no reference_mean in DB)\\n'
      : `centered cosine     ${d.centered_cosine.toFixed(4)}\\n`;
  const res = document.getElementById('res');
  res.textContent = txt; res.style.display = 'block';
  const img = document.getElementById('chart');
  img.src = 'data:image/png;base64,' + d.chart; img.style.display = 'block';
  if (d.file)
    document.getElementById('file').innerHTML =
      `saved: <a href="/files/${d.file}" target="_blank">comparisons/${d.file}</a>`;
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/':
            page = (PAGE
                    .replace('MODEL_OPTIONS', ''.join(
                        f'<option>{m}</option>' for m in MODELS))
                    .replace('TEMPLATE_OPTIONS', ''.join(
                        f'<option value="{i}">{t["name"]}</option>'
                        for i, t in enumerate(TEMPLATES))))
            self._send(200, 'text/html', page.encode())
        elif u.path == '/api/template':
            q = parse_qs(u.query)
            i = int(q.get('i', ['0'])[0])
            t = TEMPLATES[i]
            self._send(200, 'application/json',
                       json.dumps({'a': t['a'], 'b': t['b']}).encode())
        elif u.path == '/api/compare':
            q = parse_qs(u.query)
            try:
                r = compare(q['a'][0], q['b'][0], q.get('model', [MODELS[0]])[0],
                            q.get('center', ['0'])[0] == '1')
                self._send(200, 'application/json', json.dumps(r).encode())
            except SystemExit as e:
                self._send(200, 'application/json',
                           json.dumps({'error': str(e)}).encode())
            except Exception as e:
                self._send(500, 'application/json',
                           json.dumps({'error': f'{type(e).__name__}: {e}'}).encode())
        elif u.path.startswith('/files/'):
            name = Path(u.path).name  # strip any path traversal
            f = OUT_DIR / name
            if f.is_file():
                self._send(200, 'image/jpeg', f.read_bytes())
            else:
                self._send(404, 'text/plain', b'not found')
        else:
            self._send(404, 'text/plain', b'not found')

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    print(f'open http://localhost:{PORT}  (Ctrl-C to stop)')
    HTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
