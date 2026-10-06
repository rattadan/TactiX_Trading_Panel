#!/usr/bin/env python3
"""Desktop GUI for the semantic-similarity toolkit.

Two text fields, model picker over every pulled Ollama embedding model,
metric options, template loader, corpus neighbour search, and an embedded
contribution/disagreement bar chart. Embeddings run on a worker thread so
the window stays responsive.

Run:  .venv/bin/python compare_gui.py
Requires `ollama serve` and the models pulled.
"""

import json
import re
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import embed_indicators as ei
import compare_texts_app as app  # reuse BIRRIA/CHURROS/templates/embed_text

MODELS = ['qwen3-embedding:0.6b', 'qwen3-embedding:4b',
          'nomic-embed-text', 'mxbai-embed-large', 'embeddinggemma']

TEMPLATES = app.TEMPLATES  # same presets as the web UI


def db_path_for(model):
    if model == 'qwen3-embedding:0.6b':
        return Path(__file__).parent / 'indicator_embeddings.json'
    return ei.default_db_path(model)


class CompareGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('TactiX semantic compare')
        self.geometry('1180x760')

        # ---- controls ------------------------------------------------------
        top = ttk.Frame(self, padding=6)
        top.pack(fill='x')

        ttk.Label(top, text='model:').pack(side='left')
        self.model = ttk.Combobox(top, values=MODELS, state='readonly', width=26)
        self.model.set(MODELS[0])
        self.model.pack(side='left', padx=4)

        ttk.Label(top, text='template:').pack(side='left', padx=(10, 0))
        self.tpl = ttk.Combobox(top, values=[t['name'] for t in TEMPLATES],
                                state='readonly', width=34)
        self.tpl.pack(side='left', padx=4)
        self.tpl.bind('<<ComboboxSelected>>', self.load_template)

        self.center_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text='center by corpus mean',
                        variable=self.center_var).pack(side='left', padx=6)
        self.pine_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text='strip // comments (Pine)',
                        variable=self.pine_var).pack(side='left')

        btns = ttk.Frame(self, padding=6)
        btns.pack(fill='x')
        ttk.Button(btns, text='Compare A vs B', command=self.on_compare).pack(side='left')
        ttk.Button(btns, text='Find neighbours of A in corpus',
                   command=self.on_neighbours).pack(side='left', padx=6)
        self.status = ttk.Label(btns, text='ready')
        self.status.pack(side='left', padx=12)

        # ---- text inputs ----------------------------------------------------
        panes = ttk.PanedWindow(self, orient='horizontal')
        panes.pack(fill='both', expand=True, padx=6)

        fa = ttk.Frame(panes)
        ttk.Label(fa, text='Text A').pack(anchor='w')
        self.text_a = ScrolledText(fa, wrap='word', font=('monospace', 10))
        self.text_a.pack(fill='both', expand=True)
        fb = ttk.Frame(panes)
        ttk.Label(fb, text='Text B').pack(anchor='w')
        self.text_b = ScrolledText(fb, wrap='word', font=('monospace', 10))
        self.text_b.pack(fill='both', expand=True)
        panes.add(fa, weight=1)
        panes.add(fb, weight=1)

        # ---- output + chart -------------------------------------------------
        bottom = ttk.PanedWindow(self, orient='horizontal')
        bottom.pack(fill='both', expand=True, padx=6, pady=6)

        outf = ttk.Frame(bottom)
        ttk.Label(outf, text='result').pack(anchor='w')
        self.out = ScrolledText(outf, font=('monospace', 10), height=14,
                                state='disabled')
        self.out.pack(fill='both', expand=True)
        bottom.add(outf, weight=1)

        chartf = ttk.Frame(bottom)
        ttk.Label(chartf, text='contribution / disagreement').pack(anchor='w')
        self.chart_holder = ttk.Frame(chartf)
        self.chart_holder.pack(fill='both', expand=True)
        bottom.add(chartf, weight=1)
        self._chart_canvas = None

    # ---- helpers -----------------------------------------------------------

    def log(self, text):
        self.out.config(state='normal')
        self.out.delete('1.0', 'end')
        self.out.insert('1.0', text)
        self.out.config(state='disabled')

    def busy(self, msg):
        self.status.config(text=msg)
        self.update_idletasks()

    def background(self, work, done):
        """Run work() off the UI thread, hand result to done() back on it."""
        def runner():
            try:
                r = work()
            except SystemExit as e:
                self.after(0, lambda: (self.busy('error'), self.log(str(e))))
                return
            except Exception as e:
                self.after(0, lambda: (self.busy('error'),
                                       self.log(f'{type(e).__name__}: {e}')))
                return
            self.after(0, lambda: (self.busy('ready'), done(r)))
        threading.Thread(target=runner, daemon=True).start()

    def prep(self, text):
        """Optional Pine comment-strip before embedding."""
        return ei.strip_comments(text) if self.pine_var.get() else text

    def embed(self, text, model):
        ei.MODEL = model
        chunks = ei.chunk_source(self.prep(text))
        vecs = ei.embed_texts(chunks)
        pooled = vecs.mean(axis=0)
        return pooled / (np.linalg.norm(pooled) + 1e-12), len(chunks)

    def reference_mean(self, model):
        p = db_path_for(model)
        if p.exists():
            db = json.loads(p.read_text())
            if db.get('model') == model and db.get('reference_mean'):
                return db, np.asarray(db['reference_mean'])
        return (json.loads(p.read_text()) if p.exists()
                else {'model': model, 'indicators': {}}), None

    # ---- actions -----------------------------------------------------------

    def load_template(self, _evt=None):
        t = TEMPLATES[self.tpl.current()]
        for w, v in ((self.text_a, t['a']), (self.text_b, t['b'])):
            w.delete('1.0', 'end')
            w.insert('1.0', v)

    def on_compare(self):
        a, b = self.text_a.get('1.0', 'end-1c'), self.text_b.get('1.0', 'end-1c')
        if not a.strip() or not b.strip():
            self.log('Both texts must be non-empty.')
            return
        model, center = self.model.get(), self.center_var.get()
        self.busy(f'embedding with {model} ...')

        def work():
            va, na = self.embed(a, model)
            vb, nb = self.embed(b, model)
            db, mu = self.reference_mean(model) if center else (None, None)
            return va, vb, na, nb, mu

        def done(r):
            va, vb, na, nb, mu = r
            sim = ei.cosine(va, vb)
            lines = [
                f'model             {model}   (dim={va.size}, chunks: A={na}, B={nb})',
                '-' * 60,
                f'cosine similarity   {sim:>8.4f}',
                f'cosine distance     {1 - sim:>8.4f}',
                f'euclidean distance  {np.linalg.norm(va - vb):>8.4f}',
            ]
            va_plot, vb_plot, centered = va, vb, False
            if center:
                if mu is None:
                    lines.append('centered cosine      n/a (no reference_mean in DB)')
                else:
                    ca, cb = ei.center(va, mu), ei.center(vb, mu)
                    c = ei.cosine(ca, cb)
                    tag = ('REMIX?' if c >= ei.REMIX_THRESHOLD
                           else 'related' if c >= ei.RELATED_THRESHOLD else '')
                    lines.append(f'centered cosine     {c:>8.4f}  {tag}')
                    va_plot, vb_plot, centered = ca, cb, True
            self.log('\n'.join(lines))
            self.draw_bars(va_plot, vb_plot, centered, model)

        self.background(work, done)

    def on_neighbours(self):
        a = self.text_a.get('1.0', 'end-1c')
        if not a.strip():
            self.log('Text A is empty.')
            return
        model = self.model.get()
        self.busy(f'embedding A + scanning corpus ({model}) ...')

        def work():
            va, na = self.embed(a, model)
            db, mu = self.reference_mean(model)
            return va, na, db, mu

        def done(r):
            va, na, db, mu = r
            inds = db.get('indicators', {})
            if not inds:
                self.log(f'no corpus DB for {model} — run build first')
                return
            use_center = mu is not None
            q = ei.center(va, mu) if use_center else va
            rows = []
            for n, e in inds.items():
                v = np.asarray(e['vector'])
                v = ei.center(v, mu) if use_center else v
                rows.append((ei.cosine(q, v), e.get('title', n), n))
            rows.sort(reverse=True)
            scope = 'centered' if use_center else 'raw'
            lines = [f'A -> corpus ({model}, {scope} cosine, {na} chunks)', '-' * 60]
            for s, title, n in rows[:10]:
                tag = ('REMIX' if s >= ei.REMIX_THRESHOLD
                       else 'related' if s >= ei.RELATED_THRESHOLD else '')
                lines.append(f'{s:>7.4f}  {tag:<8} {title}  ({n})')
            self.log('\n'.join(lines))

        self.background(work, done)

    def draw_bars(self, va, vb, centered, model):
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            return
        contrib = va * vb
        diff2 = (va - vb) ** 2
        order = np.argsort(-np.abs(contrib))

        if self._chart_canvas:
            self._chart_canvas.get_tk_widget().destroy()
        fig = Figure(figsize=(6, 4), dpi=90)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax1.bar(np.arange(len(contrib)), contrib[order], width=1.0)
        ax1.axhline(0, color='k', lw=0.5)
        ax1.set_title(f'contribution a[i]*b[i] ({"centered" if centered else "raw"})',
                      fontsize=9)
        ax2.bar(np.arange(len(diff2)), diff2[order], width=1.0, color='tomato')
        ax2.set_title('disagreement (a[i]-b[i])^2', fontsize=9)
        fig.tight_layout()
        self._chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_holder)
        self._chart_canvas.get_tk_widget().pack(fill='both', expand=True)
        self._chart_canvas.draw()


if __name__ == '__main__':
    CompareGUI().mainloop()
