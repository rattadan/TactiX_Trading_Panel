#!/usr/bin/env python3
"""Semantic embedding pipeline for TactiX indicator reward distribution.

Embeds Pine Script indicators with a local Ollama embedding model and stores
the semantic vectors in a JSON database. Proximity between vectors is used to
detect parent/child relationships (original vs. remixed indicators) so usage
rewards can be split along the inscription curve.

Usage:
    .venv/bin/python embed_indicators.py build              # embed all indicators -> DB
    .venv/bin/python embed_indicators.py report             # pairwise similarity matrix
    .venv/bin/python embed_indicators.py check <file.pine>  # simulate upload: find parents
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODEL = "qwen3-embedding:0.6b"
INDICATOR_DIR = Path(__file__).parent / "indicator-scripts"
DB_PATH = Path(__file__).parent / "indicator_embeddings.json"

# Bump whenever the text fed to the model changes; stale entries get re-embedded.
PREPROCESS_VERSION = 2  # v1: header strip only; v2: all // comments + blank lines stripped


def default_db_path(model: str) -> Path:
    """Per-model DB path so embeddings from different models don't collide."""
    tag = re.sub(r"[^\w.-]+", "_", model)
    return Path(__file__).parent / f"indicator_embeddings_{tag}.json"

# Chunking: qwen3-embedding:0.6b has a limited context window and these files
# run up to ~2k lines, so we embed overlapping chunks and mean-pool them.
CHUNK_CHARS = 4000
CHUNK_OVERLAP = 400

# Mean-centered cosine thresholds for lineage detection, calibrated against
# qwen3-embedding:0.6b, preprocess v2, on this repo's 26 scripts (Oct 2026):
#   indicator vs its own strategy / derived copy  0.82-0.86
#   same family (macd_ema vs mtf_ema)             ~0.72, then a gap down to 0.58
#   unrelated indicators                          median ~-0.06, p95 ~0.38
# Only valid for the frozen reference_mean in the DB — refreezing needs recalibration.
REMIX_THRESHOLD = 0.80   # likely a derivative / remix of the parent
RELATED_THRESHOLD = 0.50  # same design family (reported, no split)
MAX_PARENT_SHARE = 0.5    # max fraction of a child's rewards routed to a parent


def load_db() -> dict:
    if DB_PATH.exists():
        db = json.loads(DB_PATH.read_text())
        if db.get("model") != MODEL:
            sys.exit(f"{DB_PATH.name} was built with {db.get('model')!r}, not {MODEL!r}. "
                     f"Scores from different models are not comparable — use --model/--db.")
        return db
    return {"model": MODEL, "indicators": {}}


def save_db(db: dict) -> None:
    DB_PATH.write_text(json.dumps(db, indent=1))


def parse_metadata(source: str) -> dict:
    """Extract `// @key: value` header fields from a Pine Script file."""
    meta = {}
    for m in re.finditer(r"^\s*//\s*@(\w+)\s*:\s*(.+?)\s*$", source, re.M):
        key, value = m.group(1).lower(), m.group(2)
        if key == "reward_bps":
            try:
                value = int(re.sub(r"[^\d].*$", "", value))
            except ValueError:
                pass
        meta[key] = value
    return meta


def _strip_line_comment(line: str) -> str:
    """Cut a `//` comment from one line, ignoring `//` inside string literals."""
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif line.startswith("//", i):
            return line[:i]
        i += 1
    return line


def strip_comments(source: str) -> str:
    """Remove every `//` comment (header, metadata, inline) and blank lines.

    Comments describe identity, not design: license boilerplate inflates
    similarity between unrelated scripts, and padding or copying comments
    would let an uploader spoof the score. Only executable code is embedded.
    """
    body = []
    for line in source.splitlines():
        code = _strip_line_comment(line).rstrip()
        if code.strip():
            body.append(code)
    return "\n".join(body)


def chunk_source(source: str) -> list[str]:
    """Split source into overlapping character chunks (line-aligned)."""
    if len(source) <= CHUNK_CHARS:
        return [source]
    lines = source.splitlines(keepends=True)
    chunks, buf, size = [], [], 0
    for line in lines:
        buf.append(line)
        size += len(line)
        if size >= CHUNK_CHARS:
            chunks.append("".join(buf))
            tail, tsize = [], 0
            for l in reversed(buf):
                if tsize + len(l) > CHUNK_OVERLAP:
                    break
                tail.insert(0, l)
                tsize += len(l)
            buf, size = tail, tsize
    if buf:
        chunks.append("".join(buf))
    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts via the Ollama client. Returns (n, dim) array."""
    try:
        import ollama
    except ImportError:
        sys.exit("ollama package missing — run: .venv/bin/pip install ollama")
    try:
        resp = ollama.embed(model=MODEL, input=texts)
        vectors = resp.embeddings
    except AttributeError:  # older client API
        resp = ollama.embeddings(model=MODEL, prompt=texts[0])
        vectors = [resp["embedding"]]
    except Exception as e:
        sys.exit(
            f"Ollama embed failed: {e}\n"
            f"Is the server running and the model pulled?\n"
            f"  ollama serve\n  ollama pull {MODEL}"
        )
    return np.asarray(vectors, dtype=np.float64)


def embed_indicator(source: str) -> tuple[np.ndarray, int]:
    """Embed one indicator: strip comments -> chunk -> embed -> mean-pool -> normalize."""
    chunks = chunk_source(strip_comments(source))
    vecs = embed_texts(chunks)
    pooled = vecs.mean(axis=0)
    pooled /= np.linalg.norm(pooled) + 1e-12
    return pooled, len(chunks)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def center(v: np.ndarray, mu: np.ndarray | None) -> np.ndarray:
    """Subtract the frozen corpus mean and renormalize.

    All Pine Script embeddings share a large common direction ("it's Pine
    code"), which pushes even unrelated scripts to ~0.6 raw cosine. Removing
    it leaves only what distinguishes scripts: unrelated pairs land near 0.
    """
    if mu is None:
        return v
    c = v - mu
    return c / (np.linalg.norm(c) + 1e-12)


def reference_mean(db: dict) -> np.ndarray | None:
    mu = db.get("reference_mean")
    if mu is None:
        print("warning: DB has no reference_mean — using raw cosine (run `build`)\n")
        return None
    return np.asarray(mu)


def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def cmd_build(force: bool, refreeze: bool) -> None:
    db = load_db()
    files = sorted(INDICATOR_DIR.glob("*.pine"))
    if not files:
        sys.exit(f"No .pine files in {INDICATOR_DIR}")
    for f in files:
        source = f.read_text(encoding="utf-8", errors="replace")
        digest = source_hash(source)
        entry = db["indicators"].get(f.name, {})
        if (not force and entry.get("sha256") == digest and entry.get("vector")
                and entry.get("preprocess") == PREPROCESS_VERSION):
            print(f"  skip  {f.name} (unchanged)")
            continue
        vec, n_chunks = embed_indicator(source)
        meta = parse_metadata(source)
        now = datetime.now(timezone.utc).isoformat()
        db["indicators"][f.name] = {
            "sha256": digest,
            "title": meta.get("title", f.stem),
            "author": meta.get("author", meta.get("description")),
            "reward_address": meta.get("reward_address"),
            "reward_bps": meta.get("reward_bps"),
            "lines": source.count("\n") + 1,
            "chunks": n_chunks,
            "dim": int(vec.size),
            "preprocess": PREPROCESS_VERSION,
            "first_inscribed_at": entry.get("first_inscribed_at", now),
            "embedded_at": now,
            "vector": vec.tolist(),
        }
        print(f"  embed {f.name} ({meta.get('title', f.stem)!r}, "
              f"{n_chunks} chunks, dim={vec.size})", flush=True)
    stale_mean = db.get("reference_mean_preprocess") != PREPROCESS_VERSION
    if refreeze or stale_mean or "reference_mean" not in db:
        vecs = np.asarray([e["vector"] for e in db["indicators"].values()])
        db["reference_mean"] = vecs.mean(axis=0).tolist()
        db["reference_mean_n"] = len(vecs)
        db["reference_mean_preprocess"] = PREPROCESS_VERSION
        db["reference_mean_at"] = datetime.now(timezone.utc).isoformat()
        print(f"Froze reference mean over {len(vecs)} indicators "
              f"(|mean|={np.linalg.norm(vecs.mean(axis=0)):.3f})")
    db["preprocess"] = PREPROCESS_VERSION
    save_db(db)
    print(f"Wrote {DB_PATH} ({len(db['indicators'])} indicators)")


def lineage_table(names, sims, inscriptions):
    """Given a list of (name, similarity) pairs, pick parent suggestions."""
    rows = []
    for name, sim in sorted(zip(names, sims), key=lambda t: -t[1]):
        if sim >= REMIX_THRESHOLD:
            verdict = "REMIX   — reward split recommended"
        elif sim >= RELATED_THRESHOLD:
            verdict = "related — same design family"
        else:
            verdict = "distinct"
        rows.append((name, sim, verdict, inscriptions.get(name, "?")))
    return rows


def verdict_for(sim: float) -> str:
    if sim >= REMIX_THRESHOLD:
        return "remix"
    if sim >= RELATED_THRESHOLD:
        return "related"
    return "distinct"


def cmd_report(csv_path: Path | None, top: int) -> None:
    import csv
    db = load_db()
    inds = db["indicators"]
    names = sorted(inds, key=lambda n: inds[n]["first_inscribed_at"])
    if len(inds) < 2:
        sys.exit("Need at least 2 embedded indicators — run `build` first.")
    mu = reference_mean(db)
    vecs = {n: center(np.asarray(inds[n]["vector"]), mu) for n in names}

    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["indicator"] + names)
            for a in names:
                w.writerow([a] + [f"{cosine(vecs[a], vecs[b]):.4f}" for b in names])
        print(f"Wrote {csv_path} ({len(names)}x{len(names)} matrix)\n")

    pairs = sorted(
        ((cosine(vecs[a], vecs[b]), a, b)
         for i, a in enumerate(names) for b in names[i + 1:]),
        reverse=True)
    print(f"Top {min(top, len(pairs))} pairs by similarity:")
    for sim, a, b in pairs[:top]:
        print(f"  {sim:.4f}  {verdict_for(sim):<9} {a}  vs  {b}")
    print()

    print(f"Model: {db['model']}   dim: {next(iter(vecs.values())).size}   "
          f"similarity: {'mean-centered' if mu is not None else 'raw'} cosine\n")
    compact = len(names) > 10
    if not compact:
        print("Pairwise cosine similarity (rows/cols ordered by inscription time):")
        header = " " * 22 + "".join(f"{n[:14]:>16}" for n in names)
        print(header)
        for a in names:
            row = f"{a[:20]:<22}"
            for b in names:
                row += f"{cosine(vecs[a], vecs[b]):>16.4f}"
            print(row)
        print()
    for i, child in enumerate(names):
        others, sims, inscr = [], [], {}
        for j, other in enumerate(names):
            if other == child or j > i:  # only earlier inscriptions can be parents
                continue
            others.append(other)
            sims.append(cosine(vecs[child], vecs[other]))
            inscr[other] = inds[other]["first_inscribed_at"][:10]
        rows = lineage_table(others, sims, inscr)
        parents = [r for r in rows if r[1] >= REMIX_THRESHOLD]
        print(f"{child} ({inds[child]['title']!r})")
        if not rows:
            print("  (genesis indicator — no earlier inscriptions)\n")
            continue
        shown = [r for r in rows if not compact or r[1] >= RELATED_THRESHOLD]
        for name, sim, verdict, date in shown:
            print(f"  {sim:>6.4f}  {verdict:<40} vs {name} ({date})")
        if compact and len(shown) < len(rows):
            print(f"  ... and {len(rows) - len(shown)} distinct pairs below {RELATED_THRESHOLD}")
        if parents:
            total = sum(p[1] for p in parents)
            for name, sim, _, _ in parents:
                share = MAX_PARENT_SHARE * (sim / total)
                print(f"  -> suggested parent share to {name}: {share:.1%} of rewards")
        print()


def cmd_check(path: Path) -> None:
    """Simulate an upload: embed a new file, compare against the DB."""
    db = load_db()
    inds = db["indicators"]
    if not inds:
        sys.exit("DB is empty — run `build` first.")
    mu = reference_mean(db)
    source = path.read_text(encoding="utf-8", errors="replace")
    digest = source_hash(source)
    vec, n_chunks = embed_indicator(source)
    vec = center(vec, mu)
    meta = parse_metadata(source)
    print(f"New upload: {path.name} ({meta.get('title', path.stem)!r}, {n_chunks} chunks)\n")
    names, sims, inscr = [], [], {}
    for name, e in inds.items():
        if e.get("sha256") == digest:
            print(f"  (identical code already inscribed as {name})\n")
            continue
        names.append(name)
        sims.append(cosine(vec, center(np.asarray(e["vector"]), mu)))
        inscr[name] = e["first_inscribed_at"][:10]
    rows = lineage_table(names, sims, inscr)
    for name, sim, verdict, date in rows:
        print(f"  {sim:>6.4f}  {verdict:<40} {name} ({date})")
    parents = [r for r in rows if r[1] >= REMIX_THRESHOLD]
    if parents:
        total = sum(p[1] for p in parents)
        print("\nReward split suggestion:")
        for name, sim, _, _ in parents:
            print(f"  {MAX_PARENT_SHARE * (sim / total):.1%} -> {name}")
        print(f"  {1 - MAX_PARENT_SHARE:.1%} -> uploader (remainder)")
    else:
        print("\nNo parent above remix threshold — uploader keeps full rewards.")


def main() -> None:
    global MODEL, DB_PATH
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL, help=f"Ollama embedding model (default: {MODEL})")
    p.add_argument("--db", type=Path, default=None,
                   help="vector DB path (default: per-model indicator_embeddings_<model>.json for "
                        "non-default models, else indicator_embeddings.json)")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="embed all indicators into the vector DB")
    b.add_argument("--force", action="store_true", help="re-embed even if unchanged")
    b.add_argument("--refreeze-mean", action="store_true",
                   help="recompute the frozen reference mean (changes ALL scores)")
    r = sub.add_parser("report", help="similarity matrix + parent/child suggestions")
    r.add_argument("--csv", nargs="?", const=Path("indicator_similarity.csv"),
                   default=None, type=Path,
                   help="write pairwise matrix to CSV (default: indicator_similarity.csv)")
    r.add_argument("--top", type=int, default=15, help="number of top pairs to show")
    c = sub.add_parser("check", help="embed a file and find nearest existing indicators")
    c.add_argument("file", type=Path)
    args = p.parse_args()
    MODEL = args.model
    if args.db is not None:
        DB_PATH = args.db
    elif MODEL != "qwen3-embedding:0.6b":
        DB_PATH = default_db_path(MODEL)
    if args.cmd == "build":
        cmd_build(args.force, args.refreeze_mean)
    elif args.cmd == "report":
        cmd_report(args.csv, args.top)
    elif args.cmd == "check":
        if not args.file.exists():
            sys.exit(f"File not found: {args.file}")
        cmd_check(args.file)


if __name__ == "__main__":
    main()
