# Semantic similarity for indicator lineage — method, experiments, findings

Working notes for the TactiX reward-distribution idea: embed Pine Script
indicators with a local embedding model, measure vector proximity, and use it
to detect parent/child relationships (original vs. remixed scripts) so usage
rewards can be split along the inscription curve.

Everything runs locally against Ollama (`qwen3-embedding:0.6b` default,
`qwen3-embedding:4b` as the second opinion — ~5–6× slower, not comparable
scores). No data leaves the machine.

---

## 1. The pipeline

```
source -> strip // comments -> overlapping 4000-char chunks
       -> ollama.embed per chunk -> mean-pool -> L2 normalize -> unit vector
```

- **Comment stripping** (`embed_indicators.py`, preprocess v2): comments
  describe *identity*, not *design*. License headers inflate similarity
  between unrelated scripts, and copied comments would let an uploader spoof
  the score. Only executable code is embedded.
- **Chunking**: scripts run up to ~2k lines, over the model's context window,
  so we embed 4000-char overlapping chunks (400 overlap, line-aligned) and
  mean-pool. Consequence: **chunk pooling is the system's blind spot** — see
  §5.
- **Centering**: all Pine embeddings share a large common direction ("this is
  Pine code"), which floors *raw* cosine at ~0.6 even for unrelated scripts.
  Each DB stores a frozen `reference_mean` = corpus mean; subtracting it and
  renormalizing leaves only what distinguishes a script. All thresholds below
  are for the **centered** cosine.

### Calibrated thresholds (0.6b, preprocess v2, 26-script corpus, Oct 2026)

| centered cosine | meaning |
|---|---|
| >= 0.80 | **REMIX** — likely derivative; reward split recommended |
| >= 0.50 | **related** — same design family; reported, no split |
| < 0.50 | distinct |

Calibration anchors: indicator vs its own strategy copy → 0.82–0.86;
same family (macd_ema vs mtf_ema) → ~0.72; unrelated median → ~-0.06,
p95 → ~0.38. Thresholds are only valid against the *frozen* mean — refreezing
on a different corpus shifts every score.

## 2. Tools

| file | purpose |
|---|---|
| `embed_indicators.py` | CLI: `build` (embed all → per-model JSON DB), `report` (pairwise matrix), `check <file>` (simulate upload → find parents) |
| `embed_files.ipynb` | pick which `.pine` files to (re)embed per model |
| `compare_two_scripts.ipynb` | one pair in depth: verdict, neighbours, chunk-level heatmap, contribution bars |
| `compare_texts.ipynb` / `compare_texts_app.py` | embed any two texts; the app is a browser UI (`localhost:8000`) with templates and per-pair JPG overviews in `comparisons/` |
| `merge_experiment.py` | splice two scripts 0→100% (concat or interleave), plot similarity vs each parent |
| `word_sweep_fourier.py` | token-by-token mutation drift A→B; contribution erosion heatmap + 2D trajectory map |
| `corpus_map.py` | all-vs-all: clustered similarity heatmap + labelled 2D PCA map with remix/related links |

## 3. Sanity checks — arbitrary text

Compared recipes and code in `compare_texts_app.py`:

| pair | raw cosine |
|---|---|
| birria recipe ×2 (identical) | 0.9998 |
| birria vs beef→lamb remix | 0.8482 |
| birria vs crumpet churros | 0.4449 |
| birria vs `macd_ema.pine` | 0.1204 |
| "moving average crossover strategy" vs paraphrase | 0.9049 |
| same vs "recipe for chocolate cake" | 0.4378 |

Read: the model orders these sensibly, but *raw* cosine never gets very low —
documents of the same type share a direction. Centering is what separates
"same family" from "same filetype".

## 4. Experiment — merge sweep (block splicing)

`merge_experiment.py`: `08_rsi_divergence.pine` (108 lines) × `macd_ema.pine`
(50 lines), hybrids from 0% to 100% B. Plots: `merge_sweep_*.jpg`.

**concat mode** (first X% of A + last (100−X)% of B):

| % from B | centered vs A | vs B | verdict |
|---|---|---|---|
| 0 | 1.00 | −0.07 | remix of A |
| 5 | **0.73** | −0.03 | related — *already below 0.80* |
| 15–50 | 0.71 → 0.57 | 0.29 → 0.45 | related |
| 55–80 | ~0.52 | ~0.48 | **dead zone** |
| 95 | 0.25 | 0.82 | remix of B |
| 100 | −0.07 | 1.00 | remix of B |

**interleave mode** (B's lines sprinkled through A at rate f):

| % from B | centered vs A | vs B | verdict |
|---|---|---|---|
| 0–25 | 1.00 → 0.88 | ~0 | **remix of A — held** |
| 40–55 | 0.81–0.86 | ~0.24 | remix of A — still held |
| 60–90 | 0.54–0.74 | 0.39–0.64 | related, noisy |
| 95–100 | 0.13 → −0.07 | 0.81 → 1.00 | remix of B |

**Findings:**

1. **Appending foreign code evades detection fast** — ~5% contiguous B already
   pulls the pooled vector under the remix bar, because B's lines form their
   *own chunks* and mean-pooling weighs chunks equally.
2. **Sprinkled foreign code is caught** — interleaved B stays "remix of A" to
   ~50–55%: every chunk still looks mostly like A.
3. **55–80% is a dead zone** either way — true merges score ~0.5 vs *both*
   parents, indistinguishable from "related".
4. Transition to remix-of-B only happens around ~95% B — highly asymmetric.
5. Non-monotonicity at low %: short hybrids have 1–2 chunks, so chunk-boundary
   placement jitters the score (0.73 → 0.80 → 0.71 on adjacent steps).

## 5. Experiment — word-mutation drift

`word_sweep_fourier.py`: replace random tokens of A one by one with tokens
drawn from B (cumulative), embed each mutant. Plots: contribution erosion
heatmap (dims ranked by their A×A contribution, watched as they cool) and a
2D PCA trajectory map of the mutant's path through semantic space.

**Recipe pair** (birria → churros vocab, 231 tokens):

| % replaced | cos vs birria | vs churros |
|---|---|---|
| 0 | 0.986 | 0.43 |
| 50 | 0.70 | 0.74 ← crossover |
| 100 | 0.42 | 0.83 |

**Code pair** (`08_rsi_divergence` → `macd_ema` vocab, 446 tokens, raw cos):

| % replaced | vs A | vs B |
|---|---|---|
| 0 | 0.963 | 0.658 |
| 25 | 0.905 | 0.774 |
| 60 | 0.816 | 0.817 ← crossover |
| 100 | **0.598** | 0.913 |

**Findings:**

- **The skeleton persists.** With *every* token replaced, the code mutant
  still scores 0.60 vs A (recipe: 0.42). Structure — ordering, syntax rhythm,
  line shapes — is a huge part of a script's embedding. Vocabulary swaps move
  the vector less than you'd expect.
- **Shared vocabulary compresses the range**: the two indicators already share
  `ta.*`/`input.*`/`plot` tokens, so crossover is later (~60%) and shallower
  than prose (~50%).
- Even "fully mutated" code caps ~0.91 vs B — a birria skeleton wearing
  churros words never becomes churros. Same for code.
- The first FFT version of the map looked nice but meant nothing (embedding
  dims are unordered, so there is no real spectrum); replaced by the erosion
  heatmap + trajectory map, which are directly interpretable.

## 6. Experiment — corpus map (all 26 × 26)

`corpus_map.py`: pairwise centered cosine over the whole corpus, greedily
ordered into family chains; plus a labelled PCA map with remix (red) and
related (gray) links. `corpus_heatmap_*.jpg`, `corpus_map_*.jpg`.

Strongest pairs, and where the two models disagree:

| pair | 0.6b | 4b* | note |
|---|---|---|---|
| 12_macd_crossover ↔ macd_ema | 0.855 REMIX | 0.804 REMIX | consensus |
| Pulse_mean_acc ↔ its strategy | 0.839 REMIX | 0.896 REMIX | consensus |
| zeierman_trend ↔ its strategy | 0.818 REMIX | **0.728 related** | borderline |
| macd_ema ↔ mtf_ema | 0.720 related | 0.651 related | same family |
| rsi_divergence ↔ Divergency | 0.482 | **0.500 related** | borderline |
| CME_Institutional ↔ Volume_Fib | 0.430 | 0.427 | distinct |

*4b DB has no frozen `reference_mean`; corpus mean recomputed for the map —
absolute values shift slightly vs the calibrated scale.

The two models agree on the obvious families and **disagree exactly on the
borderline cases** — the pairs where a second-model vote would matter most
for reward splits.

## 7. What this means for lineage detection

- Centered whole-vector cosine is a good *first* flag — clean separation of
  remix (>0.8) / related (0.5–0.8) / distinct in this corpus.
- **Failure mode 1 — appended blocks**: a copied script with foreign code
  bolted on drops under 0.8 quickly (concat sweep). Backstop: chunk-level
  max-similarity (`compare_two_scripts.ipynb` heatmap) still exposes the
  copied half.
- **Failure mode 2 — dead-zone merges**: ~50/50 hybrids sit at ~0.5 vs both
  parents. Whole-vector scoring alone cannot call these; chunk-level or
  region-level evidence is needed.
- **Failure mode 3 — threshold sensitivity**: low-% hybrids jitter around the
  boundary because chunking boundaries shift; scores within ±0.05 of a
  threshold shouldn't be treated as decisive without the second model's vote.
- Enrichment that survives all of this: structure. Token-for-token
  replacement leaves ~0.6 residual — plagiarism that keeps the skeleton is
  easier to catch than plagiarism that restructures.

## 8. Caveats

- Scores are **per-model**; never compare 0.6b numbers to 4b numbers.
- Centered scores depend on the frozen corpus mean; `build` keeps it frozen
  unless explicitly refrozen (which shifts *every* score → recalibrate
  thresholds after).
- Whole-vector cosine is pooled evidence, not proof: same design family
  without copying can land in "related". The remix flag should trigger a
  human/automated chunk-level review, not a payout decision alone.
- The mutation sweeps use random positions, single seed — fine for a picture
  of the shape, not statistics.
