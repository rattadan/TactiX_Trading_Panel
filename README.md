Strategies & Indicators

This folder contains the Pine Script strategies and indicators that power the
charting / backtesting engine in this app 

## Adding your own indicator or strategy

We want indicator and strategy authors to earn rewards when traders use
their work. If you publish a script here, you can attach a payout address so
a share of usage / subscription fees flows back to you.

**How to opt in:**

1. Add a metadata header at the top of your `.pine` file:

   ```pine
   // @title: My Awesome Indicator
   // @author: YourName
   // @reward_address: 0xABCDEF1234....
   // @reward_bps: 500   // optional, e.g. 5% of attributable fees
   ```

2. Open a PR adding your script. Include a short description of what it
   does and a screenshot of it running on a chart.

3. Rewards are distributed periodically based on usage metrics (times
   loaded, backtests run, and live-chart time) tracked by the app.

**Guidelines:**

- Only submit scripts you wrote or have rights to redistribute. Respect the
  original license (many TradingView scripts are MPL-2.0 — keep the license
  header intact and credit the original author).
- Reward addresses must be Injective `inj1...` addresses (EVM `0x` support
  planned).
- Scripts that are broken, plagiarized without attribution, or malicious
  (e.g. repaint scams presented as signals) will be removed and forfeit
  rewards.

Questions or want to propose a different reward split? Open an issue or
reach out to the maintainers.


## Crypto World's Fair Colosseum Hackathon

As contribution to the Crypto World's Fair Hackathon 2026 we would like to design the Indicator & Strategy rewards system as an opensourced part of our product suite.
There is a big community and even commercial creators for indicator already, offering their work as either opensourced (source code available), public (usable by TradingView Pro subscribers, but no visible sourcecode, as the pinescript execution runs serversided on Tradingview), and private (pay per indicator, also no sourcecode, pinescript execution serversided).

TactiX Pinescript execution instead runs as a webworker process client sided, so actuall all indicators are available in plain code. This makes it diffcult to reward creators for their work, as copycats have an easy play.

**Indicator Reward Distribution System**

To counterpart this, TactiX would like to create a new kind of opensourcing method:
During Code upload, the code is parsed to a LLM embedding model, returning a semantic vector that is being saved along into the indicator database contract. 
This semantic vector describes the "use" or "design" of the indicator, so two almost identical indicators would have a close proximity to each other, despite their code would look completely different.
The idea is to track semantic inscriptions and determine the reward output along that curve. 

In example this would mean: you create a new kind of indicator and upload it. Now someone else remixes your indicator and adds extra features etc. Both vectors will be close to each other, but the first one subscribed is interpreted as the parent. Now a trader uses the remixed Indicator, but the app recognizes the proximity between both and would split rewards to the parent and the child. 

Its the first time we go onto something like this, so it will be an interesting journey within the next 4 weeks.

Finally, the dApp should be able to redistribute "buildercodes" to code creators, even with plain sourcecodes

Keep track on the progress and check:

https://colosseum.com/arena/projects/tactix-trading-panel

In practice, the indicator subscription should flow like this:

The Dashboard pushes the indicator code to our embedding model getting the vector data as return. Now the contract needs to compare this vector against already saved vectors in its database and calculate the semantic distance of it. 
Proximity of 1.000 to 0.950 means, the newly subscribed Code is already present in the database, entitling the code as a remix of an already existent indicator. In this case the parent should receive the majority of the reward (>50%, precisely calculated against a curve between 1.00 and 0.95 somehow

A score between 0.950 and 0.900 would mean the indicator uses code from the parent, but add significant changes, so in this case the parent would receieve just a small part of the reward <50%, calced down to 0.900

A code score <0.90 would mean, the indicator is not a replica or remix, or at least differs more than needed to claim it a remix. in this case the full rewards would go to the subscriber.

It would be possible to run this query beforehand the contract inscription, so the user would be able to change his code accordingly, or to accept the algorithms decision. With inscription, the reward share would be set "in stone" into the conract.

If several indicators chains against each other, using a child would trigger a rewards split cascade: 

User uses Remix-2 indicator, which is split 25% to Remix-1 and 75% for itself

then those 25% coming to Remix-1 would be split again, accoring the Remix-1 splitting attributes (e.g. 45% for parent, 65% for himself)

So the parent (father of both Remix-1 and Remix-2) would receive 45% of those 25%   (calc on your own ..... approx. 12%)

Drawback is that very inscription will cost more gas, as more data needs to be processed and stored. Not sure how this will scale on a large picture (like 5000 vectors to compare against the new one).. something to work out in the ftuture

To prevent inscription frontrunning, the dApp adds a ECDSA signature to the contract call, so the message body would be dropped by the contract if someone would try to change the reward address or inscribe himself earlier.


## Semantic similarity toolkit — method & findings

Full writeup with experiment data: **[SEMANTIC_SIMILARITY.md](SEMANTIC_SIMILARITY.md)**

Experimental tooling for the reward-distribution idea above: embed indicator
source code with a local Ollama embedding model, measure vector proximity,
and use it to detect parent/child (original vs. remixed) scripts.

### Files

| file                                           | what it does                                                                                                                                                                           |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `embed_indicators.py`                          | CLI: `build` embeds all `.pine` in `indicator-scripts/` into a per-model JSON DB; `report` prints the pairwise similarity matrix; `check <file>` simulates an upload and finds parents |
| `embed_files.ipynb`                            | interactive companion to `build` — pick which files to (re)embed per model                                                                                                             |
| `compare_two_scripts.ipynb`                    | deep-dive on one pair: verdict vs thresholds, nearest neighbours, chunk-level similarity heatmap, per-dimension contribution bars                                                      |
| `compare_texts.ipynb` / `compare_texts_app.py` | embed any two texts (not just Pine) and compare; the `.py` is a browser UI at `localhost:8000` with templates and saves a JPG overview per pair to `comparisons/`                      |
| `merge_experiment.py`                          | merge-sweep: splices two scripts at 0–100% and plots similarity vs each parent                                                                                                         |

Requires `ollama serve` + `ollama pull qwen3-embedding:0.6b` (or `:4b`).

### Method

- **Preprocess**: strip all `//` comments and blank lines — comments describe
  identity, not design, and would let uploaders spoof the score. Pine only.
- **Chunk**: files are bigger than the model's context, so split into
  overlapping 4000-char chunks (400 overlap, line-aligned).
- **Embed**: one `ollama.embed` call per chunk batch, then **mean-pool** the
  chunk vectors and L2-normalize to a unit vector per script.
- **Center**: all Pine embeddings share a large common direction ("it's Pine
  code"), flooring raw cosine around ~0.6 even for unrelated scripts. Each DB
  stores a frozen `reference_mean` (corpus mean); subtracting it and
  renormalizing leaves only what distinguishes scripts — unrelated pairs land
  near 0. Thresholds are calibrated for the *centered* score.
- **Thresholds** (0.6b, preprocess v2, calibrated Oct 2026 on 26 scripts):
  `>= 0.80` remix/derivative → reward split; `>= 0.50` same design family.
  Indicator vs its own strategy copy scores 0.82–0.86; unrelated median ~-0.06.

### Findings

- **Raw cosine is not usable alone** — two *different recipes* score 0.45,
  recipe-vs-Pine-script 0.12; centering is what separates "same family" from
  "same language".
- **Merge-sweep** (`08_rsi_divergence.pine` × `macd_ema.pine`, see
  `merge_sweep_*.jpg`): detection depends heavily on *how* code is merged —
  - **concat** (B appended as a block): centered sim to A drops below the
    remix threshold at just ~5% foreign content; B's own chunks get equal
    weight in mean-pooling, diluting the A signal fast.
  - **interleave** (B's lines sprinkled through A): stays "REMIX of A" up to
    ~50–55% foreign content — every chunk still looks mostly like A.
  - **dead zone**: ~55–80% merges sit near 0.5 vs *both* parents — undetectable
    by whole-vector cosine either way.
  - transition to "REMIX of B" only happens around ~95% B content.
- **Implication for lineage detection**: a remixer bolting a foreign block
  onto a copied script *evades* the whole-vector flag; line-by-line laundering
  gets *caught*. Whole-vector centered cosine should therefore be paired with
  chunk-level max-similarity (the heatmap in `compare_two_scripts.ipynb`),
  which still exposes the copied half of a block merge.
- Scores are **not comparable across models** — each model has its own DB and
  frozen reference mean; `4b` is ~5–6× slower than `0.6b`.


