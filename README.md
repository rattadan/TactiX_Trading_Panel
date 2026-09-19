# PineTS Strategies & Indicators

This folder contains the Pine Script strategies and indicators that power the
charting / backtesting engine in this app (loaded via `pages/api/simulate.ts`
and rendered in `components/dex/OpenChartsPro.tsx`).

## Adding your own indicator or strategy

1. **Drop a `.pine` file into this folder.** Any Pine Script v5/v6
   `indicator()` or `strategy()` script works — no registration step needed.
   The strategy picker in the UI scans this directory automatically.

2. **Naming convention.** Files are listed alphabetically. Prefix with a
   number (e.g. `13_my_indicator.pine`) if you want to control ordering.

3. **Supported features.**
   - `indicator()` / `strategy()` declarations, `plot*`, `plotcandle`,
     `plotshape`, `bgcolor`, `barcolor`
   - `input.*` calls (single- and multi-line) — these become editable
     parameters in the UI
   - `box`, `line`, and `label` drawings (see caveats below)
   - `request.security` / `request.security_lower_tf` for multi-timeframe data

4. **Known limitations of the PineTS transpiler.**
   - `import` lines (e.g. `import TradingView/ta/12`) are skipped — library
     functions are unavailable; inline the helpers you need.
   - `var` declarations inside `if` blocks and `var label x = na` are not
     supported.
   - `label.delete()` is permanent — reuse labels via `label.set_*` instead.
   - Labels are single-line only (`\n` is not rendered).
   - Some built-ins are missing (e.g. `timenow`,
     `strategy.opentrades.entry_bar_index`). If a script fails, check the
     console for the missing symbol.

5. **Testing your script.** Run the regression harness to verify it
   transpiles and executes:

   ```bash
   node tmp/test-all-strategies.mjs
   ```

   Or test a single file by pointing `PineTS` at it directly (see
   `tmp/test-cme-institutional.mjs` for a template).

## Creator rewards

We want indicator and strategy authors to earn rewards when traders use
their work. If you publish a script here, you can attach a payout address so
a share of usage / subscription fees flows back to you.

**How to opt in:**

1. Add a metadata header at the top of your `.pine` file:

   ```pine
   // @title: My Awesome Indicator
   // @author: YourName
   // @reward_address: inj1yourinjectiveaddresshere...
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
