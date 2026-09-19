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
