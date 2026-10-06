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


