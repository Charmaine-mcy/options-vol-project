# Options Pricing & Implied Volatility Analysis

A from-scratch implementation of Black-Scholes pricing and implied volatility
analysis on real market data. No shortcut IV libraries — the pricing formula,
Greeks, and the Newton-Raphson/Brent root-finder are all implemented and
tested in this repo.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.test_black_scholes   # unit tests vs textbook values
python -m src.main                 # full SPY pipeline -> outputs/*.png
python -m src.earnings_vol         # stretch: NFLX earnings vol premium
```

Or open `notebooks/analysis.ipynb` for the guided walkthrough.

```
run_pipeline.py        set-and-forget pipeline: snapshot + all charts, per ticker
src/black_scholes.py   Black-Scholes price + closed-form Greeks
src/implied_vol.py     Newton-Raphson IV solver with Brent fallback
src/data_fetch.py      yfinance option-chain pull, caching, liquidity filter,
                       timestamped snapshots (save_snapshot / load_snapshots)
src/scheduler.py       recurring snapshot puller (loop mode or cron one-shot)
src/plotting.py        smile, term-structure & snapshot-history charts
src/main.py            end-to-end pipeline
src/earnings_vol.py    earnings vol premium / implied move (stretch)
data/                  cached raw pulls + solved chains (CSV, one per day)
data/snapshots/        immutable timestamped pulls (CSV + JSON sidecar each)
outputs/               charts (PNG, stable filenames — see update policy below)
logs/pipeline.log      one line per pipeline stage: counts, convergence, failures
```

## Methodology

### 1. Pricing (Black-Scholes)

European call/put prices under geometric Brownian motion with constant vol,
rate `r` (13-week T-bill via `^IRX`, converted to continuous compounding) and
continuous dividend yield `q` (trailing-12-month dividends / spot). Closed-form
delta, gamma, vega, theta. Verified against textbook values (ATM 1y call at
S=K=100, r=5%, sigma=20% prices at 10.4506) and, more stringently, every Greek
is checked against a finite-difference derivative of the price function, and
put-call parity holds to machine precision.

### 2. Implied vol solver

`price -> sigma` has no closed form, but vega > 0 makes the map monotone, so a
unique root exists whenever the price is inside the no-arbitrage bounds.

1. **No-arbitrage pre-check** — a quote below intrinsic value (or above the
   discounted spot/strike) has *no* implied vol; return `None` immediately.
2. **Newton-Raphson** — `sigma <- sigma - (BS(sigma) - P) / vega(sigma)`,
   seeded with the Brenner-Subrahmanyam approximation `P/S * sqrt(2*pi/T)`.
   Quadratic convergence, typically 3-6 iterations.
3. **Brent fallback** — Newton fails where vega is tiny (deep ITM/OTM, or
   hours to expiry: the price barely responds to vol, so the Newton step
   explodes). Brent only needs a bracketing interval, so it is slow but
   nearly unbreakable. If even Brent cannot bracket a root in [0.01%, 500%],
   the contract is recorded as **failed** rather than given a garbage number.

Round-trip tests (price at a known sigma, invert, compare) recover vol to
~1e-15 in both the Newton and Brent regimes.

### 3. Data & liquidity filtering

- Underlying: **SPY**, 12 expiries (~1 day to ~1 month out at pull time).
- Market price: bid/ask **midpoint** during market hours. This snapshot was
  pulled overnight, when Yahoo clears its quotes (99% of contracts showed a
  zero bid) — the code detects this and falls back to **lastPrice** from the
  prior session, anchoring time-to-expiry to that session's 4pm ET close so
  short-dated IVs aren't biased by phantom hours.
- Liquidity filter: in midpoint mode, drop zero-bid contracts and spreads
  wider than 20% of mid; in lastPrice mode, require session volume >= 10.
- Raw pulls are cached to `data/` (one CSV per ticker per day) so runs are
  reproducible; the filter runs after loading and reports what it dropped.

## Results (pull: 2026-07-08 close, SPY spot 745.29)

| Stage | Count |
|---|---|
| Contracts pulled (12 expiries) | 2,920 |
| Survive liquidity filter | 1,788 (61%) |
| IV solved via Newton | 774 (43%) |
| IV solved via Brent fallback | 947 (53%) |
| Failed (no root / outside bounds) | 67 (3.7%) |

Brent's share is unusually high because every expiry here is short-dated
(vega shrinks with sqrt(T)); on a chain with 6-24 month expiries Newton would
dominate. The 67 failures cluster exactly where theory predicts: the two
shortest expiries and deep-ITM strikes, where a slightly stale last price sits
below intrinsic value, so no implied vol exists at all.

### Why the smile exists

![smile](outputs/smile_otm_SPY.png)

Under Black-Scholes assumptions the IV-vs-strike line would be flat. Instead
SPY shows the classic **equity skew**: 30-day IV rises from ~11% just above
spot to ~35% at 20% below spot. Three standard explanations, all of which
push the same direction:

- **Crash risk is priced.** Since October 1987, index option markets have
  never priced equity returns as lognormal — the true return distribution has
  a fat left tail, and OTM puts are priced accordingly.
- **The leverage effect.** When equity prices fall, leverage (D/E) mechanically
  rises and realized volatility goes up — so low-strike states genuinely are
  higher-vol states.
- **Demand for downside protection.** Institutions systematically buy index
  puts as insurance; dealers who sell them charge for the inventory risk.

The slight upturn in the far call wing (>10% OTM) completes the "smirk" —
lottery-ticket demand for upside calls plus the fact that a big enough rally
is also a volatility event.

### Calls vs puts — where they diverge and why

![full smile](outputs/smile_SPY.png)

In theory (European options, put-call parity) call IV and put IV must be
identical at the same strike. In this data they agree closely where each
option is **OTM**, and diverge wildly where **ITM**:

- An ITM option's price is nearly all intrinsic value; the vol information
  lives in a few cents of time value. Any staleness in the last trade swings
  the implied vol enormously. (OTM prices are *pure* time value — much more
  informative per cent of noise.)
- SPY options are **American**-exercise while our model is European. The
  early-exercise premium is small but nonzero (larger for ITM puts, and
  around ex-dividend dates for calls), biasing ITM IVs up slightly.

This is why desks build vol surfaces from OTM puts below spot and OTM calls
above spot — exactly what the OTM chart above does.

### Term structure

![term structure](outputs/term_structure_SPY.png)

ATM IV averages ~12.8% for expiries within a week vs ~14.0% around one month:
a gently **upward-sloping (contango)** curve, the normal calm-market shape —
near-term realized vol is expected to stay low, while longer horizons carry a
risk premium for the things that haven't happened yet. Two short-end quirks
worth knowing about:

- The 1-2 day expiries print *above* the rest of the short end. Very
  short-dated IVs are the noisiest numbers on the chart (tiny vega, overnight
  prices), and pin/event effects dominate there.
- The ~5-day expiry dips because T is measured in **calendar** days: an expiry
  that spans a weekend contains dead non-trading time, which mechanically
  depresses its annualized IV. A trading-day clock would smooth this.

### Stretch: earnings vol premium (NFLX)

![earnings](outputs/earnings_NFLX.png)

NFLX reports 16 Jul 2026 after the close. With a single snapshot you can see
the event priced *cross-sectionally*: the last pre-earnings expiry (Jul 10)
carries 39% ATM IV, the first post-earnings expiry (Jul 17) jumps to 67%,
and later expiries decay back toward the ~43% baseline as the one-day jump is
diluted over more calendar time. Backing out the event variance
(`sigma2^2*T2 - sigma1^2*T1`) gives a **market-implied earnings-day move of
~10%** of spot. After the print, the Jul 17 IV should collapse onto the
baseline — the "vol crush"; re-running `src.earnings_vol` the morning after
captures the before/after pair.

## Tracking the smile over time (snapshots)

Every call to `save_snapshot("SPY")` writes an **immutable** timestamped pair
to `data/snapshots/` — raw chain CSV plus a JSON sidecar with the spot, spot
timestamp, pull timestamp, risk-free rate, dividend yield, and a stale-quotes
flag, so each snapshot is self-contained and reproducible:

```
spy_chain_2026-07-09_0124.csv         (filename clock = New York time)
spy_chain_2026-07-09_0124_meta.json
```

Old snapshots are never overwritten: same-minute collisions get a seconds
suffix, and a true collision raises instead of clobbering.

**Collecting on a cadence** — the recommended way is `run_pipeline.py`, which
refreshes data *and* charts together (next section). For data-only collection
the older options remain: `python -m src.scheduler --loop --every 30` or a
cron line on `python -m src.scheduler --once` (market-hours guarded).

## Set-and-forget pipeline

`run_pipeline.py` (project root) ties collection and charting together.
Per ticker (default `SPY NFLX`): take a new immutable snapshot → recompute
implied vols on it → refresh all charts. Two chart update policies:

- **Accumulating** — `atm_history_{TICKER}.png` is rebuilt from the *entire*
  snapshot archive, so every run appends one point to the constant-maturity
  ATM vol time series. Snapshot files are never modified.
- **Replacing** — `smile_{TICKER}.png`, `smile_otm_{TICKER}.png`,
  `term_structure_{TICKER}.png`, `earnings_{TICKER}.png` (when a report is
  within 60 days) are regenerated from the latest snapshot only and
  overwritten in place: always exactly one current version, no dated pileup.
  The smile expiry is re-picked at run time (nearest listed expiry to 30 days
  out), so it rolls forward automatically as the calendar advances.

Every stage is individually wrapped: one ticker or chart failing is logged
(with traceback) to `logs/pipeline.log` and the rest of the run continues —
and charts still refresh from the archive even if the day's pull fails. Each
run logs contract counts, the liquidity-filter yield, IV convergence rate
(newton/brent/failed split), and stale-quote status.

Cron schedule (`crontab -e`) — US market open, midday, and near close on
weekdays. Times are in the machine's local clock (here: Hong Kong, UTC+8);
each slot has two entries so the schedule survives US daylight-saving
changes — the market-hours guard turns whichever twin lands outside
9:30–16:00 ET into a no-snapshot chart refresh, and the in-hours extras just
add snapshots:

```cron
# open (9:30 ET)        midday (12:30 ET)      near close (15:45 ET)
30 21 * * 1-5 cd "/path/to/options-vol-project" && .venv/bin/python run_pipeline.py >> logs/cron.log 2>&1
30 22 * * 1-5 cd "/path/to/options-vol-project" && .venv/bin/python run_pipeline.py >> logs/cron.log 2>&1
30 0  * * 2-6 cd "/path/to/options-vol-project" && .venv/bin/python run_pipeline.py >> logs/cron.log 2>&1
30 1  * * 2-6 cd "/path/to/options-vol-project" && .venv/bin/python run_pipeline.py >> logs/cron.log 2>&1
45 3  * * 2-6 cd "/path/to/options-vol-project" && .venv/bin/python run_pipeline.py >> logs/cron.log 2>&1
45 4  * * 2-6 cd "/path/to/options-vol-project" && .venv/bin/python run_pipeline.py >> logs/cron.log 2>&1
```

(Midday/close slots use `2-6` = Tue–Sat local because those ET times fall
after midnight Hong Kong time.) macOS caveats: cron doesn't fire while the
laptop sleeps (no catch-up), and macOS privacy protection can block cron from
reading `~/Desktop` — if `logs/cron.log` shows "Operation not permitted",
grant `/usr/sbin/cron` Full Disk Access in System Settings, or move the
project out of Desktop. US exchange holidays aren't modeled; a holiday run
just logs a skipped snapshot.

**Analysis** — `load_snapshots("SPY")` combines every snapshot into one
DataFrame indexed by pull time, and `atm_iv_history` + `plot_atm_history`
turn it into a **constant-maturity** ATM vol time series (IV interpolated at
K = spot per expiry, then across expiries at fixed 7- and 30-day tenors, so
the series doesn't jump when the front expiry rolls off — the same idea as
the VIX's 30-day interpolation):

![atm history](outputs/atm_history_SPY.png)

The chart above was seeded with three overnight snapshots minutes apart —
identical prior-session prices, hence flat lines (the subtitle flags them as
stale-quote snapshots). It becomes a real vol monitor as market-hours
snapshots accumulate: expect the 7-day line to swing harder than the 30-day
(short-dated vol is the twitchier end of the curve), and both to jump on
macro prints.

## Data-quality caveats (observed, not hypothetical)

- **Overnight pulls have no quotes.** Yahoo zeroes bid/ask outside market
  hours; 99% of SPY contracts had zero bid at pull time. The pipeline detects
  this and switches to lastPrice + volume filtering, but midpoint data from
  market hours is strictly better — re-run during US trading hours for it.
- **yfinance's own `impliedVolatility` column was garbage** at pull time
  (~1e-5 across the chain) — a good illustration of why this project solves
  for IV itself rather than trusting vendor fields.
- **1,132 contracts (39%) were dropped for volume < 10**, mostly far-OTM
  wings and deep-ITM strikes nobody trades.
- **3.7% of IV solves failed**, nearly all stale ITM lastPrices below
  intrinsic value — i.e., prices that genuinely admit no implied vol.
- **American vs European**: SPY/NFLX options are American; all IVs here carry
  a small upward bias from the unmodeled early-exercise premium.
- **Discrete dividends** are approximated by a continuous yield (SPY q ~ 1.25%
  trailing); fine at this horizon, cruder for long-dated options.

## Limitations / next steps

- Binomial-tree (CRR) pricer to quantify the American premium directly.
- Trading-day (business-day) time convention to remove the weekend artifact.
- Second earnings snapshot post-print to plot the realized vol crush.
- Fit a parametric smile (e.g. SVI) instead of linear interpolation at ATM.
