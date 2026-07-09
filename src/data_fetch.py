"""
Pull and cache real option chain data via yfinance.

Design choices worth defending in an interview:

  - "Market price": during market hours we use the bid/ask midpoint — last
    trades can be hours stale on illiquid strikes, while the mid reflects the
    current quoted market. BUT Yahoo zeroes out bid/ask quotes overnight, so
    a pull outside US market hours has no usable quotes. We detect that
    (majority of contracts with zero bid) and fall back to lastPrice from the
    prior session, with a volume-based liquidity filter instead of a
    spread-based one. The price source is recorded, never hidden.

  - Time to expiry must be measured from when the prices were SET, not when
    we pulled them. With stale (overnight) prices we anchor T to the prior
    session's 4pm ET close; otherwise 0-DTE options would look like they had
    extra hours of life and their implied vols would be biased low.

  - The raw pull is cached to data/ (one CSV per ticker per day) so re-runs
    are reproducible and don't hammer the API. Derived columns (mid, T) are
    recomputed on load; liquidity filtering happens after loading so we can
    report exactly what was discarded and why.

  - Risk-free rate: 13-week T-bill yield (^IRX), converted to a continuously-
    compounded rate. Falls back to a constant if the quote is unavailable.

  - yfinance's own impliedVolatility column is kept (renamed yf_iv) purely as
    a cross-check; overnight it is visibly garbage (~1e-5), which is exactly
    why this project solves for IV itself.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAP_DIR = DATA_DIR / "snapshots"
NY = "America/New_York"

FALLBACK_RISK_FREE = 0.04  # used only if the ^IRX quote can't be fetched


def get_risk_free_rate():
    """13-week T-bill yield from ^IRX, as a continuously-compounded decimal."""
    try:
        quote = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        simple = float(quote.iloc[-1]) / 100.0     # ^IRX quotes in percent
        return float(np.log(1.0 + simple))          # -> continuous compounding
    except Exception as exc:
        print(f"  [warn] could not fetch ^IRX ({exc}); using r={FALLBACK_RISK_FREE}")
        return FALLBACK_RISK_FREE


def get_dividend_yield(ticker, spot):
    """
    Trailing-12-month dividends / spot, as a continuous-yield approximation.
    SPY pays quarterly, so this is an approximation, but ignoring dividends
    entirely would bias call IVs down and put IVs up relative to each other.
    """
    try:
        divs = yf.Ticker(ticker).dividends
        last_12m = divs[divs.index >= divs.index.max() - pd.Timedelta(days=365)]
        return float(last_12m.sum()) / spot
    except Exception as exc:
        print(f"  [warn] could not fetch dividends ({exc}); using q=0")
        return 0.0


def get_spot(ticker):
    """(price, timestamp) of the most recent trade in the underlying."""
    hist = yf.Ticker(ticker).history(period="1d", interval="1m")["Close"].dropna()
    if hist.empty:  # no intraday data — fall back to daily closes
        hist = yf.Ticker(ticker).history(period="5d")["Close"].dropna()
    ts = hist.index[-1]
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY)
    return float(hist.iloc[-1]), ts.tz_convert(NY)


def _attach_derived(df, spot_time, verbose=True):
    """
    Add the columns downstream code uses: price (the market price we trust),
    price_source ('mid' or 'last'), and T (years to expiry).

    Detects the overnight/stale-quote case: if most contracts show a zero
    bid, Yahoo has cleared its quotes and only lastPrice is meaningful.
    """
    frac_zero_bid = (df["bid"] <= 0).mean()
    stale = frac_zero_bid > 0.5

    if stale:
        df["price"] = df["lastPrice"]
        df["price_source"] = "last"
        # Prices were set during the prior session -> anchor T to that
        # session's 4pm ET close (spot_time is the underlying's last trade).
        anchor = spot_time.normalize() + pd.Timedelta(hours=16)
        if verbose:
            print(f"  [data quality] {frac_zero_bid:.0%} of contracts have zero bid -> "
                  f"market is closed / quotes cleared. Using lastPrice from the "
                  f"session ending {anchor:%Y-%m-%d %H:%M %Z} as the market price.")
    else:
        df["price"] = (df["bid"] + df["ask"]) / 2.0
        df["price_source"] = "mid"
        anchor = spot_time

    df["mid"] = (df["bid"] + df["ask"]) / 2.0  # kept for reference either way
    expiry_close = df["expiry"].dt.tz_localize(NY) + pd.Timedelta(hours=16)
    df["T"] = (expiry_close - anchor).dt.total_seconds() / (365.0 * 24 * 3600)
    return df, stale


def _pull_raw_chain(ticker, max_expiries):
    """One live pull: (raw chain DataFrame, spot, spot_time). No caching."""
    tk = yf.Ticker(ticker)
    spot, spot_time = get_spot(ticker)
    expiries = tk.options[:max_expiries]
    print(f"  {ticker} spot = {spot:.2f} @ {spot_time:%Y-%m-%d %H:%M %Z}; "
          f"pulling {len(expiries)} expiries...")

    frames = []
    for exp in expiries:
        chain = tk.option_chain(exp)
        for opt_type, leg in (("call", chain.calls), ("put", chain.puts)):
            leg = leg.copy()
            leg["option_type"] = opt_type
            leg["expiry"] = exp
            frames.append(leg)

    df = pd.concat(frames, ignore_index=True)
    keep = ["contractSymbol", "expiry", "option_type", "strike", "bid", "ask",
            "lastPrice", "volume", "openInterest", "impliedVolatility"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["expiry"] = pd.to_datetime(df["expiry"])
    df = df.rename(columns={"impliedVolatility": "yf_iv"})
    return df, spot, spot_time


def fetch_option_chain(ticker="SPY", max_expiries=12, force_refresh=False):
    """
    Pull the option chain for `ticker` across up to `max_expiries` expiries.

    Returns (df, spot, stale) where stale=True means overnight lastPrice data.
    The raw chain is cached to data/{ticker}_chain_{YYYY-MM-DD}.csv and reused
    for same-day runs unless force_refresh=True.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    cache_file = DATA_DIR / f"{ticker}_chain_{today}.csv"
    meta_file = DATA_DIR / f"{ticker}_meta_{today}.csv"

    if cache_file.exists() and meta_file.exists() and not force_refresh:
        df = pd.read_csv(cache_file, parse_dates=["expiry"])
        meta = pd.read_csv(meta_file)
        spot = float(meta["spot"].iloc[0])
        spot_time = pd.Timestamp(meta["spot_time"].iloc[0]).tz_convert(NY)
        print(f"  loaded cached chain from {cache_file.name} "
              f"({len(df)} contracts, spot={spot:.2f} @ {spot_time:%Y-%m-%d %H:%M %Z})")
    else:
        df, spot, spot_time = _pull_raw_chain(ticker, max_expiries)
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(cache_file, index=False)
        pd.DataFrame({"spot": [spot], "spot_time": [spot_time]}).to_csv(meta_file, index=False)
        print(f"  cached {len(df)} contracts to {cache_file.name}")

    df, stale = _attach_derived(df, spot_time)
    return df, spot, stale


# ---------------------------------------------------------------------------
# Timestamped snapshots (for tracking the vol surface over time)
# ---------------------------------------------------------------------------

def save_snapshot(ticker="SPY", max_expiries=12):
    """
    One live pull saved to data/snapshots/ as an immutable, timestamped pair:

        {ticker}_chain_{YYYY-MM-DD_HHMM}.csv        raw chain
        {ticker}_chain_{YYYY-MM-DD_HHMM}_meta.json  spot, timestamps, r, q

    Timestamps in the filename are New York time (the market's clock, so
    filenames sort in session order regardless of where the puller runs).
    Existing files are NEVER overwritten: a same-minute collision gets a
    seconds suffix, and a full collision raises instead of clobbering.
    Returns the CSV path.
    """
    df, spot, spot_time = _pull_raw_chain(ticker, max_expiries)
    pull_time = pd.Timestamp.now(tz=NY)
    _, stale = _attach_derived(df.copy(), spot_time)  # detect, for the metadata

    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{ticker.lower()}_chain_{pull_time:%Y-%m-%d_%H%M}"
    if (SNAP_DIR / f"{stem}.csv").exists():          # same-minute pull
        stem = f"{ticker.lower()}_chain_{pull_time:%Y-%m-%d_%H%M%S}"
    csv_path = SNAP_DIR / f"{stem}.csv"
    meta_path = SNAP_DIR / f"{stem}_meta.json"
    if csv_path.exists() or meta_path.exists():
        raise FileExistsError(f"refusing to overwrite existing snapshot {csv_path.name}")

    meta = {
        "ticker": ticker.upper(),
        "spot": spot,
        "spot_time": spot_time.isoformat(),
        "pull_time": pull_time.isoformat(),
        "stale_quotes": bool(stale),
        "risk_free_rate": get_risk_free_rate(),
        "dividend_yield": get_dividend_yield(ticker, spot),
        "n_contracts": len(df),
    }
    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  snapshot saved: {csv_path.name} ({len(df)} contracts, "
          f"spot {spot:.2f}, stale_quotes={stale})")
    return csv_path


def load_snapshots(ticker="SPY", verbose=True):
    """
    Load every snapshot for `ticker` from data/snapshots/ into one DataFrame
    indexed by pull_time (NY timezone, sorted; one block of contract rows per
    pull). Derived columns (price, price_source, T) are recomputed per
    snapshot from its own sidecar metadata, and each snapshot's spot, r, q,
    and stale flag ride along as columns. Read-only: never touches the files.
    """
    pairs = sorted(SNAP_DIR.glob(f"{ticker.lower()}_chain_*.csv"))
    blocks = []
    for csv_path in pairs:
        meta_path = csv_path.with_name(csv_path.stem + "_meta.json")
        if not meta_path.exists():
            print(f"  [warn] {csv_path.name} has no meta sidecar; skipping")
            continue
        meta = json.loads(meta_path.read_text())
        df = pd.read_csv(csv_path, parse_dates=["expiry"])
        spot_time = pd.Timestamp(meta["spot_time"])
        df, stale = _attach_derived(df, spot_time, verbose=False)
        df["pull_time"] = pd.Timestamp(meta["pull_time"])
        df["spot"] = meta["spot"]
        df["r"] = meta["risk_free_rate"]
        df["q"] = meta["dividend_yield"]
        df["stale_quotes"] = stale
        blocks.append(df)

    if not blocks:
        raise FileNotFoundError(f"no snapshots for {ticker} in {SNAP_DIR}")
    out = pd.concat(blocks, ignore_index=True).set_index("pull_time").sort_index()
    if verbose:
        n = len(pairs)
        print(f"  loaded {n} snapshot(s) for {ticker}: "
              f"{out.index.min():%Y-%m-%d %H:%M} -> {out.index.max():%Y-%m-%d %H:%M} NY, "
              f"{len(out)} contract rows")
    return out


def filter_liquid(df, stale, max_rel_spread=0.20, min_volume=10, verbose=True):
    """
    Keep only contracts whose price is trustworthy, and report what was
    dropped — the discard pattern is itself diagnostically interesting.

    Live-quote mode: drop zero-bid contracts and those with bid-ask spread
    wider than `max_rel_spread` of the mid.
    Stale (overnight lastPrice) mode: spreads are meaningless, so instead
    require the contract actually traded in the session (volume >= min_volume)
    and has a positive last price.
    """
    n0 = len(df)
    if stale:
        no_trade = df["volume"].fillna(0) < min_volume
        no_price = df["price"].fillna(0) <= 0
        out = df[~no_trade & ~no_price].copy()
        diag = {"total": n0, "mode": "stale/lastPrice",
                f"volume<{min_volume}": int(no_trade.sum()),
                "no_last_price": int((no_price & ~no_trade).sum()),
                "kept": len(out)}
    else:
        zero_bid = df["bid"] <= 0
        no_ask = df["ask"] <= 0
        quoted = df[~zero_bid & ~no_ask].copy()
        rel_spread = (quoted["ask"] - quoted["bid"]) / quoted["mid"]
        wide = rel_spread > max_rel_spread
        out = quoted[~wide].copy()
        out["rel_spread"] = rel_spread[~wide]
        diag = {"total": n0, "mode": "live/mid",
                "zero_bid": int(zero_bid.sum()),
                "no_ask": int((no_ask & ~zero_bid).sum()),
                f"spread>{max_rel_spread:.0%}": int(wide.sum()),
                "kept": len(out)}
    if verbose:
        dropped = ", ".join(f"{v} {k}" for k, v in diag.items()
                            if k not in ("total", "mode", "kept"))
        print(f"  liquidity filter [{diag['mode']}]: {n0} -> kept {diag['kept']} "
              f"({diag['kept']/n0:.0%}); dropped: {dropped}")
    return out, diag


if __name__ == "__main__":
    df, spot, stale = fetch_option_chain("SPY")
    r = get_risk_free_rate()
    print(f"\n  risk-free rate (13w T-bill, cont. comp.): {r:.4%}")
    liquid, diag = filter_liquid(df, stale)

    print(f"\n  expiries pulled: {[str(d) for d in sorted(df['expiry'].dt.date.unique())]}")
    print("\n  liquid contracts nearest the money (first expiry):")
    first = liquid[liquid["expiry"] == liquid["expiry"].min()].copy()
    first["dist"] = (first["strike"] - spot).abs()
    near = first.nsmallest(10, "dist")
    cols = ["expiry", "option_type", "strike", "price", "price_source",
            "volume", "openInterest", "T"]
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(near.sort_values(["option_type", "strike"])[cols].to_string(index=False))
