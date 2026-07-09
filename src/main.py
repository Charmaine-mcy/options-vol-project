"""
End-to-end pipeline:  python -m src.main [--ticker SPY] [--refresh]

  1. Pull (or load cached) option chain, spot, risk-free rate, dividend yield.
  2. Filter to liquid contracts, reporting what was dropped and why.
  3. Solve implied vol for every contract (Newton -> Brent -> None).
  4. Save the solved chain to data/ and write charts to outputs/:
       - full smile (calls + puts) for the expiry nearest 30 days out
       - OTM-only smile for the same expiry
       - ATM term structure across all expiries
"""

import argparse
from datetime import datetime

import pandas as pd

from src.data_fetch import (fetch_option_chain, filter_liquid,
                            get_risk_free_rate, get_dividend_yield, DATA_DIR)
from src.implied_vol import solve_chain
from src.plotting import plot_smile, plot_term_structure, pick_smile_expiry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore today's cache and re-pull")
    args = ap.parse_args()
    t = args.ticker.upper()

    print(f"=== {t} implied volatility analysis ===\n")
    print("[1/4] market data")
    df, spot, stale = fetch_option_chain(t, force_refresh=args.refresh)
    r = get_risk_free_rate()
    q = get_dividend_yield(t, spot)
    print(f"  r = {r:.4%} (13w T-bill, cont. comp.), q = {q:.4%} (trailing 12m div yield)")

    print("\n[2/4] liquidity filter")
    liquid, _ = filter_liquid(df, stale)

    print("\n[3/4] implied vol solve")
    solved, diag = solve_chain(liquid, spot, r, q)
    out_csv = DATA_DIR / f"{t}_solved_{datetime.now():%Y-%m-%d}.csv"
    solved.to_csv(out_csv, index=False)
    print(f"  solved chain saved to {out_csv.name}")

    print("\n[4/4] charts")
    note = "lastPrice data (overnight pull)" if stale else "bid/ask midpoints"
    # Stable filenames (overwritten each run) so there is always exactly one
    # current version of each chart; the expiry shown is in the chart title.
    exp = pick_smile_expiry(solved)
    p1 = plot_smile(solved, spot, exp, f"smile_{t}.png", note=note, ticker=t)
    p2 = plot_smile(solved, spot, exp, f"smile_otm_{t}.png",
                    note="OTM options only", otm_only=True, ticker=t)
    p3, atm = plot_term_structure(solved, spot, out_name=f"term_structure_{t}.png",
                                  note=note, ticker=t)
    for p in (p1, p2, p3):
        print(f"  wrote {p}")

    atm_30 = atm[atm["T_days"] > 20]["atm_iv"].mean()
    atm_short = atm[atm["T_days"] <= 7]["atm_iv"].mean()
    slope = "upward" if atm_30 > atm_short else "downward"
    print(f"\nSummary: ATM vol {atm_short:.1%} (<=1wk) vs {atm_30:.1%} (~1mo) -> "
          f"{slope}-sloping term structure; "
          f"IV solve failure rate {diag['failed']/diag['total']:.1%}.")


if __name__ == "__main__":
    main()
