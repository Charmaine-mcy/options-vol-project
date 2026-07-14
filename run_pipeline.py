#!/usr/bin/env python3
"""
Set-and-forget pipeline: one run = new data snapshot + refreshed charts.

    .venv/bin/python run_pipeline.py [--tickers SPY NFLX] [--force]

Per ticker, in order:
  1. snapshot   — live pull saved to data/snapshots/ (timestamped, immutable;
                  see data_fetch.save_snapshot). Skipped when the US market
                  is closed unless --force, so holiday/overnight runs don't
                  pile up duplicate stale-quote snapshots.
  2. IV solve   — implied vols recomputed on the latest snapshot.
  3. accumulating chart — atm_history_{TICKER}.png rebuilt from the ENTIRE
                  snapshot archive, so each run appends one point to the
                  constant-maturity ATM vol time series. Snapshots are never
                  modified; only the rendered chart is rewritten.
  4. replacing charts — smile_{TICKER}.png, smile_otm_{TICKER}.png,
                  term_structure_{TICKER}.png, and (if earnings are upcoming)
                  earnings_{TICKER}.png, regenerated from the latest snapshot
                  only and OVERWRITTEN in place — always exactly one current
                  version, no dated file pileup. The smile expiry is re-picked
                  each run as whichever listed expiry is nearest 30 days out.

Every stage is individually wrapped: a failure is logged with a traceback to
logs/pipeline.log and the run moves on to the next stage/ticker. Charts still
refresh from the existing archive even if today's pull fails.
"""

import argparse
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_fetch import save_snapshot, load_snapshots, filter_liquid
from src.implied_vol import solve_chain
from src.plotting import (plot_smile, plot_term_structure, pick_smile_expiry,
                          atm_iv_history, plot_atm_history)
from src.earnings_vol import next_earnings_date, earnings_curve_chart
from src.scheduler import market_is_open

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "logs" / "pipeline.log"
EARNINGS_LOOKAHEAD_DAYS = 60   # draw the earnings chart if a report is this close


def log(msg, level="INFO"):
    """Append one timestamped line to logs/pipeline.log and echo to stdout."""
    LOG_FILE.parent.mkdir(exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {level:<5} {msg}"
    with open(LOG_FILE, "a") as fh:
        fh.write(line + "\n")
    print(line)


def stage(name, fn, *args, **kwargs):
    """
    Run one pipeline stage. On success returns its result; on any exception
    logs the traceback and returns None so the rest of the run continues.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        log(f"FAILED {name}\n{traceback.format_exc().rstrip()}", level="ERROR")
        return None


def run_ticker(ticker, force=False):
    """Full snapshot -> solve -> charts sequence for one ticker."""
    outcomes = {}

    # -- 1. new snapshot -----------------------------------------------------
    if market_is_open() or force:
        snap = stage(f"{ticker} snapshot", save_snapshot, ticker)
        outcomes["snapshot"] = "ok" if snap else "FAILED"
    else:
        log(f"{ticker}: market closed — no new snapshot this run "
            f"(charts will still refresh from the existing archive)")
        outcomes["snapshot"] = "skipped(closed)"

    # -- 2. load archive + solve IVs on the latest snapshot -------------------
    combined = stage(f"{ticker} load_snapshots",
                     load_snapshots, ticker, verbose=False)
    if combined is None:
        log(f"{ticker}: no snapshots in archive — nothing to chart", "ERROR")
        outcomes["run"] = "aborted"
        return outcomes

    latest_time = combined.index.max()
    latest = combined[combined.index == latest_time].reset_index(drop=True)
    spot = float(latest["spot"].iloc[0])
    stale = bool(latest["stale_quotes"].iloc[0])
    r, q = float(latest["r"].iloc[0]), float(latest["q"].iloc[0])

    def solve_latest():
        liquid, fdiag = filter_liquid(latest, stale, verbose=False)
        solved, sdiag = solve_chain(liquid, spot, r, q, verbose=False)
        return liquid, solved, fdiag, sdiag

    res = stage(f"{ticker} IV solve", solve_latest)
    solved = None
    if res is not None:
        _, solved, fdiag, sdiag = res
        conv = (sdiag["newton"] + sdiag["brent"]) / max(sdiag["total"], 1)
        outcomes["iv_solve"] = f"{conv:.1%} converged"
        log(f"{ticker}: latest snapshot {latest_time:%Y-%m-%d %H:%M %Z} | "
            f"spot {spot:.2f} | {fdiag['total']} contracts, {fdiag['kept']} liquid | "
            f"IV {conv:.1%} converged ({sdiag['newton']} newton / "
            f"{sdiag['brent']} brent / {sdiag['failed']} failed) | "
            f"{'STALE lastPrice quotes' if stale else 'live midpoints'}")
    else:
        outcomes["iv_solve"] = "FAILED"

    # -- 3. accumulating chart (whole archive -> one more point) -------------
    def build_history():
        hist = atm_iv_history(combined, tenors=(7, 30))
        return plot_atm_history(hist, out_name=f"atm_history_{ticker}.png",
                                ticker=ticker), len(hist)
    acc = stage(f"{ticker} atm_history chart", build_history)
    if acc is not None:
        path, n_points = acc
        outcomes["atm_history"] = f"ok ({n_points} points)"
        log(f"{ticker}: accumulating chart {path.name} now spans {n_points} snapshot(s)")
    else:
        outcomes["atm_history"] = "FAILED"

    # -- 4. replacing charts (latest snapshot only, stable filenames) --------
    if solved is not None:
        note = "lastPrice data (overnight pull)" if stale else "bid/ask midpoints"

        def smiles():
            exp = pick_smile_expiry(solved)          # re-picked every run
            p1 = plot_smile(solved, spot, exp, f"smile_{ticker}.png",
                            note=note, ticker=ticker)
            p2 = plot_smile(solved, spot, exp, f"smile_otm_{ticker}.png",
                            note="OTM options only", otm_only=True, ticker=ticker)
            return exp, p1, p2
        sm = stage(f"{ticker} smile charts", smiles)
        if sm is not None:
            exp, p1, p2 = sm
            outcomes["smile"] = f"ok (expiry {exp:%Y-%m-%d})"
            log(f"{ticker}: smile charts rewritten for expiry {exp:%Y-%m-%d} "
                f"(nearest 30d at run time)")
        else:
            outcomes["smile"] = "FAILED"

        ts = stage(f"{ticker} term structure chart", plot_term_structure,
                   solved, spot, out_name=f"term_structure_{ticker}.png",
                   note=note, ticker=ticker)
        outcomes["term_structure"] = "ok" if ts is not None else "FAILED"

        earnings = stage(f"{ticker} earnings date lookup", next_earnings_date, ticker)
        if earnings is not None:
            days_out = (earnings - pd.Timestamp.now(tz=earnings.tz)).total_seconds() / 86400
            if 0 <= days_out <= EARNINGS_LOOKAHEAD_DAYS:
                s = stage(f"{ticker} earnings chart", earnings_curve_chart,
                          solved, spot, ticker, earnings)
                if s is not None:
                    outcomes["earnings"] = "ok"
                    msg = f"{ticker}: earnings chart rewritten (report {earnings:%Y-%m-%d}"
                    if s["implied_move"] is not None:
                        msg += (f", implied move {s['implied_move']:.1%} "
                                f"via {s['move_method']}")
                    log(msg + ")")
                else:
                    outcomes["earnings"] = "FAILED"
            else:
                outcomes["earnings"] = f"skipped ({days_out:.0f}d away)"
        else:
            outcomes["earnings"] = "skipped (no date found)"
    else:
        log(f"{ticker}: skipping replacing charts — no solved IVs this run", "WARN")

    return outcomes


def main():
    ap = argparse.ArgumentParser(description="snapshot + chart refresh pipeline")
    ap.add_argument("--tickers", nargs="+", default=["SPY", "NFLX"])
    ap.add_argument("--force", action="store_true",
                    help="pull a snapshot even when the US market is closed")
    args = ap.parse_args()

    log(f"=== pipeline run start (tickers: {', '.join(args.tickers)}"
        f"{', forced' if args.force else ''}) ===")
    for ticker in args.tickers:
        outcomes = stage(f"{ticker} run", run_ticker, ticker.upper(), args.force)
        summary = ", ".join(f"{k}={v}" for k, v in (outcomes or {}).items()) \
            or "run crashed before any stage completed"
        log(f"{ticker.upper()} summary: {summary}")
    log("=== pipeline run end ===")


if __name__ == "__main__":
    main()
