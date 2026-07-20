"""
Implied vol around an earnings date.  python -m src.earnings_vol [--ticker NFLX]

With a single data snapshot you can't watch vol crush happen in the time
series (that needs a pull before AND after the event), but you can see the
same economics cross-sectionally in the term structure:

  - Expiries BEFORE earnings only cover ordinary trading days -> lower IV.
  - The FIRST expiry AFTER earnings contains the announcement jump -> its
    IV is visibly elevated ("the earnings premium").
  - Later expiries dilute that one-day jump over more calendar time, so
    IV falls back toward the long-run level. After the announcement, the
    front expiry's IV collapses onto that baseline — the "vol crush".

Bonus metric interviewers like: the market-implied earnings-day move. Total
implied variance to an expiry is sigma^2 * T. If T2 is the first expiry after
the event and T1 the last one before, the variance ATTRIBUTABLE to the event is
approximately  sigma2^2*T2 - sigma1^2*T1 , so the implied one-day move is

    move ~ sqrt( sigma2^2 * T2 - sigma1^2 * T1 )

(as a fraction of spot; this treats non-event variance as flat across the gap).
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from src.data_fetch import (fetch_option_chain, filter_liquid,
                            get_risk_free_rate, get_dividend_yield, DATA_DIR)
from src.implied_vol import solve_chain
from src.plotting import (atm_iv_by_expiry, _new_axes, OUT_DIR,
                          CALL_COLOR, PUT_COLOR, MUTED, INK_2, BASELINE, SURFACE)


def earnings_dates_around_now(ticker):
    """(most recent past earnings, next upcoming earnings) — either may be
    None; (None, None) for tickers without earnings (ETFs like SPY)."""
    ed = yf.Ticker(ticker).earnings_dates
    if ed is None or ed.empty:
        return None, None
    now = pd.Timestamp.now(tz=ed.index.tz)
    past, future = ed.index[ed.index <= now], ed.index[ed.index > now]
    return (past.max() if len(past) else None,
            future.min() if len(future) else None)


def next_earnings_date(ticker):
    return earnings_dates_around_now(ticker)[1]


# ---------------------------------------------------------------------------
# Earnings analysis archive (data/earnings_archive_{ticker}.json)
#
# Each in-window pipeline run persists its computed numbers here — the ATM
# curve, implied move, spot — under a "pre" or "post" capture for the event
# being tracked. When the event passes out of the tracking window, the
# out-of-window chart re-renders from this archive instead of recomputing,
# so the last completed analysis survives indefinitely. A new event resets
# the captures.
# ---------------------------------------------------------------------------

def _archive_path(ticker):
    return DATA_DIR / f"earnings_archive_{ticker}.json"


def load_earnings_archive(ticker):
    p = _archive_path(ticker)
    return json.loads(p.read_text()) if p.exists() else None


def save_earnings_capture(ticker, event, phase, summary, spot, captured_at):
    """
    Record one in-window analysis in the archive. `phase` is 'pre' or 'post'
    (relative to the event); re-captures of the same phase overwrite so the
    archive ends up holding the LAST pre-event and LAST post-event analyses.
    Written atomically (tmp + os.replace), like the snapshot files.
    """
    arch = load_earnings_archive(ticker) or {}
    event_iso = event.isoformat()
    if arch.get("event") != event_iso:          # new event -> fresh archive
        arch = {"ticker": ticker, "event": event_iso, "captures": {}}
    curve = summary["atm_curve"]
    arch["captures"][phase] = {
        "captured_at": captured_at.isoformat(),
        "spot": float(spot),
        "implied_move": summary["implied_move"],
        "move_method": summary["move_method"],
        "curve": [{"expiry": e.strftime("%Y-%m-%d"),
                   "T_days": float(t), "atm_iv": float(v)}
                  for e, t, v in zip(curve["expiry"], curve["T_days"],
                                     curve["atm_iv"])],
    }
    p = _archive_path(ticker)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(arch, indent=2))
    os.replace(tmp, p)
    return arch


def render_earnings_status(ticker, next_earnings, archive, out_name=None,
                           display_horizon_days=120):
    """
    The out-of-window earnings chart: a clear "no upcoming earnings in the
    tracking window" state plus the archived last-completed analysis (final
    pre-earnings curve vs post-earnings crush, overlaid by expiry date),
    clearly labeled as historical. Overwrites earnings_{ticker}.png so the
    file never just goes quiet.
    """
    if next_earnings is not None:
        days = (next_earnings - pd.Timestamp.now(tz=next_earnings.tz)).total_seconds() / 86400
        state = f"next report {next_earnings:%d %b %Y} (~{days:.0f} days away)"
    else:
        state = "next report date unknown"

    subtitle = f"{state} · tracking resumes automatically inside the window"
    has_archive = bool(archive and archive.get("captures"))
    if has_archive:
        event = pd.Timestamp(archive["event"])
        pre, post = (archive["captures"].get(p) for p in ("pre", "post"))
        banner = f"ARCHIVED — last completed analysis: earnings {event:%d %b %Y}"
        if pre and pre.get("implied_move") is not None:
            banner += f" · implied move {pre['implied_move']:.1%}"
        if pre and post:
            realized = post["spot"] / pre["spot"] - 1.0
            banner += f" · realized move {realized:+.1%}"
        subtitle += "\n" + banner

    fig, ax = _new_axes(
        f"{ticker} earnings vol — no upcoming report in tracking window",
        subtitle)

    if has_archive:
        for phase, color, label in (("pre", CALL_COLOR, "final pre-earnings"),
                                    ("post", PUT_COLOR, "post-earnings (crush)")):
            cap = archive["captures"].get(phase)
            if not cap:
                continue
            cv = pd.DataFrame(cap["curve"])
            cv["expiry"] = pd.to_datetime(cv["expiry"])
            cv = cv[cv["expiry"] <= event.tz_localize(None) +
                    pd.Timedelta(days=display_horizon_days)]
            captured = pd.Timestamp(cap["captured_at"])
            ax.plot(cv["expiry"], cv["atm_iv"] * 100, color=color, lw=2,
                    marker="o", ms=4.5,
                    label=f"{label} · captured {captured:%d %b %H:%M}")
        ax.axvline(event.tz_localize(None), color=BASELINE, lw=1.2, ls="--")
        y_lo, y_hi = ax.get_ylim()
        ax.annotate(f"earnings\n{event:%b %d}",
                    (event.tz_localize(None), y_lo + 0.05 * (y_hi - y_lo)),
                    xytext=(6, 0), textcoords="offset points",
                    color=MUTED, fontsize=8.5)
        ax.set_xlabel("Expiry date", color=INK_2, fontsize=10)
        ax.set_ylabel("ATM implied volatility (%)", color=INK_2, fontsize=10)
        ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="upper right")
        fig.autofmt_xdate(rotation=0, ha="center")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No archived earnings analysis yet.\n"
                "The first in-window run will populate this chart.",
                transform=ax.transAxes, ha="center", va="center",
                color=MUTED, fontsize=11)

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / (out_name or f"earnings_{ticker}.png")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def earnings_curve_chart(solved, spot, ticker, earnings, out_name=None):
    """
    Draw the ATM term structure with the earnings date marked, from an
    already-solved chain. Overwrites outputs/earnings_{ticker}.png (stable
    filename — one current version). Returns a summary dict with the
    pre/post-earnings ATM IVs and the implied earnings-day move (fields are
    None when an expiry on that side of the event isn't available).
    """
    atm = atm_iv_by_expiry(solved, spot)
    if atm.empty:
        raise ValueError(f"{ticker}: no ATM IVs available to chart")
    # average call/put ATM IV per expiry (they should agree; averaging cancels
    # some lastPrice noise)
    atm_avg = (atm.groupby("expiry")
                  .agg(T_days=("T_days", "first"), atm_iv=("atm_iv", "mean"))
                  .reset_index().sort_values("T_days"))
    atm_avg["expiry"] = pd.to_datetime(atm_avg["expiry"])

    earnings_naive = earnings.tz_localize(None)
    pre = atm_avg[atm_avg["expiry"] < earnings_naive].tail(1)
    post = atm_avg[atm_avg["expiry"] >= earnings_naive].head(1)

    fig, ax = _new_axes(
        f"{ticker} ATM implied vol around earnings",
        f"earnings {earnings:%d %b %Y} · spot {spot:.2f} · "
        f"ATM IV = call/put average, interpolated at spot")
    ax.plot(atm_avg["T_days"], atm_avg["atm_iv"] * 100, color=CALL_COLOR,
            lw=2, marker="o", ms=5)
    days_to_earnings = (earnings_naive - pd.Timestamp.now()).total_seconds() / 86400
    ax.axvline(days_to_earnings, color=BASELINE, lw=1.2, ls="--")
    y_lo, y_hi = ax.get_ylim()
    ax.annotate(f"earnings\n{earnings:%b %d}",
                (days_to_earnings, y_lo + 0.06 * (y_hi - y_lo)),
                xytext=(6, 0), textcoords="offset points",
                color=MUTED, fontsize=8.5)
    if len(post):
        x, y = post["T_days"].iloc[0], post["atm_iv"].iloc[0] * 100
        ax.annotate(f"first post-earnings expiry: {y:.1f}%", (x, y),
                    xytext=(16, -6), textcoords="offset points",
                    color=INK_2, fontsize=8.5, va="top",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("Days to expiry", color=INK_2, fontsize=10)
    ax.set_ylabel("ATM implied volatility (%)", color=INK_2, fontsize=10)
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / (out_name or f"earnings_{ticker}.png")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    summary = {"path": path, "earnings": earnings, "atm_curve": atm_avg,
               "pre_expiry": None, "pre_iv": None,
               "post_expiry": None, "post_iv": None,
               "implied_move": None, "move_method": None}
    if len(pre):
        summary["pre_expiry"] = pre["expiry"].iloc[0]
        summary["pre_iv"] = float(pre["atm_iv"].iloc[0])
    if len(post):
        summary["post_expiry"] = post["expiry"].iloc[0]
        summary["post_iv"] = float(post["atm_iv"].iloc[0])

    if len(pre) and len(post):
        s1, T1 = summary["pre_iv"], pre["T_days"].iloc[0] / 365
        s2, T2 = summary["post_iv"], post["T_days"].iloc[0] / 365
        summary["implied_move"] = float(np.sqrt(max(s2**2 * T2 - s1**2 * T1, 0.0)))
        summary["move_method"] = "pre/post expiry variance difference"
    elif len(post):
        # In the final days before a report the last pre-earnings expiry has
        # often already expired. Same economics, different decomposition:
        # the post-earnings expiry's total variance is the one-day event
        # variance plus ordinary diffusion at the long-run vol, so strip a
        # long-dated baseline out of it:  move ~ sqrt((s2^2 - sb^2) * T2).
        base = atm_avg[atm_avg["T_days"] >= 45]
        if len(base):
            sb = float(base["atm_iv"].mean())
            s2, T2 = summary["post_iv"], post["T_days"].iloc[0] / 365
            summary["implied_move"] = float(np.sqrt(max((s2**2 - sb**2) * T2, 0.0)))
            summary["move_method"] = (f"post-earnings expiry vs long-dated "
                                      f"baseline ({sb:.0%})")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="NFLX")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    t = args.ticker.upper()

    earnings = next_earnings_date(t)
    if earnings is None:
        print(f"  no upcoming earnings date found for {t}; nothing to plot")
        return
    print(f"  {t} next earnings: {earnings:%Y-%m-%d %H:%M %Z}")

    df, spot, stale = fetch_option_chain(t, max_expiries=10, force_refresh=args.refresh)
    r, q = get_risk_free_rate(), get_dividend_yield(t, spot)
    liquid, _ = filter_liquid(df, stale)
    solved, diag = solve_chain(liquid, spot, r, q)

    s = earnings_curve_chart(solved, spot, t, earnings)
    print(f"  wrote {s['path']}")
    show = s["atm_curve"].copy()
    show["atm_iv"] = (show["atm_iv"] * 100).round(2)
    print("\n  ATM IV by expiry:")
    print(show.to_string(index=False))

    if s["implied_move"] is not None:
        if s["pre_expiry"] is not None:
            print(f"\n  last pre-earnings expiry  {s['pre_expiry']:%Y-%m-%d}: "
                  f"ATM IV {s['pre_iv']:.1%}")
        print(f"  first post-earnings expiry {s['post_expiry']:%Y-%m-%d}: "
              f"ATM IV {s['post_iv']:.1%}")
        print(f"  => market-implied earnings-day move ~ {s['implied_move']:.1%} of spot "
              f"(~${s['implied_move'] * spot:.0f} on {t}) [{s['move_method']}]")
        print("  After the print, the post-earnings expiry's IV should collapse "
              "toward the pre-earnings level — re-run this script the day after "
              "to capture the crush with a second snapshot.")


if __name__ == "__main__":
    main()
