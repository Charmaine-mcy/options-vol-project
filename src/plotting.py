"""
Charts: volatility smile for one expiry, and ATM term structure.

Style notes: one y-axis per chart (never dual), thin 2px lines, direct series
labels plus a legend, recessive grid. Calls are always blue, puts always red,
regardless of which chart they appear on — color follows the entity.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render to file; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# Reference palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CALL_COLOR = "#2a78d6"   # blue
PUT_COLOR = "#e34948"    # red


def _new_axes(title, subtitle):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.suptitle(title, x=0.065, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK_2, pad=12)
    return fig, ax


def plot_smile(solved, spot, expiry, out_name="smile.png",
               moneyness_range=(0.80, 1.20), note=None, otm_only=False,
               ticker="SPY"):
    """
    Implied vol vs moneyness (K/spot) for a single expiry, calls and puts as
    separate series so any divergence is visible rather than averaged away.

    otm_only=True keeps puts below spot and calls above (desk convention):
    ITM prices are almost pure intrinsic value, so tiny staleness in the
    quote swings their implied vol wildly — the OTM side of each wing is
    where the vol information actually lives.
    """
    exp = pd.Timestamp(expiry)
    day = solved[(solved["expiry"] == exp) & solved["iv"].notna()].copy()
    day["moneyness"] = day["strike"] / spot
    day = day[day["moneyness"].between(*moneyness_range)]
    if otm_only:
        day = day[((day["option_type"] == "put") & (day["strike"] <= spot)) |
                  ((day["option_type"] == "call") & (day["strike"] >= spot))]
    T_days = day["T"].iloc[0] * 365

    fig, ax = _new_axes(
        f"{ticker} implied volatility smile — {exp:%d %b %Y} expiry",
        f"{T_days:.0f} days to expiry · spot {spot:.2f} · "
        f"{len(day)} liquid contracts" + (f" · {note}" if note else ""))

    for typ, color, label in (("call", CALL_COLOR, "Calls"),
                              ("put", PUT_COLOR, "Puts")):
        leg = day[day["option_type"] == typ].sort_values("strike")
        ax.plot(leg["moneyness"], leg["iv"] * 100, color=color, lw=2,
                marker="o", ms=3.5, label=label)
        # direct label at the right end of each series
        if len(leg):
            ax.annotate(label, (leg["moneyness"].iloc[-1], leg["iv"].iloc[-1] * 100),
                        xytext=(8, 0), textcoords="offset points",
                        color=color, fontsize=9, fontweight="bold", va="center")

    ax.axvline(1.0, color=BASELINE, lw=1, ls="--")
    ax.annotate("ATM", (1.0, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", color=MUTED, fontsize=8)
    ax.set_xlabel("Moneyness  K / S", color=INK_2, fontsize=10)
    ax.set_ylabel("Implied volatility (%)", color=INK_2, fontsize=10)
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="upper right")

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / out_name
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def pick_smile_expiry(solved, target_days=30):
    """
    Expiry nearest `target_days` out that has a healthy number of solved IVs.
    Recomputed at every run — as calendar time passes, 'the ~30-day expiry'
    rolls forward automatically instead of drifting like a hardcoded date.
    """
    ok = solved.dropna(subset=["iv"])
    per_exp = ok.groupby("expiry").agg(n=("iv", "size"), T=("T", "first"))
    per_exp = per_exp[per_exp["n"] >= 40]          # need breadth for a smile
    return (per_exp["T"] * 365 - target_days).abs().idxmin()


def atm_iv_by_expiry(solved, spot):
    """
    Interpolate implied vol at moneyness exactly 1.0 for each expiry and
    option type. Requires strikes with valid IVs on both sides of spot;
    expiries that can't bracket spot are skipped (and reported by caller).
    """
    rows = []
    for (exp, typ), g in solved.dropna(subset=["iv"]).groupby(["expiry", "option_type"]):
        g = g.sort_values("strike")
        if g["strike"].min() < spot < g["strike"].max() and len(g) >= 3:
            iv_atm = float(np.interp(spot, g["strike"], g["iv"]))
            rows.append({"expiry": exp, "option_type": typ,
                         "T_days": g["T"].iloc[0] * 365, "atm_iv": iv_atm})
    return pd.DataFrame(rows)


def plot_term_structure(solved, spot, out_name="term_structure.png", note=None,
                        ticker="SPY"):
    """ATM implied vol vs days to expiry, one line per option type."""
    atm = atm_iv_by_expiry(solved, spot)

    fig, ax = _new_axes(
        f"{ticker} ATM implied volatility term structure",
        f"interpolated at K = spot ({spot:.2f}) · "
        f"{atm['expiry'].nunique()} expiries" + (f" · {note}" if note else ""))

    for typ, color, label in (("call", CALL_COLOR, "Calls"),
                              ("put", PUT_COLOR, "Puts")):
        leg = atm[atm["option_type"] == typ].sort_values("T_days")
        ax.plot(leg["T_days"], leg["atm_iv"] * 100, color=color, lw=2,
                marker="o", ms=4.5, label=label)
        if len(leg):
            ax.annotate(label, (leg["T_days"].iloc[-1], leg["atm_iv"].iloc[-1] * 100),
                        xytext=(8, 0), textcoords="offset points",
                        color=color, fontsize=9, fontweight="bold", va="center")

    ax.set_xlabel("Days to expiry", color=INK_2, fontsize=10)
    ax.set_ylabel("ATM implied volatility (%)", color=INK_2, fontsize=10)
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="lower right")

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / out_name
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path, atm


# ---------------------------------------------------------------------------
# ATM implied vol across snapshots (time series)
# ---------------------------------------------------------------------------

TENOR_COLORS = {7: "#2a78d6", 30: "#1baf7a", 60: "#eda100"}  # categorical slots 1-3


def atm_iv_history(combined, tenors=(7, 30)):
    """
    From a load_snapshots() DataFrame, build a constant-maturity ATM IV time
    series: one row per snapshot, one column per tenor (days).

    Per snapshot: liquidity-filter, solve IV only for strikes within 5% of
    that snapshot's spot (the ATM neighborhood — cheap and all we need),
    interpolate call and put IV at K = spot per expiry, average them, then
    interpolate across expiries at the fixed tenors. Constant-maturity
    interpolation is what makes snapshots comparable over days: 'the 30-day
    vol' doesn't jump when the front expiry rolls off.
    """
    from src.data_fetch import filter_liquid
    from src.implied_vol import solve_chain

    rows = []
    for pull_time, block in combined.groupby(level=0):
        block = block.reset_index(drop=True)
        spot = block["spot"].iloc[0]
        stale = bool(block["stale_quotes"].iloc[0])
        near = block[(block["strike"] / spot).between(0.95, 1.05)]
        liquid, _ = filter_liquid(near, stale, verbose=False)
        solved, _ = solve_chain(liquid, spot, block["r"].iloc[0],
                                block["q"].iloc[0], verbose=False)
        atm = atm_iv_by_expiry(solved, spot)
        if atm.empty:
            continue
        curve = (atm.groupby("expiry")
                    .agg(T_days=("T_days", "first"), atm_iv=("atm_iv", "mean"))
                    .sort_values("T_days"))
        row = {"pull_time": pull_time, "spot": spot, "stale_quotes": stale}
        for tenor in tenors:
            if curve["T_days"].min() <= tenor <= curve["T_days"].max():
                row[f"{tenor}d"] = float(np.interp(tenor, curve["T_days"],
                                                   curve["atm_iv"]))
            else:                       # tenor outside the quoted expiries
                row[f"{tenor}d"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("pull_time").sort_index()


def plot_atm_history(history, out_name="atm_history.png", ticker="SPY"):
    """Constant-maturity ATM implied vol over snapshot time, one line per tenor."""
    tenor_cols = [c for c in history.columns if c.endswith("d") and c != "stale_quotes"]
    n_stale = int(history["stale_quotes"].sum())
    fig, ax = _new_axes(
        f"{ticker} ATM implied vol across snapshots",
        f"{len(history)} snapshots · {history.index.min():%d %b %H:%M} -> "
        f"{history.index.max():%d %b %H:%M} NY · constant-maturity, "
        f"interpolated at K = spot"
        + (f" · {n_stale} snapshot(s) from stale/overnight quotes" if n_stale else ""))

    for col in tenor_cols:
        tenor = int(col[:-1])
        color = TENOR_COLORS.get(tenor, "#4a3aa7")
        ax.plot(history.index, history[col] * 100, color=color, lw=2,
                marker="o", ms=4.5, label=f"{tenor}-day")
        ax.annotate(f"{tenor}-day", (history.index[-1], history[col].iloc[-1] * 100),
                    xytext=(8, 0), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold", va="center")

    ax.set_xlabel("Snapshot time (NY)", color=INK_2, fontsize=10)
    ax.set_ylabel("ATM implied volatility (%)", color=INK_2, fontsize=10)
    if len(tenor_cols) > 1:
        ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="best")
    fig.autofmt_xdate(rotation=0, ha="center")

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / out_name
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path
