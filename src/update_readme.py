"""
Auto-update the data-dependent sections of README.md.

Sections are delimited by HTML-comment markers:

    <!-- AUTO:NAME:START -->
    ...generated text...
    <!-- AUTO:NAME:END -->

Only the text between markers is replaced; everything else in the README is
left byte-for-byte untouched. Each section's renderer contains light
conditional logic keyed to what the data actually shows (skew steep vs flat
vs inverted, contango vs backwardation, earnings in-window vs archived), so
the prose stays *accurate* over time rather than merely current — a flat
smile gets flagged as atypical instead of being described as the standard
skew with new numbers pasted in.

Failure policy: if a section's stats can't be computed, that section is left
exactly as it was and the failure is reported — a data problem can never
corrupt the hand-written parts of the README. The file is only rewritten when
something actually changed (idempotent: same data in, zero diff out).

Run standalone:  python -m src.update_readme
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_fetch import load_snapshots, filter_liquid, NY
from src.implied_vol import solve_chain
from src.plotting import atm_iv_by_expiry, pick_smile_expiry
from src.earnings_vol import (earnings_dates_around_now, earnings_numbers,
                              load_earnings_archive,
                              EARNINGS_LOOKAHEAD_DAYS, EARNINGS_GRACE_DAYS)

README = Path(__file__).resolve().parent.parent / "README.md"

# Wording thresholds (vol points, as decimals). A 10%-OTM put more than 2 vol
# points over ATM reads as the standard equity skew; under 1 point is "flat".
SKEW_STEEP, SKEW_FLAT = 0.02, 0.01
TERM_FLAT = 0.003          # |short - long| under 0.3 vol pts -> "flat"
PARITY_GAP_WARN = 0.015    # ATM call/put IV gap that stops being ignorable


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(ticker):
    """Everything the renderers need, from the latest snapshot for `ticker`."""
    combined = load_snapshots(ticker, verbose=False)
    latest_time = combined.index.max()
    latest = combined[combined.index == latest_time].reset_index(drop=True)
    spot = float(latest["spot"].iloc[0])
    stale = bool(latest["stale_quotes"].iloc[0])

    liquid, fdiag = filter_liquid(latest, stale, verbose=False)
    solved, sdiag = solve_chain(liquid, spot, latest["r"].iloc[0],
                                latest["q"].iloc[0], verbose=False)

    # ATM IV per expiry (call/put averaged) and constant-maturity tenors
    atm = atm_iv_by_expiry(solved, spot)
    atm_avg = (atm.groupby("expiry")
                  .agg(T_days=("T_days", "first"), atm_iv=("atm_iv", "mean"))
                  .reset_index().sort_values("T_days"))
    short_iv = float(atm_avg[atm_avg["T_days"] <= 7]["atm_iv"].mean())
    long_iv = float(atm_avg[atm_avg["T_days"] >= 20]["atm_iv"].mean())

    # smile metrics on the ~30d expiry, OTM side of each wing
    exp = pick_smile_expiry(solved)
    day = solved[(solved["expiry"] == exp) & solved["iv"].notna()].copy()
    day["m"] = day["strike"] / spot

    def wing_iv(m):
        """Interpolated OTM-wing IV at moneyness m (puts below 1, calls above)."""
        side = day[day["option_type"] == ("put" if m <= 1 else "call")]
        side = side[(side["m"] <= 1) if m <= 1 else (side["m"] >= 1)].sort_values("m")
        if len(side) >= 3 and side["m"].min() <= m <= side["m"].max():
            return float(np.interp(m, side["m"], side["iv"]))
        return None

    exp_atm = atm[atm["expiry"] == exp]
    atm_by_type = exp_atm.set_index("option_type")["atm_iv"].to_dict()
    iv_atm = float(exp_atm["atm_iv"].mean()) if len(exp_atm) else None
    iv_put90, iv_call110 = wing_iv(0.90), wing_iv(1.10)

    # where the failed solves live
    failed = solved[solved["iv_method"] == "failed"]
    intrinsic = np.where(failed["option_type"] == "call",
                         np.maximum(spot - failed["strike"], 0.0),
                         np.maximum(failed["strike"] - spot, 0.0))
    n_below_intrinsic = int((failed["price"] < intrinsic).sum())
    frac_failed_long = (float((failed["T"] * 365 > 90).mean())
                        if len(failed) else 0.0)

    return {
        "ticker": ticker, "spot": spot, "stale": stale,
        "latest_time": latest_time, "n_expiries": int(latest["expiry"].nunique()),
        "fdiag": fdiag, "sdiag": sdiag,
        "q": float(latest["q"].iloc[0]),
        "smile_expiry": exp, "iv_atm": iv_atm, "iv_put90": iv_put90,
        "iv_call110": iv_call110, "atm_by_type": atm_by_type,
        "short_iv": short_iv, "long_iv": long_iv, "atm_avg": atm_avg,
        "solved": solved,
        "n_failed_below_intrinsic": n_below_intrinsic,
        "frac_failed_long_dated": frac_failed_long,
    }


# ---------------------------------------------------------------------------
# Renderers — one per AUTO section. `S` maps ticker -> compute_stats() dict.
# ---------------------------------------------------------------------------

def render_results_table(S):
    s = S["SPY"]
    f, d = s["fdiag"], s["sdiag"]
    total, kept = f["total"], f["kept"]
    solvable = max(d["total"], 1)
    lines = [
        f"## Results (latest snapshot: {s['latest_time']:%Y-%m-%d %H:%M} ET, "
        f"SPY spot {s['spot']:.2f})",
        "",
        "| Stage | Count |",
        "|---|---|",
        f"| Contracts pulled ({s['n_expiries']} expiries) | {total:,} |",
        f"| Survive liquidity filter | {kept:,} ({kept / total:.0%}) |",
        f"| IV solved via Newton | {d['newton']:,} ({d['newton'] / solvable:.0%}) |",
        f"| IV solved via Brent fallback | {d['brent']:,} ({d['brent'] / solvable:.0%}) |",
        f"| Failed (no root / outside bounds) | {d['failed']:,} ({d['failed'] / solvable:.1%}) |",
        "",
    ]
    if d["brent"] > d["newton"]:
        lines.append(
            "Brent carries most of the load in this snapshot — typical when "
            "short-dated expiries or noisy wing quotes dominate (vega shrinks "
            "with sqrt(T), so the Newton step explodes where the price barely "
            "responds to vol).")
    else:
        lines.append(
            "Newton dominates, as expected when quotes are healthy and vega is "
            "meaningful across the chain; Brent mops up the deep wings and the "
            "shortest expiries.")
    where = ("deep ITM/OTM wings of the longer-dated expiries"
             if s["frac_failed_long_dated"] > 0.5
             else "the shortest expiries and deep-ITM strikes")
    lines.append(
        f"The {d['failed']} failures cluster in {where}; "
        f"{s['n_failed_below_intrinsic']} of them are quotes sitting below "
        "intrinsic value — prices that genuinely admit no implied vol, which "
        "the solver refuses rather than forcing a number.")
    return "\n".join(lines)


def render_smile(S):
    s = S["SPY"]
    if s["iv_atm"] is None or s["iv_put90"] is None:
        raise ValueError("smile wing IVs unavailable")
    skew = s["iv_put90"] - s["iv_atm"]
    exp = s["smile_expiry"]
    numbers = (f"the {exp:%d %b %Y} expiry prices {s['iv_atm']:.0%} at the "
               f"money vs {s['iv_put90']:.0%} for puts 10% below spot "
               f"({skew * 100:+.1f} vol pts)")

    head = ["### Why the smile exists", "", "![smile](outputs/smile_otm_SPY.png)", ""]
    explanations = [
        "",
        "- **Crash risk is priced.** Since October 1987, index option markets "
        "have never priced equity returns as lognormal — the true return "
        "distribution has a fat left tail, and OTM puts are priced accordingly.",
        "- **The leverage effect.** When equity prices fall, leverage (D/E) "
        "mechanically rises and realized volatility goes up — so low-strike "
        "states genuinely are higher-vol states.",
        "- **Demand for downside protection.** Institutions systematically buy "
        "index puts as insurance; dealers who sell them charge for the "
        "inventory risk.",
    ]
    if skew >= SKEW_STEEP:
        body = [
            "Under Black-Scholes assumptions the IV-vs-strike line would be "
            f"flat. Instead SPY shows the classic **equity skew**: {numbers}. "
            "Three standard explanations, all pushing the same direction:"
        ] + explanations
    elif skew >= SKEW_FLAT:
        body = [
            f"The skew is currently **milder than the textbook picture** — "
            f"{numbers}. The direction is still the standard equity skew, "
            "just compressed; the usual drivers apply with the volume "
            "turned down:"
        ] + explanations
    elif skew > -SKEW_FLAT:
        body = [
            f"The smile is currently **essentially flat** — {numbers}. That is "
            "atypical for an equity index, so treat it as a finding to "
            "investigate rather than a fact to explain: check whether the OTM "
            "put quotes are real (spread widths, volume) before concluding the "
            "market has genuinely stopped charging for crash risk. The "
            "long-run structural drivers of the skew (priced crash risk, the "
            "leverage effect, insurance demand) have not gone away."]
    else:
        body = [
            f"The smile is currently **inverted** — {numbers}, i.e. OTM puts "
            "priced *below* ATM. For an equity index this is a data-quality "
            "red flag first and a market signal second: verify the put wing "
            "quotes (staleness, crossed markets) before reading anything into "
            "it."]

    tail = []
    if s["iv_call110"] is not None:
        wing = s["iv_call110"] - s["iv_atm"]
        if wing > 0.005:
            tail = ["", f"The upturn in the far call wing ({s['iv_call110']:.0%} "
                    "at 10% above spot) completes the \"smirk\" — lottery-ticket "
                    "demand for upside calls, plus the fact that a big enough "
                    "rally is also a volatility event."]
        else:
            tail = ["", f"The far call wing is currently flat-to-soft "
                    f"({s['iv_call110']:.0%} at 10% above spot) — no meaningful "
                    "lottery-ticket bid for upside calls in this snapshot."]
    return "\n".join(head + body + tail)


def render_calls_puts(S):
    s = S["SPY"]
    by_type = s["atm_by_type"]
    if "call" not in by_type or "put" not in by_type:
        raise ValueError("need both call and put ATM IVs")
    c, p = by_type["call"], by_type["put"]
    gap = abs(c - p)
    exp = s["smile_expiry"]
    noise_src = ("staleness in last-trade prices (overnight snapshot)"
                 if s["stale"] else "quote noise and wide ITM spreads")

    head = ["### Calls vs puts — where they diverge and why", "",
            "![full smile](outputs/smile_SPY.png)", ""]
    if gap < PARITY_GAP_WARN:
        opener = (
            "In theory (European options, put-call parity) call IV and put IV "
            "must be identical at the same strike, and where both are **OTM** "
            f"the data agrees: at the money on the {exp:%d %b %Y} expiry, "
            f"calls solve to {c:.1%} and puts to {p:.1%} — a "
            f"{gap * 100:.1f}-vol-point gap, parity doing its job. Where "
            "options are **ITM**, the two series diverge sharply:")
    else:
        opener = (
            "Put-call parity says call IV and put IV should be identical at "
            f"the same strike — but in this snapshot they differ by "
            f"{gap * 100:.1f} vol points even at the money ({c:.1%} calls vs "
            f"{p:.1%} puts on the {exp:%d %b %Y} expiry), larger than parity "
            f"should allow. The likely culprit is {noise_src}; treat "
            "single-strike IV readings from this snapshot with caution. The "
            "structural ITM divergence is on top of that:")
    bullets = [
        "",
        "- An ITM option's price is nearly all intrinsic value; the vol "
        "information lives in a few cents of time value, and "
        f"{noise_src} can swing ITM implied vols enormously. (OTM prices are "
        "*pure* time value — much more informative per cent of noise.)",
        "- SPY options are **American**-exercise while our model is European. "
        "The early-exercise premium is small but nonzero (larger for ITM "
        "puts, and around ex-dividend dates for calls), biasing ITM IVs up "
        "slightly.",
        "",
        "This is why desks build vol surfaces from OTM puts below spot and OTM "
        "calls above spot — exactly what the OTM chart above does.",
    ]
    return "\n".join(head + [opener] + bullets)


def render_term_structure(S):
    s = S["SPY"]
    short, long_ = s["short_iv"], s["long_iv"]
    head = ["### Term structure", "",
            "![term structure](outputs/term_structure_SPY.png)", ""]
    nums = (f"ATM IV currently averages {short:.1%} for expiries within a week "
            f"vs {long_:.1%} around one month")
    if long_ - short > TERM_FLAT:
        shape = [
            f"{nums}: an **upward-sloping (contango)** curve, the normal "
            "calm-market shape — near-term realized vol is expected to stay "
            "low, while longer horizons carry a risk premium for the things "
            "that haven't happened yet."]
    elif short - long_ > TERM_FLAT:
        shape = [
            f"{nums}: a **downward-sloping (backwardation)** curve — the "
            "market is pricing *more* volatility near-term than long-term. "
            "That is the stressed shape: either a known near-dated event "
            "(macro print, earnings cluster) or a recent vol spike that the "
            "market expects to mean-revert. Worth checking the calendar "
            "before reading it as generalized fear."]
    else:
        shape = [
            f"{nums}: an **essentially flat** curve — neither calm-market "
            "contango nor stress backwardation. Flat term structures are "
            "transitional; watch whether the short end lifts (event risk "
            "approaching) or the long end reasserts the usual premium."]
    quirks = [
        "",
        "Short-end readings deserve suspicion in general: 1-2 day expiries "
        "are the noisiest numbers on the chart (tiny vega), and T is measured "
        "in **calendar** days, so an expiry spanning a weekend contains dead "
        "non-trading time that mechanically depresses its annualized IV. A "
        "trading-day clock would smooth this.",
    ]
    return "\n".join(head + shape + quirks)


def render_earnings(S):
    s = S["NFLX"]
    prev_e, next_e = earnings_dates_around_now("NFLX")
    archive = load_earnings_archive("NFLX")
    today = pd.Timestamp.now(tz=NY).normalize()
    head = ["### Stretch: earnings vol premium (NFLX)", "",
            "![earnings](outputs/earnings_NFLX.png)", ""]

    days_to_next = ((next_e.tz_convert(NY).normalize() - today).days
                    if next_e is not None else None)
    days_since_prev = ((today - prev_e.tz_convert(NY).normalize()).days
                       if prev_e is not None else None)

    if days_to_next is not None and days_to_next <= EARNINGS_LOOKAHEAD_DAYS:
        # PRE-EVENT: live premium from the current chain
        _, _, _, num = earnings_numbers(s["solved"], s["spot"],
                                        next_e.tz_convert(NY))
        body = [
            f"NFLX reports {next_e:%d %b %Y} ({days_to_next} days away) — "
            "the event is being priced *cross-sectionally* right now: the "
            f"first post-earnings expiry ({num['post_expiry']:%d %b}) carries "
            f"{num['post_iv']:.0%} ATM IV while later expiries decay back "
            "toward baseline as the one-day jump is diluted over more "
            "calendar time."]
        if num["implied_move"] is not None:
            body.append("")
            body.append(
                f"Backing out the event variance gives a **market-implied "
                f"earnings-day move of {num['implied_move']:.1%}** of spot "
                f"(method: {num['move_method']}). After the print, the front "
                "expiry's IV should collapse onto the baseline — the \"vol "
                "crush\" — which the pipeline captures automatically in its "
                "post-event grace window.")
    elif days_since_prev is not None and days_since_prev <= EARNINGS_GRACE_DAYS:
        # POST-EVENT GRACE: crush framing
        body = [
            f"NFLX reported {prev_e:%d %b %Y}, {days_since_prev} day(s) ago — "
            "this is the **vol crush** window. The front expiry's IV is "
            "collapsing back onto the long-dated baseline now that the event "
            "variance has been realized."]
        if archive and archive.get("captures", {}).get("pre"):
            pre_cap = archive["captures"]["pre"]
            front_now = float(s["atm_avg"]["atm_iv"].iloc[0])
            body.append("")
            body.append(
                f"Final pre-print front-expiry IV was "
                f"{pre_cap['curve'][0]['atm_iv']:.0%} (implied move "
                f"{pre_cap['implied_move']:.1%}); the front expiry now solves "
                f"to {front_now:.0%}.")
    elif archive and archive.get("captures"):
        # OUT OF WINDOW: archived framing
        event = pd.Timestamp(archive["event"])
        pre_c = archive["captures"].get("pre")
        post_c = archive["captures"].get("post")
        nxt = (f"the next report is {next_e:%d %b %Y} (~{days_to_next} days "
               "away); tracking resumes automatically inside the 60-day window"
               if next_e is not None else
               "the next report date is not yet published; tracking resumes "
               "automatically once it enters the 60-day window")
        body = [
            f"No earnings inside the tracking window at the moment — {nxt}. "
            f"The chart shows the **archived analysis of the {event:%d %b %Y} "
            "event**, rendered from numbers persisted at capture time "
            "(`data/earnings_archive_NFLX.json`), not recomputed:"]
        facts = []
        if pre_c:
            facts.append(
                f"- Final pre-print capture ({pd.Timestamp(pre_c['captured_at']):%d %b %H:%M} ET): "
                f"front-expiry ATM IV {pre_c['curve'][0]['atm_iv']:.0%} "
                f"(annualized — an entire earnings move packed into one day), "
                f"implied move {pre_c['implied_move']:.1%}.")
        if post_c:
            facts.append(
                f"- Post-print capture ({pd.Timestamp(post_c['captured_at']):%d %b %H:%M} ET): "
                f"front expiry crushed to "
                f"{post_c['curve'][0]['atm_iv']:.0%}.")
        if pre_c and post_c:
            realized = post_c["spot"] / pre_c["spot"] - 1.0
            facts.append(
                f"- Realized next-day move **{realized:+.1%}** vs "
                f"{pre_c['implied_move']:.1%} implied — the market "
                f"{'overpaid for' if abs(realized) < pre_c['implied_move'] else 'underpriced'} "
                "the event, which is the whole vol-crush trade in one line.")
        body += [""] + facts
    else:
        body = [
            "No earnings event is inside the tracking window and no completed "
            "analysis has been archived yet — the first in-window pipeline "
            "run will populate this section and the chart automatically."]
    return "\n".join(head + body)


def render_data_quality(S):
    s = S["SPY"]
    f, d = s["fdiag"], s["sdiag"]
    total = f["total"]
    bullets = ["## Data-quality caveats (observed, not hypothetical)", ""]
    if s["stale"]:
        vol_key = next((k for k in f if k.startswith("volume<")), None)
        bullets.append(
            "- **This snapshot was pulled outside market hours.** Yahoo zeroes "
            "bid/ask overnight, so prices are prior-session lastPrice with a "
            f"volume filter ({f.get(vol_key, 0):,} contracts dropped, "
            f"{f.get(vol_key, 0) / total:.0%}); midpoint data from market "
            "hours is strictly better.")
        bullets.append(
            "- **yfinance's own `impliedVolatility` column is garbage on "
            "overnight pulls** (~1e-5 across the chain when observed) — a "
            "good illustration of why this project solves for IV itself.")
    else:
        n_spread = next((v for k, v in f.items() if k.startswith("spread>")), 0)
        bullets.append(
            "- **This snapshot uses live bid/ask midpoints** (market-hours "
            f"pull): {f.get('zero_bid', 0):,} zero-bid contracts dropped and "
            f"{n_spread:,} more for spreads wider than 20% of mid.")
        bullets.append(
            "- **yfinance's own `impliedVolatility` column** has been observed "
            "returning garbage (~1e-5) on overnight pulls — one reason this "
            "project solves for IV itself rather than trusting vendor fields.")
    where = ("deep ITM/OTM wings of longer-dated expiries"
             if s["frac_failed_long_dated"] > 0.5
             else "the shortest expiries and deep-ITM strikes")
    bullets += [
        f"- **{d['failed'] / max(d['total'], 1):.1%} of IV solves failed** "
        f"({d['failed']:,} contracts, mostly {where}); "
        f"{s['n_failed_below_intrinsic']:,} were prices below intrinsic value "
        "— quotes that genuinely admit no implied vol.",
        "- **American vs European**: SPY/NFLX options are American; all IVs "
        "here carry a small upward bias from the unmodeled early-exercise "
        "premium.",
        f"- **Discrete dividends** are approximated by a continuous yield "
        f"(SPY q ~ {s['q']:.2%} trailing); fine at this horizon, cruder for "
        "long-dated options.",
    ]
    return "\n".join(bullets)


SECTIONS = {
    "RESULTS_TABLE": render_results_table,
    "SMILE": render_smile,
    "CALLS_PUTS": render_calls_puts,
    "TERM_STRUCTURE": render_term_structure,
    "EARNINGS": render_earnings,
    "DATA_QUALITY": render_data_quality,
}


# ---------------------------------------------------------------------------
# Marker replacement
# ---------------------------------------------------------------------------

def _replace_section(text, name, body):
    """Swap the text between NAME's markers; returns (new_text, changed)."""
    start, end = f"<!-- AUTO:{name}:START -->", f"<!-- AUTO:{name}:END -->"
    pattern = re.compile(re.escape(start) + r"\n(.*?)\n" + re.escape(end), re.S)
    m = pattern.search(text)
    if m is None:
        raise ValueError(f"markers for {name} not found in README")
    new_block = f"{start}\n{body.strip()}\n{end}"
    return text[:m.start()] + new_block + text[m.end():], m.group(0) != new_block


def update_readme(readme_path=README, tickers=("SPY", "NFLX")):
    """
    Re-render every AUTO section from live stats. Sections whose stats fail
    are left untouched. Returns (outcomes dict, file_changed bool); writes
    the file only if something changed.
    """
    text = original = readme_path.read_text()
    stats = {}
    for t in tickers:
        try:
            stats[t] = compute_stats(t)
        except Exception as exc:
            stats[t] = exc          # renderers needing it will fail cleanly

    outcomes = {}
    for name, renderer in SECTIONS.items():
        try:
            body = renderer(stats)   # raises if its ticker's stats are an Exception
            text, changed = _replace_section(text, name, body)
            outcomes[name] = "updated" if changed else "unchanged"
        except Exception as exc:
            outcomes[name] = f"left untouched ({type(exc).__name__}: {exc})"

    file_changed = text != original
    if file_changed:
        tmp = readme_path.with_name(readme_path.name + ".tmp")
        tmp.write_text(text)
        tmp.replace(readme_path)
    return outcomes, file_changed


if __name__ == "__main__":
    outcomes, changed = update_readme()
    for name, result in outcomes.items():
        print(f"  {name:<15} {result}")
    print(f"\nREADME.md {'rewritten' if changed else 'unchanged'}")
