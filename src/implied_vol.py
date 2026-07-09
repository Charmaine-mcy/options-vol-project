"""
Numerical implied volatility solver: Newton-Raphson with a Brent fallback.

The problem: given an observed market price P, find the sigma such that
BS_price(S, K, T, r, sigma, q) = P. There is no closed form, but the map
sigma -> price is strictly increasing (vega > 0), so a unique root exists
whenever P sits inside the no-arbitrage bounds.

Method (the part to be able to explain in an interview):

  1. No-arbitrage pre-check. A call must be worth at least
     max(S*e^{-qT} - K*e^{-rT}, 0) and at most S*e^{-qT} (puts analogously).
     A price outside those bounds has NO implied vol — typically a stale or
     nonsense quote — so we return None immediately instead of letting the
     root-finder thrash.

  2. Newton-Raphson:  sigma_{n+1} = sigma_n - (BS(sigma_n) - P) / vega(sigma_n).
     Quadratic convergence (typically 3-6 iterations) because we have the
     exact analytic derivative. Seeded with the Brenner-Subrahmanyam
     approximation  sigma_0 ~ P / S * sqrt(2*pi / T), which is exact for an
     ATM-forward option and a decent start elsewhere.

  3. Brent (bisection-family) fallback via scipy.optimize.brentq on
     [1e-4, 5.0]. Newton fails when vega is tiny — deep ITM/OTM or very
     short-dated options, where the price barely responds to vol and the
     Newton step explodes. Brent only needs a sign change, so it is slow but
     nearly unbreakable; if even Brent can't bracket a root, we return None
     and count it. The failure rate is a data-quality diagnostic, not a bug.
"""

import numpy as np
from scipy.optimize import brentq

from src.black_scholes import bs_price, vega

SIGMA_MIN, SIGMA_MAX = 1e-4, 5.0   # 0.01% to 500% annualized vol


def _no_arb_bounds(S, K, T, r, q, option_type):
    """(lower, upper) arbitrage bounds for a European option price."""
    disc_S, disc_K = S * np.exp(-q * T), K * np.exp(-r * T)
    if option_type == "call":
        return max(disc_S - disc_K, 0.0), disc_S
    return max(disc_K - disc_S, 0.0), disc_K


def implied_vol(price, S, K, T, r, q=0.0, option_type="call",
                tol=1e-8, max_iter=50):
    """
    Invert Black-Scholes for sigma. Returns (iv, method) where method is
    'newton', 'brent', or None. iv is None if the price admits no implied
    vol or nothing converged.
    """
    if T <= 0 or price <= 0 or np.isnan(price):
        return None, None

    lo, hi = _no_arb_bounds(S, K, T, r, q, option_type)
    # A tiny tolerance around the bounds absorbs penny-rounding in quotes.
    if not (lo - 1e-10 < price < hi):
        return None, None

    # --- Newton-Raphson ----------------------------------------------------
    # Brenner-Subrahmanyam seed (exact for ATM-forward), clamped to range.
    sigma = np.clip(price / S * np.sqrt(2.0 * np.pi / T), 0.05, 2.0)
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, q, option_type) - price
        if abs(diff) < tol:
            return float(sigma), "newton"
        v = vega(S, K, T, r, sigma, q)
        if v < 1e-10:          # flat objective — Newton step would explode
            break
        step = diff / v
        sigma -= step
        if not (SIGMA_MIN < sigma < SIGMA_MAX):   # jumped out of range
            break

    # --- Brent fallback ----------------------------------------------------
    def objective(s):
        return bs_price(S, K, T, r, s, q, option_type) - price

    try:
        f_lo, f_hi = objective(SIGMA_MIN), objective(SIGMA_MAX)
        if f_lo * f_hi > 0:    # no sign change: root outside [1e-4, 5]
            return None, None
        iv = brentq(objective, SIGMA_MIN, SIGMA_MAX, xtol=1e-10, maxiter=200)
        return float(iv), "brent"
    except (ValueError, RuntimeError):
        return None, None


def solve_chain(df, spot, r, q=0.0, price_col="price", verbose=True):
    """
    Solve implied vol for every contract in a chain DataFrame (needs columns
    strike, T, option_type, and `price_col`). Adds 'iv' and 'iv_method'
    columns; returns (df, diagnostics dict) with the convergence breakdown.
    """
    ivs, methods = [], []
    for row in df.itertuples(index=False):
        iv, method = implied_vol(getattr(row, price_col), spot, row.strike,
                                 row.T, r, q, row.option_type)
        ivs.append(np.nan if iv is None else iv)
        methods.append(method or "failed")

    out = df.copy()
    out["iv"] = ivs
    out["iv_method"] = methods

    counts = out["iv_method"].value_counts().to_dict()
    diag = {"total": len(out),
            "newton": counts.get("newton", 0),
            "brent": counts.get("brent", 0),
            "failed": counts.get("failed", 0)}
    if verbose:
        print(f"  IV solver: {diag['total']} contracts -> "
              f"{diag['newton']} Newton ({diag['newton']/diag['total']:.1%}), "
              f"{diag['brent']} Brent fallback ({diag['brent']/diag['total']:.1%}), "
              f"{diag['failed']} failed ({diag['failed']/diag['total']:.1%})")
    return out, diag


if __name__ == "__main__":
    # Round-trip self-test: price an option at a known vol, then recover it.
    print("Round-trip tests (price at known sigma, then invert):")
    for S, K, T, r, q, sig, typ in [
        (100, 100, 1.0, 0.05, 0.00, 0.20, "call"),   # ATM, benign
        (100, 100, 1.0, 0.05, 0.00, 0.20, "put"),
        (100, 150, 0.5, 0.05, 0.00, 0.35, "call"),   # far OTM call
        (100, 60, 0.25, 0.05, 0.02, 0.45, "put"),    # far OTM put, with q
        (100, 100, 2/365, 0.05, 0.00, 0.18, "call"), # 2 days to expiry
        (100, 80, 0.02, 0.05, 0.00, 0.60, "put"),    # deep OTM + near expiry
    ]:
        p = bs_price(S, K, T, r, sig, q, typ)
        iv, method = implied_vol(p, S, K, T, r, q, typ)
        err = abs(iv - sig) if iv is not None else float("nan")
        print(f"  S={S} K={K} T={T:.4f} {typ:4s} true={sig:.4f} "
              f"price={p:.4f} -> iv={iv:.6f} via {method:6s} err={err:.2e}")

    print("\nBad-quote handling (should refuse, not return garbage):")
    for label, price in [("below intrinsic", 0.5), ("above spot", 120.0)]:
        iv, method = implied_vol(price, 100, 90, 0.5, 0.05, 0.0, "call")
        print(f"  call K=90 price={price} ({label}) -> iv={iv}, method={method}")
