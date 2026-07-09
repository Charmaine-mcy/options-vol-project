"""
Black-Scholes pricing for European options, implemented from scratch.

Model assumptions (worth being able to recite in an interview):
  - The underlying follows geometric Brownian motion with constant vol sigma.
  - Constant risk-free rate r and continuous dividend yield q.
  - No transaction costs, continuous trading, European exercise only.

Under these assumptions the arbitrage-free price of a European call is

    C = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)

    d1 = [ln(S/K) + (r - q + sigma^2 / 2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

where N(.) is the standard normal CDF. The put price follows from the same
formula with N(-d2)/N(-d1) (equivalently, from put-call parity).

Intuition for the two terms of the call:
  - K * exp(-r*T) * N(d2)  : PV of paying the strike, times the risk-neutral
                             probability of finishing in the money.
  - S * exp(-q*T) * N(d1)  : expected PV of receiving the stock *conditional*
                             on exercise (N(d1) > N(d2) because the stock is
                             worth more in the states where you exercise).

Only numpy/scipy are used; scipy supplies the normal CDF/PDF, nothing else.
All functions are vectorized: scalars or numpy arrays both work.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """The two standardized log-moneyness terms at the heart of the model."""
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Black-Scholes price of a European option.

    Parameters
    ----------
    S : spot price of the underlying
    K : strike price
    T : time to expiry in years (e.g. 0.25 for 3 months)
    r : continuously-compounded risk-free rate (e.g. 0.05)
    sigma : annualized volatility (e.g. 0.20)
    q : continuous dividend yield, default 0
    option_type : "call" or "put"
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    # At (or past) expiry the option is worth exactly its intrinsic value;
    # handle it explicitly because d1/d2 divide by sqrt(T).
    T = np.asarray(T, dtype=float)
    if np.any(T <= 0) or np.any(np.asarray(sigma) <= 0):
        intrinsic = np.maximum(np.asarray(S) - np.asarray(K), 0.0) \
            if option_type == "call" else np.maximum(np.asarray(K) - np.asarray(S), 0.0)
        if np.all(T <= 0):
            return intrinsic if intrinsic.ndim else float(intrinsic)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_S = np.asarray(S) * np.exp(-q * T)   # dividend-discounted spot
    disc_K = np.asarray(K) * np.exp(-r * T)   # PV of the strike

    if option_type == "call":
        price = disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    else:
        price = disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
    return price if np.ndim(price) else float(price)


# ---------------------------------------------------------------------------
# Greeks (closed form). Vega is the one the IV solver needs.
# ---------------------------------------------------------------------------

def delta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """dPrice/dSpot. Call delta in (0,1), put delta in (-1,0)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        out = np.exp(-q * T) * norm.cdf(d1)
    else:
        out = -np.exp(-q * T) * norm.cdf(-d1)
    return out if np.ndim(out) else float(out)


def gamma(S, K, T, r, sigma, q=0.0):
    """d2Price/dSpot2 — same for calls and puts. Peaks near the money."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    out = np.exp(-q * T) * norm.pdf(d1) / (np.asarray(S) * sigma * np.sqrt(T))
    return out if np.ndim(out) else float(out)


def vega(S, K, T, r, sigma, q=0.0):
    """
    dPrice/dSigma — same for calls and puts, always positive.
    Quoted here per unit of vol (i.e. per 100 vol points); divide by 100
    for the per-1%-vol convention used on desks.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    out = np.asarray(S) * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return out if np.ndim(out) else float(out)


def theta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    dPrice/dTime, per year (divide by 365 for per-calendar-day theta).
    Almost always negative for long options: time decay.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    S, K, T = map(np.asarray, (S, K, T))
    # Decay of the optionality itself (shared by calls and puts):
    term_decay = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if option_type == "call":
        out = (term_decay
               - r * K * np.exp(-r * T) * norm.cdf(d2)
               + q * S * np.exp(-q * T) * norm.cdf(d1))
    else:
        out = (term_decay
               + r * K * np.exp(-r * T) * norm.cdf(-d2)
               - q * S * np.exp(-q * T) * norm.cdf(-d1))
    return out if np.ndim(out) else float(out)
