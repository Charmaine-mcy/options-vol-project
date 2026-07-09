"""
Unit tests for black_scholes.py against known textbook values.

Reference case (Hull, "Options, Futures and Other Derivatives", ch. 15 style):
  S=100, K=100, T=1yr, r=5%, sigma=20%, q=0
  Call = 10.4506, Put = 5.5735  (standard values quoted to 4 dp)

Run with:  python -m src.test_black_scholes
"""

import numpy as np

from src.black_scholes import bs_price, delta, gamma, vega, theta

S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def check(label, got, expected, tol=1e-4):
    ok = abs(got - expected) < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<38} got {got:.6f}  expected {expected:.6f}")
    assert ok, f"{label}: {got} != {expected}"


def main():
    print("Textbook values (S=100, K=100, T=1, r=5%, sigma=20%):")
    check("ATM call price", bs_price(S, K, T, R, SIGMA, option_type="call"), 10.450584)
    check("ATM put price", bs_price(S, K, T, R, SIGMA, option_type="put"), 5.573526)

    # Hull ch.15 example: S=42, K=40, r=10%, sigma=20%, T=0.5
    print("\nHull example (S=42, K=40, T=0.5, r=10%, sigma=20%):")
    check("call", bs_price(42, 40, 0.5, 0.10, 0.20, option_type="call"), 4.759422, tol=1e-4)
    check("put", bs_price(42, 40, 0.5, 0.10, 0.20, option_type="put"), 0.808600, tol=1e-4)

    # Put-call parity: C - P = S*exp(-qT) - K*exp(-rT). This must hold to
    # machine precision because both prices come from the same d1/d2.
    print("\nPut-call parity (with dividend yield q=1.5%):")
    q = 0.015
    c = bs_price(S, K, T, R, SIGMA, q=q, option_type="call")
    p = bs_price(S, K, T, R, SIGMA, q=q, option_type="put")
    check("C - P vs S*e^-qT - K*e^-rT", c - p,
          S * np.exp(-q * T) - K * np.exp(-R * T), tol=1e-10)

    # Greeks vs central finite differences of the price function itself —
    # catches sign errors and misplaced terms that a single point value won't.
    print("\nGreeks vs finite differences:")
    h = 1e-4
    fd_delta = (bs_price(S + h, K, T, R, SIGMA) - bs_price(S - h, K, T, R, SIGMA)) / (2 * h)
    check("call delta", delta(S, K, T, R, SIGMA), fd_delta, tol=1e-6)

    fd_gamma = (bs_price(S + h, K, T, R, SIGMA) - 2 * bs_price(S, K, T, R, SIGMA)
                + bs_price(S - h, K, T, R, SIGMA)) / h**2
    check("gamma", gamma(S, K, T, R, SIGMA), fd_gamma, tol=1e-4)

    fd_vega = (bs_price(S, K, T, R, SIGMA + h) - bs_price(S, K, T, R, SIGMA - h)) / (2 * h)
    check("vega", vega(S, K, T, R, SIGMA), fd_vega, tol=1e-6)

    fd_theta = -(bs_price(S, K, T + h, R, SIGMA) - bs_price(S, K, T - h, R, SIGMA)) / (2 * h)
    check("call theta (per year)", theta(S, K, T, R, SIGMA), fd_theta, tol=1e-6)
    fd_theta_p = -(bs_price(S, K, T + h, R, SIGMA, option_type="put")
                   - bs_price(S, K, T - h, R, SIGMA, option_type="put")) / (2 * h)
    check("put theta (per year)", theta(S, K, T, R, SIGMA, option_type="put"),
          fd_theta_p, tol=1e-6)

    # Known delta value for the ATM textbook case
    print("\nSpot Greek values (textbook case):")
    check("call delta", delta(S, K, T, R, SIGMA), 0.636831)
    check("put delta", delta(S, K, T, R, SIGMA, option_type="put"), -0.363169)
    check("gamma", gamma(S, K, T, R, SIGMA), 0.018762)
    check("vega (per 100 vol pts)", vega(S, K, T, R, SIGMA), 37.524035)

    print("\nAll Black-Scholes tests passed.")


if __name__ == "__main__":
    main()
