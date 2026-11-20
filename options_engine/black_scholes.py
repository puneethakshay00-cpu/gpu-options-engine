r"""Black-76 pricing, Greeks and a robust vectorised implied-volatility solver.

Everything is written on the forward :math:`F` and discount factor :math:`D`
of an expiry, so no assumption about rates or dividends is needed (the data
layer backs both out of put-call parity):

.. math::
    C = D\,[F N(d_1) - K N(d_2)], \qquad P = D\,[K N(-d_2) - F N(-d_1)],
    \qquad d_{1,2} = \frac{\ln(F/K)}{\sigma\sqrt T} \pm \tfrac12 \sigma\sqrt T .
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _bcast(F, K, T, sigma, D, is_call):
    F, K, T, sigma, D = (np.asarray(a, dtype=float) for a in (F, K, T, sigma, D))
    is_call = np.asarray(is_call, dtype=bool)
    return np.broadcast_arrays(F, K, T, sigma, D, is_call)


def _scalar(x):
    return x[()] if isinstance(x, np.ndarray) and x.ndim == 0 else x


def black_price(F, K, T, sigma, D=1.0, is_call=True):
    """Black-76 price of a European call (``is_call=True``) or put."""
    F, K, T, sigma, D, is_call = _bcast(F, K, T, sigma, D, is_call)
    w = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(F / K) / w + 0.5 * w
        d2 = d1 - w
        call = F * ndtr(d1) - K * ndtr(d2)
        put = K * ndtr(-d2) - F * ndtr(-d1)
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    undiscounted = np.where(w > 0.0, np.where(is_call, call, put), intrinsic)
    return _scalar(D * undiscounted)


def black_vega(F, K, T, sigma, D=1.0):
    """dPrice/dsigma (same for calls and puts)."""
    F, K, T, sigma, D, _ = _bcast(F, K, T, sigma, D, True)
    sq = np.sqrt(T)
    w = sigma * sq
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(F / K) / w + 0.5 * w
        vega = D * F * _INV_SQRT_2PI * np.exp(-0.5 * d1 * d1) * sq
    return _scalar(np.where(w > 0.0, vega, 0.0))


def black_delta(F, K, T, sigma, is_call=True):
    """Forward (undiscounted) delta: N(d1) for calls, N(d1) - 1 for puts."""
    F, K, T, sigma, _, is_call = _bcast(F, K, T, sigma, 1.0, is_call)
    w = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(F / K) / w + 0.5 * w
    nd1 = ndtr(d1)
    return _scalar(np.where(is_call, nd1, nd1 - 1.0))


def implied_vol(price, F, K, T, D=1.0, is_call=True, *, tol=1e-12, max_iter=100,
                vol_lo=1e-6, vol_hi=8.0):
    """Vectorised implied volatility: Newton's method inside a shrinking bracket.

    Each iteration takes a Newton step when it stays inside the current
    bracket [lo, hi] and a bisection step otherwise, so it converges for every
    arbitrage-free price (including deep out-of-the-money options where vega is
    tiny and plain Newton diverges). Prices outside the no-arbitrage bounds
    return NaN.
    """
    price, F, K, T, D, is_call = _bcast(price, F, K, T, D, is_call)
    p = price / D
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    upper = np.where(is_call, F, K)
    valid = np.isfinite(p) & (p > intrinsic) & (p < upper) & (T > 0)

    # Brenner-Subrahmanyam style starting point from the time value
    with np.errstate(divide="ignore", invalid="ignore"):
        guess = np.sqrt(2.0 * np.pi / T) * (p - intrinsic) / F
    sig = np.clip(np.where(valid & np.isfinite(guess), guess, 0.2), 0.02, 2.0)
    lo = np.full(sig.shape, vol_lo)
    hi = np.full(sig.shape, vol_hi)
    active = valid.copy()
    for _ in range(max_iter):
        if not active.any():
            break
        f = black_price(F, K, T, sig, 1.0, is_call) - p
        f = np.asarray(f)
        hi = np.where(active & (f > 0.0), sig, hi)
        lo = np.where(active & (f <= 0.0), sig, lo)
        v = np.asarray(black_vega(F, K, T, sig, 1.0))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            newton = sig - f / v
        exact = np.abs(f) <= 1e-14 * p          # already at the root: keep sig
        use_bisect = ~np.isfinite(newton) | (newton < lo) | (newton > hi)
        new = np.where(use_bisect, 0.5 * (lo + hi), newton)
        new = np.where(exact, sig, new)
        converged = exact | (np.abs(new - sig) <= tol * np.maximum(1.0, sig))
        sig = np.where(active, new, sig)
        active &= ~converged
    return _scalar(np.where(valid, sig, np.nan))
