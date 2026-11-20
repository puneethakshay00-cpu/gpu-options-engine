r"""Gatheral's raw SVI smile per expiry, plus static-arbitrage diagnostics.

.. math::
    w(k) = a + b\left(\rho (k - m) + \sqrt{(k-m)^2 + \sigma^2}\right),
    \qquad w = \sigma_{BS}^2 T,\ k = \ln(K/F).

Butterfly arbitrage is absent when Durrleman's density condition holds:

.. math::
    g(k) = \Big(1 - \frac{k w'}{2w}\Big)^2 - \frac{w'^2}{4}\Big(\frac1w + \frac14\Big)
    + \frac{w''}{2} \ge 0,

and calendar arbitrage is absent when total variance is non-decreasing in T at
every k.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares

from .market import Slice


def svi_w(k, a, b, rho, m, sig):
    d = np.asarray(k) - m
    return a + b * (rho * d + np.sqrt(d * d + sig * sig))


def svi_derivatives(k, a, b, rho, m, sig):
    d = np.asarray(k) - m
    s = np.sqrt(d * d + sig * sig)
    return a + b * (rho * d + s), b * (rho + d / s), b * sig * sig / s**3


def durrleman_g(k, params) -> np.ndarray:
    w, w1, w2 = svi_derivatives(k, *params)
    k = np.asarray(k)
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - (w1 * w1 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0


@dataclass
class SVIFit:
    expiry: str
    T: float
    params: Tuple[float, float, float, float, float]   # a, b, rho, m, sigma
    rmse_vol_pts: float
    min_g: float          # min of Durrleman's g on the fitted range (>= 0: no butterfly arb)
    k_min: float
    k_max: float

    @property
    def butterfly_free(self) -> bool:
        return self.min_g >= -1e-8

    def iv(self, k) -> np.ndarray:
        return np.sqrt(np.maximum(svi_w(k, *self.params), 0.0) / self.T)

    def to_dict(self) -> dict:
        return {"expiry": self.expiry, "T": self.T,
                **dict(zip(("a", "b", "rho", "m", "sigma"), map(float, self.params))),
                "rmse_vol_pts": self.rmse_vol_pts, "min_g": self.min_g,
                "butterfly_free": self.butterfly_free}


def fit_svi(sl: Slice) -> SVIFit:
    """Least squares in (approximate) implied-vol units with a few starting points."""
    k, iv, T = sl.k, sl.iv_mid, sl.T
    w_mkt = iv * iv * T
    scale = 2.0 * iv * T        # dw = 2 sigma T dsigma  ->  residuals ~ IV errors

    def unpack(z):
        wmin, b, rho, m, sig = z
        return wmin - b * sig * np.sqrt(1.0 - rho * rho), b, rho, m, sig

    def resid(z):
        return (svi_w(k, *unpack(z)) - w_mkt) / scale

    lb = [0.0, 1e-6, -0.999, k.min() - 0.5, 1e-4]
    ub = [max(w_mkt.max(), 1e-6), 20.0, 0.999, k.max() + 0.5, 3.0]
    k_at_min = float(k[np.argmin(w_mkt)])
    best = None
    for rho0 in (-0.7, -0.2):
        for sig0 in (0.05, 0.25):
            for m0 in (k_at_min,):
                z0 = np.clip([0.9 * w_mkt.min(), 0.1 * np.sqrt(T) + 1e-3, rho0, m0, sig0 * np.sqrt(T) + 1e-3],
                             np.array(lb) + 1e-9, np.array(ub) - 1e-9)
                try:
                    r = least_squares(resid, z0, bounds=(lb, ub), method="trf", max_nfev=600)
                except Exception:
                    continue
                if best is None or r.cost < best.cost:
                    best = r
    params = tuple(float(v) for v in unpack(best.x))
    grid = np.linspace(k.min(), k.max(), 400)
    return SVIFit(expiry=sl.expiry, T=sl.T, params=params,
                  rmse_vol_pts=float(100.0 * np.sqrt(np.mean(resid(best.x) ** 2))),
                  min_g=float(durrleman_g(grid, params).min()),
                  k_min=float(k.min()), k_max=float(k.max()))


def calendar_violations(fits: List[SVIFit], n_grid: int = 200, tol: float = 1e-6) -> int:
    """Number of (k, adjacent-expiry) pairs where total variance decreases with T."""
    count = 0
    fits = sorted(fits, key=lambda f: f.T)
    for f1, f2 in zip(fits[:-1], fits[1:]):
        lo, hi = max(f1.k_min, f2.k_min), min(f1.k_max, f2.k_max)
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, n_grid)
        count += int(np.sum(svi_w(grid, *f2.params) < svi_w(grid, *f1.params) - tol))
    return count


def svi_surface(fits: List[SVIFit], k_grid: np.ndarray, margin: float = 0.02) -> np.ndarray:
    """Implied-vol surface (n_expiries x n_k), NaN outside each slice's quoted range."""
    fits = sorted(fits, key=lambda f: f.T)
    Z = np.full((len(fits), len(k_grid)), np.nan)
    for i, f in enumerate(fits):
        inside = (k_grid >= f.k_min - margin) & (k_grid <= f.k_max + margin)
        Z[i, inside] = f.iv(k_grid[inside])
    return Z


def remove_smile_outliers(slices, fits, n_mad: float = 6.0, floor_vol_pts: float = 1.0):
    """Drop quotes far from their own expiry's SVI smile (stale or erroneous quotes).

    A quote is removed when |IV - SVI| exceeds max(n_mad robust SDs, floor) vol points.
    Returns (new_slices, removed) where removed lists (expiry, strike, residual_vol_pts).
    """
    out, removed = [], []
    for sl, f in zip(slices, fits):
        r = 100.0 * (sl.iv_mid - f.iv(sl.k))
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        bad = np.abs(r) > max(n_mad * mad, floor_vol_pts)
        removed += [(sl.expiry, float(k), float(x)) for k, x in zip(sl.K[bad], r[bad])]
        out.append(sl.take(np.where(~bad)[0]) if bad.any() else sl)
    return out, removed
