"""Heston and Bates calibration to an implied-volatility surface.

Objective: vega-weighted price errors, which approximate implied-vol errors
without inverting model prices at every iteration::

    r_i = w_slice * (P_model_i - P_mid_i) / vega_i      (~ sigma_model - sigma_mkt)

Each expiry gets the same total weight, so 300 short-dated strikes cannot drown
out 40 long-dated ones. Prices come from the COS method with a cached payoff
matrix per expiry; the truncation range is refreshed at the optimum and the fit
is polished again ("range refresh"). Several starting points are tried because
the Heston likelihood surface has flat valleys (kappa-theta-sigma trade-offs).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .black_scholes import implied_vol
from .cos import COSPricer
from .market import Slice
from .models import BatesParams, HestonParams


@dataclass
class CalibrationResult:
    model: str
    params: object
    rmse_vol_pts: float        # on the calibration sample (vega approximation)
    seconds: float
    n_options: int
    n_starts: int
    nfev: int
    success: bool
    message: str
    start_costs: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"model": self.model, "params": self.params.to_dict(),
                "feller_ratio": float(self.params.feller_ratio),
                "rmse_vol_pts_calibration_sample": self.rmse_vol_pts,
                "seconds": self.seconds, "n_options": self.n_options,
                "n_starts": self.n_starts, "nfev": self.nfev,
                "success": self.success, "message": self.message}


def _pricers(params, slices: Sequence[Slice], N: int, L: float):
    return [COSPricer.for_model(params.log_cf, s.F, s.K, s.T, s.D, s.is_call, N=N, L=L)
            for s in slices]


def calibrate(cls, slices: Sequence[Slice], starts: Sequence[Sequence[float]], *,
              max_per_slice: int = 40, N: int = 1024, L: float = 12.0,
              refresh: int = 2, max_nfev: int = 4000, verbose: bool = True) -> CalibrationResult:
    """Fit ``cls`` (HestonParams or BatesParams) to the quotes in ``slices``."""
    t0 = time.perf_counter()
    cal = [s.subsample(max_per_slice) for s in slices]
    n_tot = sum(s.n for s in cal)
    weights = [np.sqrt(n_tot / (len(cal) * s.n)) for s in cal]
    vegas = [np.maximum(s.vega, 0.01 * s.vega.max()) for s in cal]
    lb, ub = np.array(cls.lower), np.array(cls.upper)

    def make_fun(pricers):
        def fun(z):
            p = cls.from_array(z)
            return np.concatenate([w * (pr.price(p.log_cf) - s.mid) / vg
                                   for s, pr, w, vg in zip(cal, pricers, weights, vegas)])
        return fun

    def solve(z0):
        pricers = _pricers(cls.from_array(z0), cal, N, L)
        return least_squares(make_fun(pricers), z0, bounds=(lb, ub), x_scale=np.array(cls.scale),
                             method="trf", max_nfev=max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10)

    best, costs, nfev = None, [], 0
    for i, x0 in enumerate(starts):
        z0 = np.clip(np.asarray(x0, float), lb + 1e-8 * (ub - lb), ub - 1e-8 * (ub - lb))
        try:
            r = solve(z0)
        except Exception as exc:  # pragma: no cover - defensive
            if verbose:
                print(f"    start {i + 1}: failed ({exc})")
            costs.append(float("nan"))
            continue
        nfev += r.nfev
        costs.append(float(r.cost))
        if verbose:
            rm = 100 * np.sqrt(2 * r.cost / len(r.fun))
            print(f"    start {i + 1}/{len(starts)}: weighted RMSE {rm:.3f} vol pts")
        if best is None or r.cost < best.cost:
            best = r
    for _ in range(refresh):          # rebuild COS ranges at the optimum and polish
        r = solve(best.x)
        nfev += r.nfev
        if r.cost <= best.cost:
            best = r
    params = cls.from_array(best.x)
    # unweighted RMSE on the calibration sample
    pricers = _pricers(params, cal, N, L)
    raw = np.concatenate([(pr.price(params.log_cf) - s.mid) / vg for s, pr, vg in zip(cal, pricers, vegas)])
    return CalibrationResult(model=cls.model, params=params,
                             rmse_vol_pts=float(100 * np.sqrt(np.mean(raw**2))),
                             seconds=time.perf_counter() - t0, n_options=n_tot,
                             n_starts=len(starts), nfev=int(nfev), success=bool(best.success),
                             message=str(best.message), start_costs=costs)


def heston_starts(slices: Sequence[Slice]) -> List[List[float]]:
    """Data-driven starting points: v0 from the shortest ATM vol, theta from the longest."""
    s_sorted = sorted(slices, key=lambda s: s.T)
    v_short = s_sorted[0].atm_iv() ** 2
    v_long = s_sorted[-1].atm_iv() ** 2
    return [[v_short, kappa, v_long, sigma, rho]
            for kappa in (1.0, 4.0) for sigma in (0.5, 1.2) for rho in (-0.8, -0.5)]


def bates_starts(h: HestonParams) -> List[List[float]]:
    """Seeds from the Heston fit plus two jump regimes: frequent small jumps and rare crashes.

    On SPX the best fit is usually the rare-crash regime, which a local optimiser
    started from small jumps does not reach - hence the explicit crash-type starts.
    """
    out = []
    for lam, mu, dl in ((0.2, -0.10, 0.08), (1.0, -0.05, 0.05), (0.1, -0.30, 0.20), (0.05, -0.60, 0.40)):
        for sfac in (1.0, 0.6):
            out.append([h.v0, h.kappa, h.theta, h.sigma * sfac, h.rho, lam, mu, dl])
    return out


def evaluate(params, slices: Sequence[Slice], *, N: int = 1024, L: float = 12.0) -> pd.DataFrame:
    """Model vs market for every quote: prices, implied vols, errors, inside-spread flags."""
    frames = []
    for s in slices:
        model = COSPricer.for_model(params.log_cf, s.F, s.K, s.T, s.D, s.is_call, N=N, L=L).price(params.log_cf)
        iv_model = implied_vol(model, s.F, s.K, s.T, s.D, s.is_call)
        iv_bid = np.where(np.isfinite(s.iv_bid), s.iv_bid, 0.0)
        frames.append(pd.DataFrame({
            "expiry": s.expiry, "T": s.T, "dte": s.dte, "F": s.F, "K": s.K, "k": s.k,
            "is_call": s.is_call, "delta": s.delta, "mid": s.mid, "bid": s.bid, "ask": s.ask,
            "iv_mid": s.iv_mid, "iv_bid": s.iv_bid, "iv_ask": s.iv_ask,
            "model_price": model, "iv_model": iv_model,
            "err_vol_pts": 100.0 * (iv_model - s.iv_mid),
            "inside_spread": (iv_model >= iv_bid) & (iv_model <= s.iv_ask),
        }))
    return pd.concat(frames, ignore_index=True)


MATURITY_BUCKETS = (("<= 30d", 0.0, 30.5), ("31-90d", 30.5, 90.5), ("> 90d", 90.5, 1e9))


def moneyness_bucket(df: pd.DataFrame) -> pd.Series:
    absd = df["delta"].abs()
    lab = np.where(absd >= 0.25, "Near ATM (25-50 delta)",
                   np.where(df["is_call"], "OTM calls (<25 delta)", "OTM puts (<25 delta)"))
    return pd.Series(lab, index=df.index)


def _stats(e: pd.DataFrame) -> dict:
    err = e["err_vol_pts"].dropna()
    return {"n": int(len(e)), "rmse": float(np.sqrt(np.mean(err**2))) if len(err) else float("nan"),
            "mae": float(err.abs().mean()) if len(err) else float("nan"),
            "bias": float(err.mean()) if len(err) else float("nan"),
            "inside_spread_pct": float(100 * e["inside_spread"].mean()) if len(e) else float("nan")}


def error_table(evals: dict) -> pd.DataFrame:
    """RMSE / inside-spread by maturity, moneyness and their cross, for each model."""
    rows = []
    for model, df in evals.items():
        df = df.assign(mat=pd.cut(df["dte"], [b[1] for b in MATURITY_BUCKETS] + [1e9],
                                  labels=[b[0] for b in MATURITY_BUCKETS], right=False),
                       mny=moneyness_bucket(df))
        rows.append({"model": model, "group": "All", "bucket": "All", **_stats(df)})
        for lab, g in df.groupby("mat", observed=True):
            rows.append({"model": model, "group": "Maturity", "bucket": str(lab), **_stats(g)})
        for lab, g in df.groupby("mny"):
            rows.append({"model": model, "group": "Moneyness", "bucket": str(lab), **_stats(g)})
        for (m, n), g in df.groupby(["mat", "mny"], observed=True):
            rows.append({"model": model, "group": "Maturity x Moneyness",
                         "bucket": f"{m} | {n}", **_stats(g)})
    return pd.DataFrame(rows)
