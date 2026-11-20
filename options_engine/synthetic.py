"""Synthetic SPX-like option chain from a known Bates model.

Used by the tests and by ``run_pipeline.py --synthetic``: the whole pipeline
(cleaning, parity forwards, calibration) can be checked against parameters we
know, with realistic noise, bid-ask spreads and tick rounding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .black_scholes import black_price, black_vega, implied_vol
from .cos import cos_price
from .models import BatesParams

NY = "America/New_York"
TRUE_PARAMS = BatesParams(v0=0.018, kappa=3.0, theta=0.035, sigma=0.7, rho=-0.75,
                          lam=0.6, mu_j=-0.08, delta_j=0.07)


def make_synthetic_chain(params: BatesParams = TRUE_PARAMS, spot: float = 6000.0, r: float = 0.04,
                         q: float = 0.013, dtes=(7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365),
                         iv_noise: float = 0.002, half_spread_vol: float = 0.004, seed: int = 7,
                         asof: str = "2026-11-19 16:15"):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(asof, tz=NY).tz_convert("UTC")
    rows = []
    expiries = []
    for dte in dtes:
        exp_date = (t0.tz_convert(NY) + pd.Timedelta(days=dte)).date()
        expiries.append(str(exp_date))
        settle = pd.Timestamp(f"{exp_date} 16:00", tz=NY).tz_convert("UTC")
        T = (settle - t0).total_seconds() / (365.0 * 86400.0)
        F, D = spot * np.exp((r - q) * T), np.exp(-r * T)
        step = 5.0 if dte <= 30 else (10.0 if dte <= 90 else 25.0)
        sd = max(0.16 * np.sqrt(T), 0.02)
        K = np.arange(np.ceil(F * np.exp(-6 * sd) / step) * step, F * np.exp(3 * sd), step)
        otm_call = K >= F
        true_otm = cos_price(params.log_cf, F, K, T, D, otm_call)
        iv_true = implied_vol(true_otm, F, K, T, D, otm_call)
        good = np.isfinite(iv_true)
        K, iv_true = K[good], iv_true[good]
        for cp, is_call in (("C", True), ("P", False)):
            iv_obs = iv_true + iv_noise * rng.standard_normal(K.size)
            mid = black_price(F, K, T, iv_obs, D, is_call)
            hs = np.maximum(0.05, half_spread_vol * black_vega(F, K, T, iv_obs, D))
            bid = np.floor(np.maximum(mid - hs, 0.0) / 0.05) * 0.05
            ask = np.ceil((mid + hs) / 0.05) * 0.05
            live = ask >= 0.10
            yymmdd = pd.Timestamp(exp_date).strftime("%y%m%d")
            rows.append(pd.DataFrame({
                "contractSymbol": [f"SPXW{yymmdd}{cp}{int(round(k * 1000)):08d}" for k in K[live]],
                "lastTradeDate": t0, "strike": K[live], "lastPrice": mid[live], "bid": bid[live],
                "ask": ask[live], "change": 0.0, "percentChange": 0.0,
                "volume": rng.integers(1, 500, live.sum()), "openInterest": rng.integers(10, 5000, live.sum()),
                "impliedVolatility": np.nan, "inTheMoney": (K[live] < F) if is_call else (K[live] > F),
                "contractSize": "REGULAR", "currency": "USD", "type": cp, "expiry": str(exp_date)}))
    raw = pd.concat(rows, ignore_index=True)
    meta = {"ticker": "SYNTHETIC", "source": "synthetic Bates chain (known parameters)",
            "fetched_utc": t0.isoformat(), "spot": spot, "expiries": expiries,
            "true_params": params.to_dict(), "true_rate": r, "true_div_yield": q}
    return raw, meta
