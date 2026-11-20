"""Market data: download an option chain, clean it, and back out forwards.

Design choices (each one is a common interview question):

* **SPX, not SPY.** SPX options are European and cash-settled. SPY options are
  American, so their prices carry an early-exercise premium that a European
  model should not be fitted to.
* **Forwards and discount factors from put-call parity**,
  ``C - P = D (F - K)``, fitted by weighted least squares on near-the-money
  strikes. No assumption about the risk-free rate or the dividend yield.
* **Out-of-the-money quotes only** (puts below F, calls above): the liquid side
  of each strike, and where the volatility information is.
* **PM-settled SPXW** contracts are preferred to AM-settled SPX on dates where
  both exist, and time to expiry runs to the actual settlement time.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .black_scholes import black_delta, black_vega, implied_vol
from .market import Slice

NY = "America/New_York"
CONTRACT_RE = r"^(?P<root>[A-Z]+)(?P<yymmdd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$"
DEFAULT_TARGET_DTES = (7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365)
AM_SETTLED_ROOTS = {"SPX", "NDX", "RUT", "XSP"}   # monthly index options settled at the open
YEAR_SECONDS = 365.0 * 86400.0


def select_expiries(expiries, today, targets=DEFAULT_TARGET_DTES, min_dte=5, max_dte=400) -> List[str]:
    """Pick the listed expiry closest to each target days-to-expiry."""
    today = pd.Timestamp(today)
    cands = [(e, (pd.Timestamp(e) - today).days) for e in expiries]
    cands = [(e, d) for e, d in cands if min_dte <= d <= max_dte]
    chosen = []
    for t in targets:
        if not cands:
            break
        e, _ = min(cands, key=lambda ed: abs(ed[1] - t))
        if e not in chosen:
            chosen.append(e)
    return sorted(chosen)


def fetch_chain(ticker: str = "^SPX", targets=DEFAULT_TARGET_DTES, min_dte: int = 5,
                max_dte: int = 400, pause: float = 0.3, verbose: bool = True):
    """Download calls and puts for a spread of expiries from Yahoo Finance."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    expiries = list(tk.options)
    if not expiries:
        raise RuntimeError(f"No option expiries returned for {ticker} - check the internet connection.")
    now = pd.Timestamp.now(tz="UTC")
    chosen = select_expiries(expiries, now.tz_convert(NY).date(), targets, min_dte, max_dte)
    frames, underlying = [], {}
    for e in chosen:
        for attempt in range(3):
            try:
                oc = tk.option_chain(e)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2.0)
        if oc.underlying:
            underlying = oc.underlying
        n = {}
        for df, cp in ((oc.calls, "C"), (oc.puts, "P")):
            n[cp] = 0 if df is None else len(df)
            if df is None or len(df) == 0:
                continue
            frames.append(df.assign(type=cp, expiry=e))
        if verbose:
            print(f"    {e}: {n['C']} calls, {n['P']} puts")
        time.sleep(pause)
    raw = pd.concat(frames, ignore_index=True)
    spot = underlying.get("regularMarketPrice")
    mtime = underlying.get("regularMarketTime")
    meta = {
        "ticker": ticker,
        "source": "Yahoo Finance via yfinance (quotes delayed ~15 min)",
        "fetched_utc": now.isoformat(),
        "spot": float(spot) if spot else None,
        "underlying_time_utc": pd.Timestamp(int(mtime), unit="s", tz="UTC").isoformat() if mtime else None,
        "expiries": chosen,
    }
    return raw, meta


def save_snapshot(raw: pd.DataFrame, meta: dict, folder="data") -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(meta["fetched_utc"]).strftime("%Y%m%d_%H%M")
    path = folder / f"{meta['ticker'].replace('^', '')}_{stamp}.csv"
    raw.to_csv(path, index=False)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return path


def load_snapshot(path) -> Tuple[pd.DataFrame, dict]:
    path = Path(path)
    raw = pd.read_csv(path)
    with open(path.with_suffix(".json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    return raw, meta


def parity_forward(K, C, P, spread, S_guess, T=None, bands=(0.05, 0.10, 0.20), min_pts=4,
                   D_fixed=None):
    """Forward F and discount factor D from put-call parity, C - P = D F - D K.

    1. Strike band around the money, widening with maturity (about +/- 0.3 sqrt(T)),
       because the slope D is pinned down by the strike range.
    2. Robust line (Theil-Sen median of pairwise slopes): stale or erroneous quotes
       sit hundreds of index points off the line and must not drive the fit.
    3. Quotes more than 5 robust standard deviations from that line are dropped.
    4. Weighted least squares on the rest, weights 1 / (call spread + put spread)^2.
       With ``D_fixed`` only F is estimated (used for short maturities, where D is
       not identified by a few weeks of discounting).

    Returns (F, D, n_points, ok).
    """
    from scipy.stats import theilslopes

    K, C, P, spread = (np.asarray(a, dtype=float) for a in (K, C, P, spread))
    y = C - P
    if T is not None:
        bands = tuple(max(b, 0.3 * np.sqrt(T) * b / bands[0]) for b in bands)
    for band in bands:
        sel = np.abs(K / S_guess - 1.0) <= band
        if sel.sum() >= min_pts:
            break
    else:
        return float("nan"), float("nan"), 0, False
    Ks, ys = K[sel], y[sel]
    w = np.where(np.isfinite(spread[sel]), 1.0 / np.maximum(spread[sel], 1e-4) ** 2, 1.0)
    slope, intercept = theilslopes(ys, Ks)[:2]
    resid = ys - (intercept + slope * Ks)
    scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    keep = np.abs(resid) <= max(5.0 * scale, 1e-6 * S_guess)
    if keep.sum() >= min_pts:
        Ks, ys, w = Ks[keep], ys[keep], w[keep]
    if D_fixed is not None:
        D = float(D_fixed)
        F = float(np.sum(w * (Ks + ys / D)) / np.sum(w))
    else:
        X = np.column_stack([np.ones_like(Ks), Ks])
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(X * sw[:, None], ys * sw, rcond=None)[0]
        D = float(-coef[1])
        F = float(coef[0] / D) if D != 0 else float("nan")
    ok = bool(np.isfinite(F) and 0.8 < D <= 1.02 and 0.7 < F / S_guess < 1.4)
    return F, D, int(len(Ks)), ok


def build_slices(raw: pd.DataFrame, meta: dict, *, min_dte: float = 5.0, max_rel_spread: float = 0.5,
                 min_abs_delta: float = 0.02, max_iv: float = 3.0, min_iv: float = 0.02,
                 min_price: float = 0.05, min_quotes: int = 8, price_source: str = "auto",
                 rate_min_T: float = 0.1, verbose: bool = True):
    """Clean a raw chain into per-expiry OTM slices with parity forwards and implied vols."""
    df = raw.copy()
    parts = df["contractSymbol"].astype(str).str.extract(CONTRACT_RE)
    df["root"] = parts["root"].fillna("")
    df["lastTradeDate"] = pd.to_datetime(df["lastTradeDate"], utc=True, errors="coerce")
    for col in ("strike", "bid", "ask", "lastPrice", "volume", "openInterest"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    fetched = pd.Timestamp(meta["fetched_utc"])
    fetched = fetched.tz_localize("UTC") if fetched.tzinfo is None else fetched.tz_convert("UTC")
    last_trade = df["lastTradeDate"].max()
    t0 = min(fetched, last_trade) if pd.notna(last_trade) else fetched
    quoted = (df["bid"] > 0) & (df["ask"] > df["bid"])
    if price_source == "auto":
        price_source = "mid" if quoted.mean() > 0.2 else "last"
    spot = meta.get("spot")

    # ---- pass 1: quotes and put-call parity for every expiry
    staged, report = [], []
    for expiry, g in df.groupby("expiry"):
        roots = sorted(r for r in set(g["root"]) if r)
        root = next((r for r in roots if r.endswith("W")), roots[0] if roots else "")
        if root:
            g = g[g["root"] == root]
        settle_clock = "09:30" if root in AM_SETTLED_ROOTS else "16:00"
        settle = pd.Timestamp(f"{expiry} {settle_clock}", tz=NY).tz_convert("UTC")
        T = (settle - t0).total_seconds() / YEAR_SECONDS
        info = {"expiry": expiry, "root": root, "T": T, "dte": T * 365.0}
        if T * 365.0 < min_dte:
            report.append({**info, "status": "skipped: too short"})
            continue
        if price_source == "mid":
            g = g[(g["bid"] > 0) & (g["ask"] > g["bid"])].copy()
            g["price"] = 0.5 * (g["bid"] + g["ask"])
            g["spread"] = g["ask"] - g["bid"]
        else:
            recent = g["lastTradeDate"] >= t0 - pd.Timedelta(days=1)
            g = g[(g["lastPrice"] > 0) & recent & (g["volume"].fillna(0) > 0)].copy()
            g["price"] = g["lastPrice"]
            g["bid"] = np.nan
            g["ask"] = np.nan
            g["spread"] = np.nan
        g = g.drop_duplicates(subset=["type", "strike"])
        calls = g[g["type"] == "C"].set_index("strike")
        puts = g[g["type"] == "P"].set_index("strike")
        common = calls.index.intersection(puts.index).sort_values()
        if len(common) < 4:
            report.append({**info, "status": "skipped: too few call/put pairs"})
            continue
        C, P = calls.loc[common, "price"].values, puts.loc[common, "price"].values
        spread = calls.loc[common, "spread"].values + puts.loc[common, "spread"].values
        S_guess = spot if spot else float(common[np.argmin(np.abs(C - P))])
        F, D, n_par, ok = parity_forward(common.values, C, P, spread, S_guess, T)
        if not ok:
            report.append({**info, "status": "skipped: put-call parity fit failed"})
            continue
        staged.append({"info": info, "g": g, "T": T, "F": F, "D": D, "n_par": n_par,
                       "parity_args": (common.values, C, P, spread, S_guess, T), "rate_source": "parity"})

    # ---- discount factors are not identified by parity at short maturities:
    #      borrow the median implied rate of the well-identified (T >= rate_min_T) expiries
    long_rates = [-np.log(st["D"]) / st["T"] for st in staged if st["T"] >= rate_min_T]
    r_hat = float(np.median(long_rates)) if long_rates else None
    for st in staged:
        if r_hat is not None and st["T"] < rate_min_T:
            D_fix = float(np.exp(-r_hat * st["T"]))
            F, D, n_par, ok = parity_forward(*st["parity_args"], D_fixed=D_fix)
            if ok:
                st.update(F=F, D=D, n_par=n_par, rate_source="median of longer expiries")

    # ---- pass 2: out-of-the-money quotes, implied vols, filters
    slices, rows = [], []
    for st in staged:
        info, g, T, F, D = st["info"], st["g"], st["T"], st["F"], st["D"]
        root = info["root"]
        otm = g[((g["type"] == "P") & (g["strike"] < F)) | ((g["type"] == "C") & (g["strike"] >= F))]
        otm = otm.sort_values("strike")
        K = otm["strike"].values.astype(float)
        is_call = (otm["type"] == "C").values
        mid = otm["price"].values.astype(float)
        bid, ask = otm["bid"].values.astype(float), otm["ask"].values.astype(float)
        iv = implied_vol(mid, F, K, T, D, is_call)
        iv_b = implied_vol(bid, F, K, T, D, is_call)
        iv_a = implied_vol(ask, F, K, T, D, is_call)
        delta = black_delta(F, K, T, iv, is_call)
        vega = black_vega(F, K, T, iv, D)
        keep = (np.isfinite(iv) & (iv > min_iv) & (iv < max_iv) & (np.abs(delta) >= min_abs_delta)
                & (mid >= min_price))
        if price_source == "mid":
            keep &= (ask - bid) / mid <= max_rel_spread
        if keep.sum() < min_quotes:
            report.append({**info, "F": F, "D": D, "status": f"skipped: only {int(keep.sum())} usable quotes"})
            continue
        sl = Slice(expiry=info["expiry"], T=T, F=F, D=D, K=K[keep], is_call=is_call[keep], mid=mid[keep],
                   bid=bid[keep], ask=ask[keep], iv_mid=iv[keep], iv_bid=iv_b[keep], iv_ask=iv_a[keep],
                   vega=vega[keep], delta=delta[keep], root=root)
        slices.append(sl)
        report.append({**info, "F": F, "D": D, "rate_pct": 100 * sl.rate, "rate_source": st["rate_source"],
                       "parity_points": st["n_par"], "n_quotes": sl.n, "atm_iv_pct": 100 * sl.atm_iv(),
                       "status": "ok"})
        rows.append(pd.DataFrame({"expiry": info["expiry"], "T": T, "F": F, "D": D, "K": sl.K, "k": sl.k,
                                  "is_call": sl.is_call, "mid": sl.mid, "bid": sl.bid, "ask": sl.ask,
                                  "iv_mid": sl.iv_mid, "iv_bid": sl.iv_bid, "iv_ask": sl.iv_ask,
                                  "delta": sl.delta, "vega": sl.vega}))
    report.sort(key=lambda r: r["expiry"])
    slices.sort(key=lambda sl: sl.T)
    if not slices:
        raise RuntimeError("No usable expiries after cleaning. If the market is closed and quotes are "
                           "empty, re-run during US market hours (9:30-16:00 New York time).")
    info = {"price_source": price_source, "valuation_time_utc": t0.isoformat(),
            "short_rate_from_long_expiries_pct": None if r_hat is None else 100 * r_hat,
            "n_raw_contracts": int(len(raw)), "n_clean_quotes": int(sum(s.n for s in slices)),
            "expiries": report}
    if verbose:
        print(f"    price source: {price_source}; valuation time {t0.isoformat()}")
        for r in report:
            extra = (f"F={r['F']:.2f} D={r['D']:.5f} r={r['rate_pct']:.2f}% ({r['rate_source']}) "
                     f"n={r['n_quotes']} ATM vol={r['atm_iv_pct']:.1f}%") if r["status"] == "ok" else ""
            print(f"    {r['expiry']} ({r['dte']:.1f}d, {r['root']}): {r['status']} {extra}")
    table = pd.concat(rows, ignore_index=True)
    return slices, table, info
