"""Interactive dashboard: volatility surface, smile fits, fit diagnostics, pricer, GPU benchmark.

    streamlit run app/dashboard.py

Reads the outputs of ``run_pipeline.py`` from results/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from options_engine.backend import GPU_AVAILABLE, hardware_info  # noqa: E402
from options_engine.black_scholes import implied_vol  # noqa: E402
from options_engine.cos import cos_price  # noqa: E402
from options_engine.mc import heston_mc_price  # noqa: E402
from options_engine.models import BatesParams, HestonParams  # noqa: E402
from options_engine.svi import svi_w  # noqa: E402

RES = Path(os.environ.get("OE_RESULTS", ROOT / "results"))
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"

st.set_page_config(page_title="GPU Options Engine", layout="wide")


@st.cache_data
def load():
    with open(RES / "summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    mvm = pd.read_csv(RES / "model_vs_market.csv")
    svi = pd.read_csv(RES / "svi_params.csv")
    extra = {}
    for name in ("mc_benchmark", "variance_reduction", "mc_convergence", "error_table"):
        p = RES / f"{name}.csv"
        extra[name] = pd.read_csv(p) if p.exists() else None
    return summary, mvm, svi, extra


REQUIRED = ("summary.json", "model_vs_market.csv", "svi_params.csv", "error_table.csv")
if not all((RES / name).exists() for name in REQUIRED):
    st.error("No results yet - run `python run_pipeline.py` (or `--snapshot latest`) first.")
    st.stop()

summary, mvm, svi, extra = load()
H = HestonParams(**summary["calibration"]["heston"]["params"])
B = BatesParams(**summary["calibration"]["bates"]["params"])
clean = summary["cleaning"]

st.title("GPU options pricing & volatility-surface engine")
st.caption(f"{summary['data']['ticker']} option chain, valuation time {clean['valuation_time_utc']} "
           f"({clean['price_source']} prices) - {clean['n_clean_quotes']:,} clean quotes")

hl = summary["headline"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Heston IV RMSE (all quotes)", f"{hl['all']['Heston']['rmse']:.2f} vol pts")
c2.metric("Bates IV RMSE (all quotes)", f"{hl['all']['Bates']['rmse']:.2f} vol pts")
if "short_dated_otm_puts" in hl and "rmse_reduction_pct" in hl["short_dated_otm_puts"]:
    c3.metric("Short-dated OTM put error cut by jumps", f"{hl['short_dated_otm_puts']['rmse_reduction_pct']:.0f}%")
if "monte_carlo" in summary:
    c4.metric("Time-to-accuracy speed-up (GPU + conditional MC)",
              f"{summary['monte_carlo']['time_to_1bp']['effective_speedup']:,.0f}x")

tabs = st.tabs(["Surface", "Smile fits", "Fit quality", "Pricer", "Monte Carlo & GPU"])

# ------------------------------------------------------------------ surface
with tabs[0]:
    svi = svi.sort_values("T")
    zg = np.linspace(-0.65, 0.25, 80)
    Z = []
    for _, r in svi.iterrows():
        k = zg * np.sqrt(r["T"])
        w = svi_w(k, r["a"], r["b"], r["rho"], r["m"], r["sigma"])
        Z.append(100 * np.sqrt(np.maximum(w, 0) / r["T"]))
    fig = go.Figure(go.Surface(x=zg, y=svi["T"], z=np.array(Z), colorscale="Blues", reversescale=False,
                               colorbar=dict(title="IV (%)"),
                               hovertemplate="ln(K/F)/sqrt(T)=%{x:.2f}<br>T=%{y:.2f}y<br>IV=%{z:.2f}%<extra></extra>"))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0),
                      scene=dict(xaxis_title="ln(K/F)/sqrt(T)", yaxis_title="maturity (years)",
                                 zaxis_title="implied vol (%)"))
    st.plotly_chart(fig, width="stretch")
    st.caption("SVI fit per expiry. Arbitrage checks: "
               f"{summary['svi']['butterfly_free_slices']}/{summary['svi']['n_slices']} slices butterfly-free, "
               f"{summary['svi']['calendar_violations']} calendar violations.")

# ------------------------------------------------------------------ smiles
with tabs[1]:
    expiries = sorted(mvm["expiry"].unique())
    exp = st.selectbox("Expiry", expiries, index=0)
    d = mvm[(mvm["expiry"] == exp) & (mvm["model"] == "Heston")].sort_values("K")
    F, T = float(d["F"].iloc[0]), float(d["T"].iloc[0])
    k = np.linspace(d["k"].min(), d["k"].max(), 200)
    K = F * np.exp(k)
    fig = go.Figure()
    lo = 100 * d["iv_bid"].fillna(d["iv_mid"])
    hi = 100 * d["iv_ask"].fillna(d["iv_mid"])
    fig.add_trace(go.Scatter(x=d["k"], y=100 * d["iv_mid"], mode="markers", name="Market mid (bid-ask bars)",
                             marker=dict(color=GREY, size=4),
                             error_y=dict(type="data", symmetric=False, array=hi - 100 * d["iv_mid"],
                                          arrayminus=100 * d["iv_mid"] - lo, color=GREY, thickness=1, width=0)))
    for name, p, col in (("Heston", H, BLUE), ("Bates", B, ORANGE)):
        price = cos_price(p.log_cf, F, K, T, 1.0, K >= F)
        fig.add_trace(go.Scatter(x=k, y=100 * implied_vol(price, F, K, T, 1.0, K >= F), mode="lines",
                                 name=name, line=dict(color=col, width=2)))
    fig.update_layout(height=520, xaxis_title="log-moneyness ln(K/F)", yaxis_title="implied vol (%)",
                      legend=dict(orientation="h", y=1.08), margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{T * 365:.0f} days to expiry, forward {F:,.2f} (from put-call parity).")

# ------------------------------------------------------------------ fit quality
with tabs[2]:
    err = extra["error_table"]
    if err is not None:
        view = err[err["group"] != "Maturity x Moneyness"].copy()
        st.dataframe(view.round({"rmse": 3, "mae": 3, "bias": 3, "inside_spread_pct": 1}), width="stretch")
        cross = err[err["group"] == "Maturity x Moneyness"]
        fig = go.Figure()
        for model, col in (("Heston", BLUE), ("Bates", ORANGE)):
            c = cross[cross["model"] == model]
            fig.add_trace(go.Bar(x=c["bucket"], y=c["rmse"], name=model, marker_color=col))
        fig.update_layout(barmode="group", height=420, yaxis_title="IV RMSE (vol points)",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
    p = pd.DataFrame({"Heston": pd.Series(H.to_dict()), "Bates": pd.Series(B.to_dict())})
    st.subheader("Calibrated parameters")
    st.dataframe(p.round(4), width="stretch")
    st.caption(f"Feller ratio 2 kappa theta / sigma^2: Heston {H.feller_ratio:.2f}, Bates {B.feller_ratio:.2f}")

# ------------------------------------------------------------------ pricer
with tabs[3]:
    left, right = st.columns([1, 2])
    with left:
        model = st.radio("Model", ["Heston", "Bates"], horizontal=True)
        base = H if model == "Heston" else B
        F0 = st.number_input("Forward", value=float(mvm["F"].median()), step=10.0)
        days = st.slider("Days to expiry", 2, 730, 90)
        mny = st.slider("Strike / forward (%)", 50, 150, 100)
        vals = {}
        for name in base.names:
            v = getattr(base, name)
            vals[name] = st.number_input(name, value=float(v), format="%.5f")
        params = (HestonParams if model == "Heston" else BatesParams)(**vals)
    T, K = days / 365.0, F0 * mny / 100.0
    is_call = K >= F0
    price = float(cos_price(params.log_cf, F0, np.array([K]), T, 1.0, is_call)[0])
    with right:
        st.metric(f"{'Call' if is_call else 'Put'} price (COS, undiscounted)", f"{price:,.4f}")
        st.metric("Implied vol", f"{100 * float(implied_vol(price, F0, K, T, 1.0, is_call)):.3f}%")
        if model == "Heston":
            n = st.select_slider("Monte Carlo paths", [2**k for k in range(12, 23)], value=2**18)
            dev = st.radio("Device", ["gpu", "cpu"] if GPU_AVAILABLE else ["cpu"], horizontal=True)
            if st.button("Run Monte Carlo (QE scheme, conditional + antithetic)"):
                r = heston_mc_price(params, F0, K, T, 1.0, is_call, n_paths=n, n_steps=max(16, days),
                                    device=dev, dtype="float32" if dev == "gpu" else "float64",
                                    conditional=True)
                st.write(f"MC price **{r.price[0]:,.4f} +/- {1.96 * r.stderr[0]:.4f}** (95% CI), "
                         f"{r.n_paths:,} paths on {r.device.upper()} in {r.total_seconds:.3f}s; "
                         f"COS - MC = {price - r.price[0]:+.4f} ({(price - r.price[0]) / r.stderr[0]:+.2f} SE)")

# ------------------------------------------------------------------ MC & GPU
with tabs[4]:
    b = extra["mc_benchmark"]
    if b is not None:
        fig = go.Figure()
        palette = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4"]
        for i, lab in enumerate(b["label"].unique()):
            d = b[b["label"] == lab]
            fig.add_trace(go.Scatter(x=d["n_paths"], y=d["seconds"], mode="lines+markers", name=lab,
                                     line=dict(color=palette[i % len(palette)], width=2)))
        fig.update_layout(height=460, xaxis_type="log", yaxis_type="log", xaxis_title="paths",
                          yaxis_title="seconds", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(b[["label", "n_paths", "seconds", "speedup_vs_numpy_f64"]], width="stretch")
    vr = extra["variance_reduction"]
    if vr is not None:
        st.subheader("Variance reduction (same path budget)")
        st.dataframe(vr.pivot_table(index="K_over_F", columns="method", values="variance_reduction").round(1),
                     width="stretch")
    st.caption(f"Hardware: {hardware_info()}")
