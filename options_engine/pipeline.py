"""End-to-end run: market data -> cleaning -> SVI -> Heston/Bates calibration ->
Monte Carlo validation, variance reduction and CPU/GPU benchmark -> figures and reports.

    python run_pipeline.py                 # live SPX snapshot from Yahoo Finance
    python run_pipeline.py --snapshot data/SPX_20261120_1500.csv
    python run_pipeline.py --synthetic     # offline: chain generated from known Bates parameters
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, plots
from .backend import GPU_AVAILABLE, hardware_info
from .benchmark import (benchmark_qe, convergence_study, speedup_table, time_to_accuracy,
                        validate_against_cos, variance_reduction_study)
from .calibration import bates_starts, calibrate, error_table, evaluate, heston_starts
from .data import build_slices, fetch_chain, load_snapshot, save_snapshot
from .mc import heston_mc_price
from .models import BatesParams, HestonParams
from .svi import calendar_violations, fit_svi, remove_smile_outliers
from .synthetic import make_synthetic_chain

ROOT = Path(__file__).resolve().parents[1]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _clean(o):
    """Make numpy / pandas objects JSON serialisable."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


def _md_table(df: pd.DataFrame, floatfmt: dict = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(format(v, floatfmt.get(c, ".3f")) if np.isfinite(v) else "-")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _headline_bucket(err: pd.DataFrame, bucket: str) -> dict:
    out = {}
    for model in ("Heston", "Bates"):
        row = err[(err["model"] == model) & (err["bucket"] == bucket)]
        if len(row):
            out[model] = row.iloc[0][["n", "rmse", "inside_spread_pct"]].to_dict()
    if "Heston" in out and "Bates" in out and out["Heston"]["rmse"] > 0:
        out["rmse_reduction_pct"] = 100 * (1 - out["Bates"]["rmse"] / out["Heston"]["rmse"])
    return out


def run(args) -> dict:
    root = Path(args.root)
    out = root / args.out
    figs = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    quick = args.quick
    summary = {"engine_version": __version__,
               "run_started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
               "hardware": hardware_info(), "gpu_available": GPU_AVAILABLE}
    _log(f"GPU available: {GPU_AVAILABLE}  ({summary['hardware'].get('gpu') or 'CPU only'})")

    _log("[1/7] Market data")
    snap = None
    if args.synthetic:
        raw, meta = make_synthetic_chain()
    elif args.snapshot:
        if args.snapshot == "latest":
            found = sorted((root / "data").glob("*.csv"), key=lambda p: p.stat().st_mtime)
            if not found:
                raise SystemExit("No saved snapshot in data/ - run without --snapshot first.")
            snap = found[-1]
        else:
            snap = Path(args.snapshot)
        raw, meta = load_snapshot(snap)
    else:
        try:
            raw, meta = fetch_chain(args.ticker)
        except Exception as exc:
            if args.ticker != "^SPX":
                raise
            _log(f"    ^SPX chain unavailable ({exc}); falling back to SPY (American-style)")
            raw, meta = fetch_chain("SPY")
        snap = save_snapshot(raw, meta, root / "data")
    if snap is not None:
        try:
            meta["snapshot_file"] = snap.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            meta["snapshot_file"] = str(snap)
    summary["data"] = meta
    _log(f"    {len(raw):,} contracts from {meta['source']}")

    _log("[2/7] Cleaning, put-call-parity forwards, implied vols")
    slices, table, info = build_slices(raw, meta)
    summary["cleaning"] = info

    _log("[3/7] SVI smiles, smile-outlier filter, static-arbitrage checks")
    t = time.perf_counter()
    fits = [fit_svi(s) for s in slices]
    slices, removed = remove_smile_outliers(slices, fits)
    if removed:
        changed = {e for e, _, _ in removed}
        fits = [fit_svi(s) if s.expiry in changed else f for s, f in zip(slices, fits)]
        for e, k, r in removed:
            _log(f"    removed quote {e} K={k:g}: {r:+.2f} vol pts from its smile")
    info["smile_outliers_removed"] = [{"expiry": e, "K": k, "residual_vol_pts": r} for e, k, r in removed]
    info["n_quotes_before_smile_filter"] = info["n_clean_quotes"]
    info["n_clean_quotes"] = int(sum(s.n for s in slices))
    pd.concat([pd.DataFrame({"expiry": s.expiry, "T": s.T, "F": s.F, "D": s.D, "K": s.K, "k": s.k,
                             "is_call": s.is_call, "mid": s.mid, "bid": s.bid, "ask": s.ask,
                             "iv_mid": s.iv_mid, "iv_bid": s.iv_bid, "iv_ask": s.iv_ask,
                             "delta": s.delta, "vega": s.vega}) for s in slices],
              ignore_index=True).to_csv(out / "clean_quotes.csv", index=False)
    svi_df = pd.DataFrame([f.to_dict() for f in fits])
    svi_df.to_csv(out / "svi_params.csv", index=False)
    summary["svi"] = {"n_slices": len(fits), "butterfly_free_slices": int(svi_df["butterfly_free"].sum()),
                      "calendar_violations": calendar_violations(fits),
                      "median_rmse_vol_pts": float(svi_df["rmse_vol_pts"].median()),
                      "seconds": time.perf_counter() - t}

    _log("[4/7] Heston calibration (multi-start)")
    h = calibrate(HestonParams, slices, heston_starts(slices))
    _log(f"    {h.params}  ({h.seconds:.1f}s)")
    _log("[5/7] Bates calibration (Heston + jumps, multi-start)")
    b = calibrate(BatesParams, slices, bates_starts(h.params))
    _log(f"    {b.params}  ({b.seconds:.1f}s)")
    eh, eb = evaluate(h.params, slices), evaluate(b.params, slices)
    pd.concat([eh.assign(model="Heston"), eb.assign(model="Bates")]).to_csv(
        out / "model_vs_market.csv", index=False)
    err = error_table({"Heston": eh, "Bates": eb})
    err.to_csv(out / "error_table.csv", index=False)
    summary["calibration"] = {"heston": h.to_dict(), "bates": b.to_dict()}
    summary["headline"] = {
        "all": _headline_bucket(err, "All"),
        "short_dated_otm_puts": _headline_bucket(err, "<= 30d | OTM puts (<25 delta)"),
    }
    if "true_params" in meta:
        summary["synthetic_recovery"] = {"true": meta["true_params"], "bates_fit": b.params.to_dict()}

    mc = {}
    if not args.skip_mc:
        _log("[6/7] Monte Carlo: validation vs COS, variance reduction, CPU vs GPU benchmark")
        hp = h.params
        ref = min(slices, key=lambda s: abs(s.T - 1.0))
        F, T, D = float(ref.F), 1.0, 1.0
        Ks = F * np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        dev = "gpu" if GPU_AVAILABLE else "cpu"
        fast_dtype = "float32" if GPU_AVAILABLE else "float64"
        n_big = 2**14 if quick else (2**20 if GPU_AVAILABLE else 2**18)
        n_cpu = 2**14 if quick else 2**18
        steps = 32 if quick else 252
        mc["setup"] = {"F": F, "T": T, "strikes_over_F": [0.8, 0.9, 1.0, 1.1, 1.2], "n_steps": steps,
                       "params": hp.to_dict(), "note": "forward (undiscounted) prices, calibrated Heston"}
        val = pd.concat([
            validate_against_cos(hp, F, Ks, T, D, n_paths=n_big, n_steps=steps, device=dev, dtype=fast_dtype),
            validate_against_cos(hp, F, Ks, T, D, n_paths=n_cpu, n_steps=steps, device="cpu", dtype="float64"),
        ], ignore_index=True)
        val.to_csv(out / "mc_validation.csv", index=False)
        mc["validation_max_abs_z"] = float(val["z_score"].abs().max())
        _log(f"    validation vs COS: max |z| = {mc['validation_max_abs_z']:.2f}")
        vr = variance_reduction_study(hp, F, Ks, T, D, n_paths=n_big, n_steps=steps, device=dev, dtype=fast_dtype)
        vr.to_csv(out / "variance_reduction.csv", index=False)
        atm = vr[np.isclose(vr["K_over_F"], 1.0)].set_index("method")
        mc["variance_reduction_atm"] = atm["variance_reduction"].to_dict()
        conv = convergence_study(hp, F, F, T, D, device=dev, dtype=fast_dtype, n_steps=steps,
                                 path_counts=(2**10, 2**12, 2**14) if quick else
                                 ((2**10, 2**12, 2**14, 2**16, 2**18, 2**20) if GPU_AVAILABLE
                                  else (2**10, 2**12, 2**14, 2**16, 2**18)))
        conv.to_csv(out / "mc_convergence.csv", index=False)
        bench = speedup_table(benchmark_qe(
            hp, T=T, n_steps=steps,
            path_counts=(2**10, 2**12) if quick else (2**14, 2**16, 2**18, 2**20, 2**22),
            max_cpu_paths=2**12 if quick else 2**20, repeats=1 if quick else 3))
        bench.to_csv(out / "mc_benchmark.csv", index=False)
        mc["benchmark"] = bench.to_dict(orient="records")
        # time to reach a 1 bp (of the forward) standard error on the ATM call
        target = 1e-4 * F
        heston_mc_price(hp, F, F, T, D, True, n_paths=2**12, n_steps=8, device=dev, dtype=fast_dtype,
                        conditional=True)  # warm-up
        base = heston_mc_price(hp, F, F, T, D, True, n_paths=n_cpu, n_steps=steps, device="cpu",
                               dtype="float64", antithetic=True, conditional=False, seed=101)
        best = heston_mc_price(hp, F, F, T, D, True, n_paths=n_big, n_steps=steps, device=dev,
                               dtype=fast_dtype, antithetic=True, conditional=True, seed=102)
        t_base = time_to_accuracy(base.total_seconds, base.stderr[0], target)
        t_best = time_to_accuracy(best.total_seconds, best.stderr[0], target)
        mc["time_to_1bp"] = {
            "target_stderr": target,
            "baseline": {"desc": "NumPy float64 on CPU, plain MC + antithetic", "n_paths": base.n_paths,
                         "seconds": base.total_seconds, "stderr": float(base.stderr[0]), "time_to_target_s": t_base},
            "best": {"desc": f"{best.impl} {best.dtype} on {best.device.upper()}, conditional MC + antithetic",
                     "n_paths": best.n_paths, "seconds": best.total_seconds, "stderr": float(best.stderr[0]),
                     "time_to_target_s": t_best},
            "effective_speedup": t_base / t_best,
        }
        _log(f"    time to 1bp accuracy: baseline {t_base:.1f}s vs best {t_best:.4f}s "
             f"({t_base / t_best:,.0f}x)")
        summary["monte_carlo"] = mc

    _log("[7/7] Figures and reports")
    paths = []
    paths += plots.plot_vol_surface(fits, slices, figs)
    paths += plots.plot_smiles(slices, h.params, b.params, figs)
    paths += plots.plot_error_buckets(err, figs)
    if not args.skip_mc:
        paths += plots.plot_mc_benchmark(bench, figs)
        paths += plots.plot_convergence(conv, figs)
    summary["figures"] = [p.relative_to(root).as_posix() for p in paths]
    summary["total_seconds"] = time.perf_counter() - t_all
    summary = _clean(summary)
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    write_results_md(summary, err, svi_df, out)
    _log(f"Done in {summary['total_seconds']:.0f}s -> {out}")
    return summary


def write_results_md(summary: dict, err: pd.DataFrame, svi_df: pd.DataFrame, out: Path) -> None:
    d, c = summary["data"], summary["cleaning"]
    h, b = summary["calibration"]["heston"], summary["calibration"]["bates"]
    L = [f"# Results ({d['ticker']}, valuation time {c['valuation_time_utc']})", "",
         f"Generated by `run_pipeline.py` (engine v{summary['engine_version']}). "
         f"Source: {d['source']}. Price source: **{c['price_source']}** quotes.", "",
         "## Data", "",
         f"- Raw contracts downloaded: {c['n_raw_contracts']:,}",
         f"- Clean out-of-the-money quotes used: {c['n_clean_quotes']:,} across "
         f"{sum(1 for e in c['expiries'] if e['status'] == 'ok')} expiries "
         f"({len(c.get('smile_outliers_removed', []))} quotes removed as smile outliers)", ""]
    exp = pd.DataFrame([e for e in c["expiries"] if e["status"] == "ok"])
    if len(exp):
        exp = exp[["expiry", "dte", "F", "D", "rate_pct", "parity_points", "n_quotes", "atm_iv_pct"]]
        L += ["Forwards and discount factors implied by put-call parity:", "",
              _md_table(exp, {"dte": ".1f", "F": ".2f", "D": ".5f", "rate_pct": ".2f", "atm_iv_pct": ".1f"}), ""]
    L += ["## SVI fits", "",
          f"- Slices butterfly-arbitrage free (Durrleman g >= 0): {summary['svi']['butterfly_free_slices']}"
          f"/{summary['svi']['n_slices']}",
          f"- Calendar-arbitrage violations (grid points): {summary['svi']['calendar_violations']}",
          f"- Median SVI fit RMSE: {summary['svi']['median_rmse_vol_pts']:.3f} vol points", "",
          "## Calibrated parameters", ""]
    names = ["v0", "kappa", "theta", "sigma", "rho", "lam", "mu_j", "delta_j"]
    prm = pd.DataFrame({"parameter": names,
                        "Heston": [h["params"].get(n, float("nan")) for n in names],
                        "Bates": [b["params"].get(n, float("nan")) for n in names]})
    L += [_md_table(prm, {"Heston": ".4f", "Bates": ".4f"}), "",
          f"Feller ratio 2*kappa*theta/sigma^2: Heston {h['feller_ratio']:.3f}, Bates {b['feller_ratio']:.3f} "
          "(below 1 means the variance process can touch zero, which is why the Monte Carlo uses the QE scheme).",
          f"Calibration time: Heston {h['seconds']:.1f}s ({h['n_starts']} starts), Bates {b['seconds']:.1f}s "
          f"({b['n_starts']} starts), on {h['n_options']} quotes.", "",
          "## Fit quality (implied-vol errors in vol points, all clean quotes)", ""]
    e = err[err["group"] != "Maturity x Moneyness"][["model", "group", "bucket", "n", "rmse", "mae", "bias",
                                                     "inside_spread_pct"]]
    L += [_md_table(e, {"inside_spread_pct": ".1f"}), "", "By maturity and moneyness:", "",
          _md_table(err[err["group"] == "Maturity x Moneyness"][["model", "bucket", "n", "rmse",
                                                                   "inside_spread_pct"]],
                    {"inside_spread_pct": ".1f"}), ""]
    if "monte_carlo" in summary:
        mc = summary["monte_carlo"]
        tt = mc["time_to_1bp"]
        L += ["## Monte Carlo", "",
              f"Setup: calibrated Heston, T = 1 year, F = {mc['setup']['F']:.2f}, {mc['setup']['n_steps']} steps, "
              "Andersen QE scheme with martingale correction.", "",
              f"- Validation against the semi-analytic COS price: max |z-score| = {mc['validation_max_abs_z']:.2f} "
              "(see `mc_validation.csv`)",
              f"- Variance reduction at the money (vs plain MC, same paths): "
              + ", ".join(f"{k}: {v:.1f}x" for k, v in mc["variance_reduction_atm"].items()),
              f"- Time to a 1 bp standard error ({tt['target_stderr']:.3f} index points) on the ATM call: "
              f"{tt['baseline']['desc']}: {tt['baseline']['time_to_target_s']:.1f}s; "
              f"{tt['best']['desc']}: {tt['best']['time_to_target_s']:.4f}s "
              f"-> **{tt['effective_speedup']:,.0f}x** (estimated from measured SE and run time, SE ~ 1/sqrt(N))", "",
              "Simulation time (seconds) and speed-up vs NumPy float64:", ""]
        bench = pd.DataFrame(mc["benchmark"])
        pv = bench.pivot_table(index="n_paths", columns="label", values="seconds")
        pv = pv.reset_index()
        pv.columns = [str(cname) for cname in pv.columns]
        L += [_md_table(pv, {cname: ".4f" for cname in pv.columns}), ""]
        sp = bench.dropna(subset=["speedup_vs_numpy_f64"]).pivot_table(
            index="n_paths", columns="label", values="speedup_vs_numpy_f64").reset_index()
        sp.columns = [str(cname) for cname in sp.columns]
        L += [_md_table(sp, {cname: ".1f" for cname in sp.columns}), ""]
    hw = summary["hardware"]
    L += ["## Hardware", "", f"- CPU: {hw.get('cpu')}", f"- GPU: {hw.get('gpu') or 'none'}"
          + (f" ({hw.get('gpu_memory_gb')} GB, CUDA {hw.get('cuda_runtime')}, CuPy {hw.get('cupy')})"
             if hw.get("gpu") else ""),
          f"- Python {hw.get('python')}, NumPy {hw.get('numpy')}, {hw.get('os')}", ""]
    with open(out / "RESULTS.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description="GPU options pricing & volatility-surface pipeline")
    ap.add_argument("--ticker", default="^SPX")
    ap.add_argument("--snapshot", help="re-run on a saved chain CSV (or 'latest') instead of downloading")
    ap.add_argument("--synthetic", action="store_true", help="offline run on a synthetic Bates chain")
    ap.add_argument("--skip-mc", action="store_true", help="skip the Monte Carlo studies")
    ap.add_argument("--quick", action="store_true", help="small path counts (smoke test)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--root", default=str(ROOT))
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
