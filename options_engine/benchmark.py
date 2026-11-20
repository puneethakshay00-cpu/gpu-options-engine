"""Timing and accuracy studies for the Monte Carlo engine.

* ``benchmark_qe``: wall-clock time of the QE simulation for several
  implementations (NumPy on CPU, CuPy array ops on GPU, fused CUDA kernel on
  GPU) and precisions, over a range of path counts. GPU timings synchronise
  the device and exclude the one-off kernel compilation (warm-up run).
* ``validate_against_cos``: Monte Carlo prices vs the semi-analytic COS price,
  as z-scores (error / standard error).
* ``variance_reduction_study``: standard error and cost of plain, antithetic,
  conditional and conditional+antithetic estimators at a fixed path budget.
* ``convergence_study``: estimate +/- 95% CI against the number of paths.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from .backend import GPU_AVAILABLE
from .cos import cos_price
from .mc import heston_mc_price, simulate_heston_qe
from .models import HestonParams


def default_configs() -> List[dict]:
    cfg = [
        {"device": "cpu", "fused": False, "dtype": "float64", "label": "NumPy float64 (CPU)"},
        {"device": "cpu", "fused": False, "dtype": "float32", "label": "NumPy float32 (CPU)"},
    ]
    if GPU_AVAILABLE:
        cfg += [
            {"device": "gpu", "fused": False, "dtype": "float32", "label": "CuPy array ops float32 (GPU)"},
            {"device": "gpu", "fused": True, "dtype": "float32", "label": "Fused CUDA kernel float32 (GPU)"},
            {"device": "gpu", "fused": True, "dtype": "float64", "label": "Fused CUDA kernel float64 (GPU)"},
        ]
    return cfg


def benchmark_qe(params: HestonParams, T: float = 1.0, n_steps: int = 252,
                 path_counts: Iterable[int] = (2**14, 2**16, 2**18, 2**20, 2**22),
                 max_cpu_paths: int = 2**20, repeats: int = 3, configs=None,
                 verbose: bool = True) -> pd.DataFrame:
    rows = []
    for cfg in configs or default_configs():
        # warm-up: CUDA context, kernel compilation, memory pool
        simulate_heston_qe(params, T, 8, 2**12, device=cfg["device"], dtype=cfg["dtype"],
                           fused=cfg["fused"])
        for n in path_counts:
            if cfg["device"] == "cpu" and n > max_cpu_paths:
                continue
            reps = repeats if (cfg["device"] == "gpu" or n <= 2**18) else 1
            times = [simulate_heston_qe(params, T, n_steps, n, device=cfg["device"], dtype=cfg["dtype"],
                                        fused=cfg["fused"], seed=r).seconds for r in range(reps)]
            sec = float(np.median(times))
            rows.append({**cfg, "n_paths": int(n), "n_steps": n_steps, "seconds": sec,
                         "path_steps_per_sec": n * n_steps / sec})
            if verbose:
                print(f"    {cfg['label']:<34s} paths={n:>9,d}  {sec:9.4f} s")
    return pd.DataFrame(rows)


def speedup_table(bench: pd.DataFrame, baseline: str = "NumPy float64 (CPU)") -> pd.DataFrame:
    base = bench[bench["label"] == baseline].set_index("n_paths")["seconds"]
    out = bench.copy()
    out["speedup_vs_numpy_f64"] = [base.get(n, np.nan) / s for n, s in zip(out["n_paths"], out["seconds"])]
    return out


def validate_against_cos(params: HestonParams, F: float, K, T: float, D: float = 1.0, *,
                         n_paths: int = 2**20, n_steps: int = 252, device: str = "auto",
                         dtype="float64", fused: bool = True, seed: int = 11) -> pd.DataFrame:
    K = np.asarray(K, dtype=float)
    calls = K >= F
    ref = cos_price(params.log_cf, F, K, T, D, calls)
    rows = []
    for conditional in (False, True):
        r = heston_mc_price(params, F, K, T, D, calls, n_paths=n_paths, n_steps=n_steps,
                            device=device, dtype=dtype, fused=fused, seed=seed,
                            conditional=conditional)
        for j in range(len(K)):
            rows.append({"method": r.method, "device": r.device, "impl": r.impl, "dtype": r.dtype,
                         "n_paths": r.n_paths, "K": K[j], "K_over_F": K[j] / F,
                         "type": "call" if calls[j] else "put", "cos_price": ref[j],
                         "mc_price": r.price[j], "stderr": r.stderr[j],
                         "z_score": (r.price[j] - ref[j]) / r.stderr[j]})
    return pd.DataFrame(rows)


def variance_reduction_study(params: HestonParams, F: float, K, T: float, D: float = 1.0, *,
                             n_paths: int = 2**18, n_steps: int = 252, device: str = "auto",
                             dtype="float64", seed: int = 5) -> pd.DataFrame:
    K = np.asarray(K, dtype=float)
    calls = K >= F
    ref = cos_price(params.log_cf, F, K, T, D, calls)
    methods = [("Plain", False, False), ("Antithetic", True, False),
               ("Conditional", False, True), ("Conditional + antithetic", True, True)]
    rows = []
    for name, anti, cond in methods:
        heston_mc_price(params, F, K, T, D, calls, n_paths=2**12, n_steps=8, device=device,
                        dtype=dtype, antithetic=anti, conditional=cond)   # warm-up
        r = heston_mc_price(params, F, K, T, D, calls, n_paths=n_paths, n_steps=n_steps,
                            device=device, dtype=dtype, antithetic=anti, conditional=cond, seed=seed)
        for j in range(len(K)):
            rows.append({"method": name, "K": K[j], "K_over_F": K[j] / F,
                         "type": "call" if calls[j] else "put", "cos_price": ref[j],
                         "mc_price": r.price[j], "stderr": r.stderr[j], "seconds": r.total_seconds,
                         "device": r.device, "dtype": r.dtype, "n_paths": r.n_paths})
    df = pd.DataFrame(rows)
    plain = df[df["method"] == "Plain"].set_index("K")
    df["variance_reduction"] = [(plain.loc[k, "stderr"] / se) ** 2 for k, se in zip(df["K"], df["stderr"])]
    df["efficiency_gain"] = [vr * plain.loc[k, "seconds"] / s
                             for k, vr, s in zip(df["K"], df["variance_reduction"], df["seconds"])]
    return df


def convergence_study(params: HestonParams, F: float, K: float, T: float, D: float = 1.0, *,
                      path_counts=(2**10, 2**12, 2**14, 2**16, 2**18, 2**20), n_steps: int = 252,
                      device: str = "auto", dtype="float32", seed: int = 21) -> pd.DataFrame:
    rows = []
    ref = float(cos_price(params.log_cf, F, np.array([K]), T, D, K >= F)[0])
    for name, cond in (("Plain + antithetic", False), ("Conditional + antithetic", True)):
        for i, n in enumerate(path_counts):
            r = heston_mc_price(params, F, K, T, D, K >= F, n_paths=n, n_steps=n_steps, device=device,
                                dtype=dtype, conditional=cond, seed=seed + i)
            rows.append({"method": name, "n_paths": n, "price": r.price[0], "stderr": r.stderr[0],
                         "seconds": r.total_seconds, "cos_price": ref})
    return pd.DataFrame(rows)


def time_to_accuracy(seconds: float, stderr: float, target: float) -> float:
    """Run time needed to reach a target standard error, using SE ~ 1/sqrt(N)."""
    return float(seconds * (stderr / target) ** 2)
