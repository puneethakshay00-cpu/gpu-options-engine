r"""Heston Monte Carlo with Andersen's (2008) Quadratic-Exponential (QE) scheme.

* One code path for NumPy (CPU) and CuPy (NVIDIA GPU).
* Optional fused CUDA kernel: the whole QE step for a path is one thread of one
  kernel launch, instead of ~40 separate array operations per time step.
* Martingale-corrected log-price step (Andersen, section 4.3), so
  :math:`E[S_T] = F` holds exactly in discrete time.
* Variance reduction:

  - antithetic variates (standard errors computed on pair averages);
  - conditional Monte Carlo (Willard 1997; Romano & Touzi 1997). Given the
    variance path, :math:`\ln S_T` is Gaussian with

    .. math::
        \tilde F = F \exp\!\Big(\rho I - \tfrac12 \rho^2 \!\int_0^T\! v\,dt\Big),\quad
        I = \frac{v_T - v_0 - \kappa\theta T + \kappa \int_0^T v\,dt}{\sigma},\quad
        \bar w = (1-\rho^2)\!\int_0^T\! v\,dt,

    so the inner expectation is a Black price and only the variance path is
    simulated noise.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backend import get_xp, is_gpu, synchronize, to_numpy
from .models import HestonParams

PSI_C = 1.5  # Andersen's switching level between the quadratic and exponential branches


@dataclass(frozen=True)
class QEConstants:
    em: float     # exp(-kappa dt)
    c1: float     # Var[v(t+dt) | v(t)] = v(t) c1 + c2
    c2: float
    theta: float
    K0: float     # log-price step coefficients (gamma1 = gamma2 = 1/2)
    K1: float
    K2: float
    K3: float
    K4: float
    psic: float = PSI_C


def qe_constants(p: HestonParams, dt: float, gamma1: float = 0.5, gamma2: float = 0.5) -> QEConstants:
    k, th, s, r = p.kappa, p.theta, p.sigma, p.rho
    em = float(np.exp(-k * dt))
    return QEConstants(
        em=em,
        c1=float(s * s * em * (1.0 - em) / k),
        c2=float(th * s * s * (1.0 - em) ** 2 / (2.0 * k)),
        theta=float(th),
        K0=float(-r * k * th * dt / s),
        K1=float(gamma1 * dt * (k * r / s - 0.5) - r / s),
        K2=float(gamma2 * dt * (k * r / s - 0.5) + r / s),
        K3=float(gamma1 * dt * (1.0 - r * r)),
        K4=float(gamma2 * dt * (1.0 - r * r)),
    )


def _qe_step_arrays(xp, v, x, iv, zv, uv, zx, c: QEConstants):
    """One QE step written with array operations (runs on NumPy or CuPy)."""
    m = c.theta + (v - c.theta) * c.em
    s2 = v * c.c1 + c.c2
    psi = s2 / (m * m)
    A = c.K2 + 0.5 * c.K4
    quad = psi <= c.psic
    # quadratic branch: v' = a (b + Z)^2
    psq = xp.where(quad, psi, 1.0)
    inv = 2.0 / psq
    b2 = inv - 1.0 + xp.sqrt(inv) * xp.sqrt(inv - 1.0)
    a = m / (1.0 + b2)
    vq = a * (xp.sqrt(b2) + zv) ** 2
    den = 1.0 - 2.0 * A * a
    okq = den > 0.0
    dsafe = xp.where(okq, den, 1.0)
    k0q = -A * b2 * a / dsafe + 0.5 * xp.log(dsafe)
    # exponential branch: point mass at 0 plus exponential tail
    pse = xp.where(quad, 3.0, psi)
    p = (pse - 1.0) / (pse + 1.0)
    beta = (1.0 - p) / m
    ve = xp.where(uv <= p, 0.0, xp.log((1.0 - p) / (1.0 - uv)) / beta)
    oke = A < beta
    k0e = -xp.log(p + beta * (1.0 - p) / xp.where(oke, beta - A, 1.0))
    vn = xp.where(quad, vq, ve)
    ok = xp.where(quad, okq, oke)
    k0 = xp.where(ok, xp.where(quad, k0q, k0e) - (c.K1 + 0.5 * c.K3) * v, c.K0)
    x_new = x + k0 + c.K1 * v + c.K2 * vn + xp.sqrt(c.K3 * v + c.K4 * vn) * zx
    iv_new = iv + 0.5 * (v + vn)
    return vn, x_new, iv_new


_QE_KERNEL_BODY = r"""
    // antithetic sign: sg = +1 original path, -1 mirrored path
    T u1 = sg > (T)0 ? uv : (T)1 - uv;
    u1 = min(max(u1, eps), (T)1 - eps);
    T z1 = sg * zv;
    T z2 = sg * zx;
    // conditional moments of v(t+dt) given v(t)
    T m   = theta + (v - theta) * em;
    T s2  = v * c1 + c2;
    T psi = s2 / (m * m);
    T A   = K2 + (T)0.5 * K4;
    T vn, k0;
    bool ok;
    if (psi <= psic) {                       // quadratic branch
        T inv = (T)2 / psi;
        T b2  = inv - (T)1 + sqrt(inv) * sqrt(inv - (T)1);
        T a   = m / ((T)1 + b2);
        T bz  = sqrt(b2) + z1;
        vn = a * bz * bz;
        T den = (T)1 - (T)2 * A * a;
        ok = den > (T)0;
        k0 = ok ? (-A * b2 * a / den + (T)0.5 * log(den)) : (T)0;
    } else {                                 // exponential branch
        T p    = (psi - (T)1) / (psi + (T)1);
        T beta = ((T)1 - p) / m;
        vn = (u1 <= p) ? (T)0 : log(((T)1 - p) / ((T)1 - u1)) / beta;
        ok = A < beta;
        k0 = ok ? -log(p + beta * ((T)1 - p) / (beta - A)) : (T)0;
    }
    // martingale correction (falls back to the plain drift K0 if not defined)
    k0 = ok ? (k0 - (K1 + (T)0.5 * K3) * v) : K0;
    T x_new  = x + k0 + K1 * v + K2 * vn + sqrt(K3 * v + K4 * vn) * z2;
    T iv_new = iv + (T)0.5 * (v + vn);
    v_out  = vn;
    x_out  = x_new;
    iv_out = iv_new;
"""

_KERNEL = None


def _qe_kernel():
    """Compile (once) the fused QE step as a CuPy ElementwiseKernel."""
    global _KERNEL
    if _KERNEL is None:
        import cupy as cp

        _KERNEL = cp.ElementwiseKernel(
            "T v, T x, T iv, T zv, T uv, T zx, T sg, T em, T c1, T c2, T theta, "
            "T K0, T K1, T K2, T K3, T K4, T psic, T eps",
            "T v_out, T x_out, T iv_out",
            _QE_KERNEL_BODY,
            "heston_qe_step",
        )
    return _KERNEL


@dataclass
class HestonPaths:
    x: object          # ln(S_T / F), shape (S, n): S = 2 rows when antithetic
    vT: object         # terminal variance
    int_var: object    # integral of v dt (trapezoid)
    xp: object = field(repr=False)
    seconds: float = 0.0
    n_paths: int = 0
    n_steps: int = 0
    device: str = "cpu"
    impl: str = "numpy"
    dtype: str = "float64"


def simulate_heston_qe(params: HestonParams, T: float, n_steps: int, n_paths: int, *,
                       device: str = "auto", dtype=np.float64, seed: int = 0,
                       antithetic: bool = True, fused: bool = True) -> HestonPaths:
    """Simulate Heston paths with the QE scheme; returns terminal quantities.

    Random numbers per step and path: one normal for the variance (quadratic
    branch), one uniform (exponential branch) and one normal for the log-price.
    """
    xp = get_xp(device)
    dtype = np.dtype(dtype)
    S = 2 if antithetic else 1
    n = max(int(n_paths) // S, 1)
    dt = T / n_steps
    c = qe_constants(params, dt)
    eps = 1e-7 if dtype == np.float32 else 1e-15
    rng = xp.random.default_rng(seed)
    v = xp.full((S, n), params.v0, dtype=dtype)
    x = xp.zeros((S, n), dtype=dtype)
    iv = xp.zeros((S, n), dtype=dtype)
    sg = xp.asarray(np.array([[1.0], [-1.0]])[:S], dtype=dtype)
    use_kernel = fused and is_gpu(xp)
    if use_kernel:
        kern = _qe_kernel()
        v2, x2, iv2 = xp.empty_like(v), xp.empty_like(x), xp.empty_like(iv)
        scalars = [dtype.type(val) for val in (c.em, c.c1, c.c2, c.theta, c.K0, c.K1,
                                                c.K2, c.K3, c.K4, c.psic, eps)]
    synchronize(xp)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        zv = rng.standard_normal(n, dtype=dtype)
        uv = rng.random(n, dtype=dtype)
        zx = rng.standard_normal(n, dtype=dtype)
        if use_kernel:
            kern(v, x, iv, zv, uv, zx, sg, *scalars, v2, x2, iv2)
            v, v2 = v2, v
            x, x2 = x2, x
            iv, iv2 = iv2, iv
        else:
            u1 = xp.clip(xp.where(sg > 0, uv, 1.0 - uv), eps, 1.0 - eps)
            v, x, iv = _qe_step_arrays(xp, v, x, iv, sg * zv, u1, sg * zx, c)
    iv = iv * dt
    synchronize(xp)
    seconds = time.perf_counter() - t0
    impl = ("cupy-fused-kernel" if use_kernel else "cupy") if is_gpu(xp) else "numpy"
    return HestonPaths(x=x, vT=v, int_var=iv, xp=xp, seconds=seconds, n_paths=S * n,
                       n_steps=n_steps, device="gpu" if is_gpu(xp) else "cpu",
                       impl=impl, dtype=dtype.name)


def _ndtr(xp):
    if is_gpu(xp):
        from cupyx.scipy.special import ndtr
    else:
        from scipy.special import ndtr
    return ndtr


@dataclass
class MCResult:
    price: np.ndarray
    stderr: np.ndarray
    seconds: float          # simulation time
    total_seconds: float    # simulation + payoff evaluation
    n_paths: int
    n_steps: int
    device: str
    impl: str
    dtype: str
    method: str


def heston_mc_price(params: HestonParams, F: float, K, T: float, D: float = 1.0, is_call=True, *,
                    n_paths: int = 2**18, n_steps: int = 64, device: str = "auto",
                    dtype=np.float64, seed: int = 0, antithetic: bool = True,
                    fused: bool = True, conditional: bool = False) -> MCResult:
    """European option prices (and standard errors) under Heston by Monte Carlo."""
    t_start = time.perf_counter()
    paths = simulate_heston_qe(params, T, n_steps, n_paths, device=device, dtype=dtype,
                               seed=seed, antithetic=antithetic, fused=fused)
    xp = paths.xp
    Ks = np.atleast_1d(np.asarray(K, dtype=float))
    calls = np.broadcast_to(np.asarray(is_call, dtype=bool), Ks.shape)
    if conditional:
        ndtr = _ndtr(xp)
        r, s, k, th, v0 = params.rho, params.sigma, params.kappa, params.theta, params.v0
        IV = paths.int_var.astype(xp.float64)
        vT = paths.vT.astype(xp.float64)
        I = (vT - v0 - k * th * T + k * IV) / s
        Ft = F * xp.exp(r * I - 0.5 * r * r * IV)
        w = xp.maximum((1.0 - r * r) * IV, 1e-300)
        sw = xp.sqrt(w)
        lnF = xp.log(Ft)
    else:
        ST = F * xp.exp(paths.x.astype(xp.float64))
    prices, errs = [], []
    for Kj, cj in zip(Ks, calls):
        if conditional:
            d1 = (lnF - np.log(Kj) + 0.5 * w) / sw
            call = Ft * ndtr(d1) - Kj * ndtr(d1 - sw)
            pay = call if cj else call - (Ft - Kj)
        else:
            pay = xp.maximum(ST - Kj, 0.0) if cj else xp.maximum(Kj - ST, 0.0)
        per_pair = pay.mean(axis=0)      # antithetic pairs are averaged first
        m = per_pair.shape[0]
        prices.append(D * float(per_pair.mean()))
        errs.append(D * float(per_pair.std(ddof=1)) / np.sqrt(m))
    synchronize(xp)
    return MCResult(price=np.array(prices), stderr=np.array(errs), seconds=paths.seconds,
                    total_seconds=time.perf_counter() - t_start, n_paths=paths.n_paths,
                    n_steps=n_steps, device=paths.device, impl=paths.impl, dtype=paths.dtype,
                    method=("conditional" if conditional else "plain")
                    + ("+antithetic" if antithetic else ""))


def martingale_check(paths: HestonPaths) -> tuple:
    """Sample mean of S_T/F and its standard error (should be 1 within noise)."""
    xp = paths.xp
    ratio = xp.exp(paths.x.astype(xp.float64)).mean(axis=0)
    return float(ratio.mean()), float(ratio.std(ddof=1)) / np.sqrt(ratio.shape[0])


__all__ = ["simulate_heston_qe", "heston_mc_price", "martingale_check", "qe_constants",
           "MCResult", "HestonPaths", "to_numpy"]
