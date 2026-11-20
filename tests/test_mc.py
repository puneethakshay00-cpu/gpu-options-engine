import numpy as np
import pytest

from options_engine.backend import GPU_AVAILABLE
from options_engine.cos import cos_price
from options_engine.mc import heston_mc_price, martingale_check, simulate_heston_qe
from options_engine.models import HestonParams

F, T = 6000.0, 1.0
P = HestonParams(v0=0.02, kappa=2.0, theta=0.04, sigma=0.9, rho=-0.7)   # Feller ratio 0.2
K = F * np.array([0.8, 0.9, 1.0, 1.1, 1.2])
REF = cos_price(P.log_cf, F, K, T, 1.0, K >= F)


@pytest.mark.parametrize("conditional", [False, True], ids=["plain", "conditional"])
def test_qe_monte_carlo_matches_cos(conditional):
    r = heston_mc_price(P, F, K, T, 1.0, K >= F, n_paths=2**17, n_steps=100, device="cpu",
                        conditional=conditional, seed=3)
    z = (r.price - REF) / r.stderr
    assert np.max(np.abs(z)) < 4.0


def test_martingale_correction():
    paths = simulate_heston_qe(P, T, 50, 2**17, device="cpu", seed=9)
    mean, se = martingale_check(paths)
    assert abs(mean - 1.0) < 4 * se


def test_conditional_mc_reduces_variance():
    common = dict(n_paths=2**15, n_steps=50, device="cpu", seed=4)
    plain = heston_mc_price(P, F, F, T, 1.0, True, conditional=False, **common)
    cond = heston_mc_price(P, F, F, T, 1.0, True, conditional=True, **common)
    assert (plain.stderr[0] / cond.stderr[0]) ** 2 > 3.0


def test_float32_matches_float64():
    a = heston_mc_price(P, F, K, T, 1.0, K >= F, n_paths=2**16, n_steps=50, device="cpu",
                        dtype="float32", seed=1)
    z = (a.price - REF) / a.stderr
    assert np.max(np.abs(z)) < 4.0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="no CUDA GPU")
def test_gpu_fused_kernel_matches_array_version():
    a = heston_mc_price(P, F, K, T, 1.0, K >= F, n_paths=2**18, n_steps=64, device="gpu",
                        dtype="float64", fused=True, seed=7)
    b = heston_mc_price(P, F, K, T, 1.0, K >= F, n_paths=2**18, n_steps=64, device="gpu",
                        dtype="float64", fused=False, seed=7)
    assert np.allclose(a.price, b.price, rtol=1e-9, atol=1e-9)   # same random numbers, same maths


@pytest.mark.skipif(not GPU_AVAILABLE, reason="no CUDA GPU")
@pytest.mark.parametrize("conditional", [False, True], ids=["plain", "conditional"])
def test_gpu_matches_cos(conditional):
    r = heston_mc_price(P, F, K, T, 1.0, K >= F, n_paths=2**20, n_steps=100, device="gpu",
                        dtype="float32", conditional=conditional, seed=5)
    assert np.max(np.abs((r.price - REF) / r.stderr)) < 4.0
