import numpy as np
import pytest

from options_engine.black_scholes import black_price, black_vega, implied_vol
from options_engine.cos import cos_price, cumulants, lewis_price
from options_engine.models import BatesParams, HestonParams

F = 6000.0
HESTON = HestonParams(v0=0.02, kappa=2.0, theta=0.04, sigma=0.9, rho=-0.7)   # Feller violated
BATES = BatesParams.from_heston(HESTON, lam=0.5, mu_j=-0.1, delta_j=0.12)


@pytest.mark.parametrize("params", [HESTON, BATES], ids=["heston", "bates"])
@pytest.mark.parametrize("T", [7 / 365, 30 / 365, 0.5, 2.0])
def test_cos_matches_direct_integration(params, T):
    K = F * np.exp(np.linspace(-3.5, 1.5, 9) * np.sqrt(0.04 * T))
    calls = K >= F
    cos = cos_price(params.log_cf, F, K, T, 0.98, calls)
    ref = np.array([lewis_price(params.log_cf, F, k, T, 0.98, c) for k, c in zip(K, calls)])
    iv_ref = implied_vol(ref, F, K, T, 0.98, calls)
    vol_err = np.abs(cos - ref) / black_vega(F, K, T, iv_ref, 0.98)
    assert np.max(vol_err) < 1e-6          # well below 0.0001 vol points


def test_black_scholes_limit():
    h = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=1e-3, rho=0.0)
    K = F * np.exp(np.linspace(-0.4, 0.3, 8))
    cos = cos_price(h.log_cf, F, K, 1.0, 0.97, K >= F)
    bs = black_price(F, K, 1.0, 0.2, 0.97, K >= F)
    assert np.max(np.abs(cos - bs)) < 1e-2    # constant variance 0.04 -> Black-Scholes with 20% vol


def test_cos_put_call_parity():
    K = F * np.exp(np.linspace(-0.5, 0.3, 12))
    c = cos_price(BATES.log_cf, F, K, 0.5, 0.98, True)
    p = cos_price(BATES.log_cf, F, K, 0.5, 0.98, False)
    assert np.allclose(c - p, 0.98 * (F - K), atol=1e-7)


@pytest.mark.parametrize("params", [HESTON, BATES])
def test_martingale(params):
    assert abs(np.exp(params.log_cf(np.array([-1j]), 1.3))[0] - 1.0) < 1e-12


def test_bates_without_jumps_is_heston():
    b = BatesParams.from_heston(HESTON, lam=0.0)
    u = np.linspace(0, 50, 11)
    assert np.allclose(b.log_cf(u, 0.8), HESTON.log_cf(u, 0.8))


def test_cumulants_of_jump_diffusion():
    # Bates with negligible vol-of-vol is Merton jump-diffusion: c2 and c4 known in closed form
    lam, mu, dl, T, v = 0.8, -0.1, 0.15, 0.3, 0.03
    p = BatesParams(v0=v, kappa=1.0, theta=v, sigma=0.02, rho=0.0, lam=lam, mu_j=mu, delta_j=dl)
    _, c2, c4 = cumulants(p.log_cf, T)
    assert c2 == pytest.approx(v * T + lam * T * (mu**2 + dl**2), rel=2e-3)
    assert c4 == pytest.approx(lam * T * (mu**4 + 6 * mu**2 * dl**2 + 3 * dl**4), rel=5e-2)
