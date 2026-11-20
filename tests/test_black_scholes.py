import numpy as np
import pytest

from options_engine.black_scholes import black_delta, black_price, black_vega, implied_vol

F = 6000.0


def test_put_call_parity():
    K = np.linspace(4000, 8000, 41)
    c = black_price(F, K, 0.7, 0.25, 0.97, True)
    p = black_price(F, K, 0.7, 0.25, 0.97, False)
    assert np.allclose(c - p, 0.97 * (F - K), atol=1e-9)


def test_implied_vol_round_trip():
    rng = np.random.default_rng(0)
    n = 50_000
    K = F * np.exp(rng.uniform(-0.8, 0.4, n))
    T = rng.uniform(0.01, 2.0, n)
    sig = rng.uniform(0.05, 1.2, n)
    call = rng.random(n) < 0.5
    price = black_price(F, K, T, sig, 0.97, call)
    iv = implied_vol(price, F, K, T, 0.97, call)
    informative = black_vega(F, K, T, sig, 0.97) > 1e-6 * F   # price still carries vol information
    assert np.all(np.isfinite(iv[informative]))
    assert np.max(np.abs(iv - sig)[informative]) < 1e-8


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(implied_vol(-1.0, F, 6000, 0.5))           # negative price
    assert np.isnan(implied_vol(F + 1.0, F, 6000, 0.5))        # above the forward
    assert np.isnan(implied_vol(10.0, F, 5000, 0.5, 1.0, True))  # below intrinsic (1000)


def test_vega_matches_finite_difference():
    K, T, s, h = 6300.0, 0.4, 0.2, 1e-6
    fd = (black_price(F, K, T, s + h) - black_price(F, K, T, s - h)) / (2 * h)
    assert black_vega(F, K, T, s) == pytest.approx(fd, rel=1e-6)


def test_delta_put_call_relation():
    K = np.array([5000.0, 6000.0, 7000.0])
    assert np.allclose(black_delta(F, K, 0.5, 0.2, True) - black_delta(F, K, 0.5, 0.2, False), 1.0)
