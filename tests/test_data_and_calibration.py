import numpy as np
import pytest

from options_engine.calibration import bates_starts, calibrate, evaluate, heston_starts
from options_engine.data import build_slices, parity_forward, select_expiries
from options_engine.models import BatesParams, HestonParams
from options_engine.svi import calendar_violations, fit_svi
from options_engine.synthetic import TRUE_PARAMS, make_synthetic_chain


@pytest.fixture(scope="module")
def synthetic():
    raw, meta = make_synthetic_chain(dtes=(7, 21, 45, 90, 180, 365))
    slices, table, info = build_slices(raw, meta, verbose=False)
    return slices, meta


def test_parity_forward_exact():
    K = np.arange(5800, 6201, 25.0)
    Fv, Dv = 6012.3, 0.987
    C = Dv * np.maximum(Fv - K, 0) + 50.0
    P = C - Dv * (Fv - K)
    F, D, n, ok = parity_forward(K, C, P, np.full(K.size, 0.5), 6000.0)
    assert ok and F == pytest.approx(Fv, abs=1e-8) and D == pytest.approx(Dv, abs=1e-12)


def test_forwards_recovered_from_noisy_quotes(synthetic):
    slices, meta = synthetic
    r, q, S = meta["true_rate"], meta["true_div_yield"], meta["spot"]
    for s in slices:
        assert s.F == pytest.approx(S * np.exp((r - q) * s.T), rel=2e-4)
        assert s.D == pytest.approx(np.exp(-r * s.T), abs=1.5e-3)


def test_expiry_selection():
    exps = ["2026-11-20", "2026-11-27", "2026-12-04", "2026-12-18", "2027-01-15", "2027-06-18"]
    assert select_expiries(exps, "2026-11-19", targets=(7, 30, 200), min_dte=5) == \
        ["2026-11-27", "2026-12-18", "2027-06-18"]


def test_svi_fits_are_arbitrage_free(synthetic):
    slices, _ = synthetic
    fits = [fit_svi(s) for s in slices]
    assert all(f.butterfly_free for f in fits)
    assert calendar_violations(fits) == 0
    assert max(f.rmse_vol_pts for f in fits) < 0.35          # injected noise is 0.2 vol pts


def test_bates_calibration_recovers_true_parameters(synthetic):
    slices, _ = synthetic
    h = calibrate(HestonParams, slices, heston_starts(slices)[:4], verbose=False)
    b = calibrate(BatesParams, slices, bates_starts(h.params)[:3], verbose=False)
    true = TRUE_PARAMS.to_dict()
    fit = b.params.to_dict()
    for name in ("v0", "theta", "rho"):
        assert fit[name] == pytest.approx(true[name], rel=0.08), name
    assert b.rmse_vol_pts < 0.3 < h.rmse_vol_pts               # jumps are needed to fit this surface
    ev = evaluate(b.params, slices)
    assert ev["inside_spread"].mean() > 0.85
