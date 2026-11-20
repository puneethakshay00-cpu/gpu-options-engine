r"""Fang & Oosterlee (2008) COS method for European options.

Let :math:`y = \ln(S_T/K) = x + X_T` with :math:`x = \ln(F/K)`. The density of
:math:`y` is expanded in a cosine series on :math:`[a, b]`; its coefficients come
straight from the characteristic function and the payoff coefficients are
analytic:

.. math::
    V \approx D \sum_{k=0}^{N-1}{}' \operatorname{Re}\!\left[\varphi(u_k)\,
    e^{i u_k (x-a)}\right] V_k, \qquad u_k = \frac{k\pi}{b-a}.

The range is shifted per strike, so :math:`b-a` is the same for all strikes:
the characteristic function is evaluated once per expiry and every strike is
priced by one matrix-vector product. With a fixed range the payoff matrix can
be cached, which is what makes calibration fast.
"""
from __future__ import annotations

import numpy as np


def cumulants(log_cf, T: float, h: float = 1e-3):
    """Cumulants c1, c2, c4 of X_T by finite differences of log phi around 0.

    ln phi(u) = sum_n c_n (iu)^n / n!, so the odd/even parts of ln phi(+-h)
    give c1 and c2; c4 comes from a Richardson combination at h' and 2h'.
    """
    lp = log_cf(np.array([h, -h]), T)
    c1 = float((lp[0].imag - lp[1].imag) / (2.0 * h))
    c2 = max(float(-(lp[0].real + lp[1].real) / (h * h)), 1e-14)
    hh = 0.1 / np.sqrt(c2)
    g = log_cf(np.array([hh, 2.0 * hh]), T).real
    c4 = float(-2.0 * (4.0 * g[0] - g[1]) / hh**4)
    return c1, c2, c4


def truncation_width(c2: float, c4: float, L: float = 12.0, max_width: float = 12.0) -> float:
    """Half-width of the COS range, L * sqrt(c2 + sqrt|c4|) (Fang & Oosterlee 2008).

    The sqrt|c4| term widens the range for fat-tailed (jump / high vol-of-vol)
    densities; the cap avoids exp(b) overflow-driven cancellation.
    """
    return float(min(L * np.sqrt(c2 + np.sqrt(abs(c4))), max_width))


def _payoff_matrix(u, a, b, lo, hi, K, sign):
    """V_k for every strike (rows) and cosine term (columns)."""
    U = u[None, :]
    a_ = a[:, None]
    lo_, hi_ = lo[:, None], hi[:, None]
    arg_hi, arg_lo = U * (hi_ - a_), U * (lo_ - a_)
    e_hi, e_lo = np.exp(hi_), np.exp(lo_)
    chi = (np.cos(arg_hi) * e_hi - np.cos(arg_lo) * e_lo
           + U * (np.sin(arg_hi) * e_hi - np.sin(arg_lo) * e_lo)) / (1.0 + U * U)
    safe_u = np.where(U > 0.0, U, 1.0)
    psi = np.where(U > 0.0, (np.sin(arg_hi) - np.sin(arg_lo)) / safe_u, hi_ - lo_)
    V = (2.0 / (b - a))[:, None] * (K * sign)[:, None] * (chi - psi)
    V[~(hi > lo)] = 0.0
    return V


class COSPricer:
    """Prices one expiry (many strikes) for any model given its log-CF.

    ``center`` and ``width`` define the truncation range of X_T:
    [center - width, center + width].
    """

    def __init__(self, F, K, T, D=1.0, is_call=True, *, center: float, width: float, N: int = 1024):
        self.F, self.T, self.D = float(F), float(T), float(D)
        self.K = np.atleast_1d(np.asarray(K, dtype=float))
        self.is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), self.K.shape).copy()
        self.N, self.center, self.width = int(N), float(center), float(width)
        x = np.log(self.F / self.K)
        a = x + self.center - self.width
        b = x + self.center + self.width
        lo = np.where(self.is_call, np.maximum(a, 0.0), a)
        hi = np.where(self.is_call, b, np.minimum(b, 0.0))
        self.u = np.arange(self.N) * np.pi / (2.0 * self.width)
        self._phase = np.exp(1j * self.u * (self.width - self.center))  # e^{i u (x - a)}
        sign = np.where(self.is_call, 1.0, -1.0)
        self._V = _payoff_matrix(self.u, a, b, lo, hi, self.K, sign)

    @classmethod
    def for_model(cls, log_cf, F, K, T, D=1.0, is_call=True, *, N: int = 1024, L: float = 12.0):
        """Truncation range from the model's own cumulants."""
        c1, c2, c4 = cumulants(log_cf, T)
        return cls(F, K, T, D, is_call, center=c1, width=truncation_width(c2, c4, L), N=N)

    def price(self, log_cf) -> np.ndarray:
        A = np.real(np.exp(log_cf(self.u, self.T)) * self._phase)
        A[0] *= 0.5
        return self.D * (self._V @ A)


def cos_price(log_cf, F, K, T, D=1.0, is_call=True, *, N: int = 1024, L: float = 12.0):
    """Convenience wrapper: price strikes K of one expiry with a model log-CF."""
    return COSPricer.for_model(log_cf, F, K, T, D, is_call, N=N, L=L).price(log_cf)


def lewis_price(log_cf, F, K, T, D=1.0, is_call=True):
    """Reference price by direct numerical integration (Lewis 2001). Slow; for tests."""
    from scipy.integrate import quad

    k = np.log(F / K)

    def integrand(u):
        val = np.exp(1j * u * k + log_cf(np.array([u - 0.5j]), T)[0])
        return val.real / (u * u + 0.25)

    integral, _ = quad(integrand, 0.0, np.inf, limit=2000, epsabs=1e-14, epsrel=1e-12)
    call = F - np.sqrt(F * K) / np.pi * integral
    return D * (call if is_call else call - (F - K))
