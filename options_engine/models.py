r"""Heston (1993) and Bates (1996) models: parameters and characteristic functions.

Characteristic functions are for the log-return of the forward,
:math:`X_T = \ln(S_T / F_0(T))`, so :math:`E[e^{X_T}] = 1` and rates and
dividends drop out of the pricing problem entirely.

Heston:  :math:`dS/S = \sqrt{v}\,dW_1`,
         :math:`dv = \kappa(\theta - v)\,dt + \sigma\sqrt{v}\,dW_2`,
         :math:`d\langle W_1, W_2\rangle = \rho\,dt`.

Bates:   Heston plus compensated Merton jumps in the log-price,
         intensity :math:`\lambda`, jump sizes :math:`N(\mu_J, \delta_J^2)`.

The Heston CF uses the "little Heston trap" form of Albrecher et al. (2007),
which stays on the correct branch of the complex logarithm for long maturities.
"""
from __future__ import annotations

from dataclasses import astuple, dataclass
from typing import ClassVar, Tuple

import numpy as np


def heston_log_cf(u, T, v0, kappa, theta, sigma, rho):
    """log E[exp(i u X_T)] under Heston (forward measure)."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    xi = kappa - rho * sigma * iu
    d = np.sqrt(xi * xi + sigma * sigma * (u * u + iu))
    g = (xi - d) / (xi + d)
    edt = np.exp(-d * T)
    log_term = np.log((1.0 - g * edt) / (1.0 - g))
    C = (kappa * theta / sigma**2) * ((xi - d) * T - 2.0 * log_term)
    Dv = ((xi - d) / sigma**2) * (1.0 - edt) / (1.0 - g * edt)
    return C + Dv * v0


def merton_jump_log_cf(u, T, lam, mu_j, delta_j):
    """log-CF contribution of compensated lognormal jumps (martingale-preserving)."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    compensator = np.exp(mu_j + 0.5 * delta_j**2) - 1.0
    return lam * T * (np.exp(iu * mu_j - 0.5 * delta_j**2 * u * u) - 1.0 - iu * compensator)


class _ParamsMixin:
    names: ClassVar[Tuple[str, ...]]

    def to_array(self) -> np.ndarray:
        return np.array(astuple(self), dtype=float)

    @classmethod
    def from_array(cls, a):
        return cls(*[float(x) for x in a])

    def to_dict(self) -> dict:
        return {n: float(getattr(self, n)) for n in self.names}

    @property
    def feller_ratio(self) -> float:
        """2 kappa theta / sigma^2; above 1 the variance process never hits zero."""
        return 2.0 * self.kappa * self.theta / self.sigma**2


@dataclass(frozen=True)
class HestonParams(_ParamsMixin):
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    model: ClassVar[str] = "Heston"
    names: ClassVar[Tuple[str, ...]] = ("v0", "kappa", "theta", "sigma", "rho")
    lower: ClassVar[Tuple[float, ...]] = (1e-4, 0.05, 1e-4, 0.02, -0.999)
    upper: ClassVar[Tuple[float, ...]] = (1.5, 25.0, 1.5, 5.0, 0.95)
    scale: ClassVar[Tuple[float, ...]] = (0.02, 2.0, 0.02, 0.5, 0.3)

    def log_cf(self, u, T):
        return heston_log_cf(u, T, self.v0, self.kappa, self.theta, self.sigma, self.rho)


@dataclass(frozen=True)
class BatesParams(_ParamsMixin):
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    lam: float
    mu_j: float
    delta_j: float

    model: ClassVar[str] = "Bates"
    names: ClassVar[Tuple[str, ...]] = (
        "v0", "kappa", "theta", "sigma", "rho", "lam", "mu_j", "delta_j")
    lower: ClassVar[Tuple[float, ...]] = (1e-4, 0.05, 1e-4, 0.02, -0.999, 0.0, -1.5, 0.005)
    upper: ClassVar[Tuple[float, ...]] = (1.5, 25.0, 1.5, 5.0, 0.95, 10.0, 0.3, 1.0)
    scale: ClassVar[Tuple[float, ...]] = (0.02, 2.0, 0.02, 0.5, 0.3, 0.5, 0.05, 0.05)

    def log_cf(self, u, T):
        return (heston_log_cf(u, T, self.v0, self.kappa, self.theta, self.sigma, self.rho)
                + merton_jump_log_cf(u, T, self.lam, self.mu_j, self.delta_j))

    @property
    def heston(self) -> HestonParams:
        return HestonParams(self.v0, self.kappa, self.theta, self.sigma, self.rho)

    @classmethod
    def from_heston(cls, h: HestonParams, lam=0.0, mu_j=0.0, delta_j=0.1) -> "BatesParams":
        return cls(h.v0, h.kappa, h.theta, h.sigma, h.rho, lam, mu_j, delta_j)
