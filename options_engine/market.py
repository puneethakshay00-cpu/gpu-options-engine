"""Container for one expiry of cleaned, out-of-the-money option quotes."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class Slice:
    expiry: str          # 'YYYY-MM-DD'
    T: float             # year fraction (ACT/365) from quote time to settlement
    F: float             # forward implied by put-call parity
    D: float             # discount factor implied by put-call parity
    K: np.ndarray
    is_call: np.ndarray
    mid: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    iv_mid: np.ndarray
    iv_bid: np.ndarray
    iv_ask: np.ndarray
    vega: np.ndarray     # Black vega at the mid implied vol
    delta: np.ndarray    # forward delta at the mid implied vol
    root: str = ""

    @property
    def k(self) -> np.ndarray:
        """log-moneyness ln(K/F)."""
        return np.log(self.K / self.F)

    @property
    def dte(self) -> float:
        return self.T * 365.0

    @property
    def n(self) -> int:
        return int(len(self.K))

    @property
    def rate(self) -> float:
        """Continuously-compounded rate implied by the discount factor."""
        return float(-np.log(self.D) / self.T)

    def atm_iv(self) -> float:
        order = np.argsort(self.k)
        return float(np.interp(0.0, self.k[order], self.iv_mid[order]))

    def take(self, idx) -> "Slice":
        idx = np.asarray(idx)
        arrays = {f: getattr(self, f)[idx] for f in (
            "K", "is_call", "mid", "bid", "ask", "iv_mid", "iv_bid", "iv_ask", "vega", "delta")}
        return replace(self, **arrays)

    def subsample(self, max_n: int) -> "Slice":
        """Keep at most ``max_n`` quotes, evenly spread in log-moneyness."""
        if self.n <= max_n:
            return self
        order = np.argsort(self.k)
        pick = order[np.unique(np.round(np.linspace(0, self.n - 1, max_n)).astype(int))]
        return self.take(np.sort(pick))
