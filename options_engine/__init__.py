"""GPU-accelerated options pricing and volatility-surface engine.

Modules
-------
black_scholes  Black-76 pricing, Greeks, robust vectorised implied-vol solver
models         Heston / Bates parameters and characteristic functions
cos            Fang-Oosterlee COS pricer (semi-analytic, all strikes at once)
mc             Heston Monte Carlo (Andersen QE scheme) on CPU or GPU
svi            SVI smile fits and static-arbitrage checks
data           Option-chain download, cleaning, put-call-parity forwards
calibration    Heston / Bates calibration and error attribution
benchmark      CPU vs GPU timing and variance-reduction studies
pipeline       End-to-end run that produces results/ and figures/
"""
__version__ = "1.0.0"

from .black_scholes import black_price, black_vega, black_delta, implied_vol  # noqa: F401
from .models import HestonParams, BatesParams  # noqa: F401
from .cos import COSPricer, cos_price  # noqa: F401
