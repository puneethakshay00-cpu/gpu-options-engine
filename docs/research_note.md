# Do jumps matter for the SPX volatility surface? Heston vs Bates on a live surface

*Research note — data: SPX options, close of 19 Nov 2026; code: this repository.*

## Summary

One parameter set per model is calibrated to 1,734 out-of-the-money SPX quotes across 11 expiries (6–365 days). Heston (stochastic volatility) fits with an implied-vol RMSE of 1.07 vol points; Bates (Heston + jumps) reaches 0.61 (43% lower). The gain concentrates where theory says it should: short-dated out-of-the-money puts, where Heston's RMSE of 2.28 falls to 1.16 (49% lower). The calibrated jump component is a rare crash: ≈0.06 jumps a year with median size −48%.

## Data

Yahoo Finance SPX chain, snapshot `data/SPX_20261120_0412.csv` (valuation time 2026-11-19 16:19 New York). SPX rather than SPY because SPX options are European and cash-settled. Forwards and discount factors come from put–call parity on near-the-money strikes, fitted robustly (Theil–Sen, then weighted least squares on the inliers). Below 36 days the discount factor is not identified by parity, so those expiries use the median implied rate of the longer ones (4.53%). Only out-of-the-money quotes with a positive bid, relative spread ≤ 50% and $|\Delta| \ge 0.02$ are kept, and 2 quotes lying more than 6 robust standard deviations off their expiry's SVI smile are dropped as stale or erroneous. The robust fit matters: a naive fit lost 2 of 11 expiries to a handful of erroneous quotes (one call was quoted about 410 points away from its neighbours).

## Method

Both models are priced with the COS method (Fang & Oosterlee, 2008) and calibrated by vega-weighted least squares with equal weight per expiry,

$$\min_\Theta \sum_j \frac{1}{n_j} \sum_{i\in j} \left( \frac{P^{\text{model}}_i(\Theta) - P^{\text{mid}}_i}{\mathcal V_i} \right)^2,$$

from 8 (Heston) and 8 (Bates) starting points. Errors are then measured exactly, by inverting every model price to an implied volatility.

## Calibrated parameters

| | $v_0$ | $\kappa$ | $\theta$ | $\sigma$ | $\rho$ | $\lambda$ | $\mu_J$ | $\delta_J$ |
|---|---|---|---|---|---|---|---|---|
| Heston | 0.0824 | 8.47 | 0.0569 | 2.57 | -0.745 | – | – | – |
| Bates | 0.0811 | 16.69 | 0.0304 | 1.91 | -0.737 | 0.062 | -0.656 | 0.469 |

Heston needs a very high vol-of-vol ($\sigma = 2.57$) to bend the short end, which violates the Feller condition by a wide margin ($2\kappa\theta/\sigma^2 = 0.15$). With jumps carrying the short-dated skew, Bates lowers $\sigma$ to 1.91.

## Results

By maturity:

| Bucket | n | Heston RMSE | Bates RMSE | Change |
|---|---|---|---|---|
| ≤ 30d | 469 | 1.37 | 0.75 | −45% |
| 31–90d | 631 | 1.16 | 0.61 | −48% |
| > 90d | 634 | 0.63 | 0.47 | −25% |

By moneyness:

| Bucket | n | Heston RMSE | Bates RMSE | Change |
|---|---|---|---|---|
| OTM puts (<25 delta) | 662 | 1.51 | 0.71 | −53% |
| Near ATM (25–50 delta) | 593 | 0.65 | 0.47 | −28% |
| OTM calls (<25 delta) | 479 | 0.67 | 0.60 | −11% |

The largest single improvement is in short-dated out-of-the-money puts (2.28 → 1.16 vol points). Bates is not better everywhere: it gives back accuracy in 2 of the 9 maturity × moneyness buckets (>90-day near-the-money: 0.37 → 0.51; >90-day out-of-the-money calls: 0.48 → 0.67), because one parameter set has to serve the whole surface and the optimiser spends its freedom on the short-dated skew. Over a few weeks a diffusion can only move the price by a few standard deviations of $\sqrt{v\,T}$, so it cannot price deep put wings; a jump can.

## Monte Carlo cross-check

Prices from the GPU Monte Carlo engine (Andersen QE scheme, 252 steps, RTX 3050 6GB Laptop GPU, 1M paths) agree with the COS prices in all 10 checks to within 1.9 standard errors. The smaller CPU check (262K paths) has one draw at 3.6 standard errors, shared by all strikes because they reuse the same paths; re-running it with six other seeds gives pooled z-scores between −1.4 and −0.9, so there is no detectable bias. At a coarse grid (63 steps) the conditional estimator does show the scheme's time-discretisation bias (−2.6 standard errors at 120% moneyness), which is gone by 252 steps. Conditional Monte Carlo reduces variance 6.4× at the money and 95× for a 20% out-of-the-money call, but only 1.6× for a 20% out-of-the-money put: with $\rho = -0.75$ most of the put's risk sits in the variance path, which conditioning cannot remove.

## Limitations

A single end-of-day snapshot; quotes are delayed and not synchronous; the models are fitted to mid prices only; parameter stability over time and hedging performance are not tested. Bates' rare-crash component is weakly identified ($\lambda$ and $\mu_J$ trade off against each other), which is why several crash-type starting points are needed.
