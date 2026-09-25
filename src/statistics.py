"""Statistical diagnostics for walk-forward OOS research.

The module deliberately separates:
- window-level descriptive summaries;
- dependent-data bootstrap inference for OOS returns;
- Probabilistic Sharpe Ratio (PSR);
- Deflated Sharpe Ratio (DSR) for a *declared* nominal number of trials.

DSR is not presented as an oracle: the effective number of independent trials is
usually unknown. The report therefore records the actual candidate-trial ledger
and the nominal-trial assumption used by the DSR calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int
    method: str = "iid"
    block_length: int | None = None


def _validate_confidence_samples(confidence: float, samples: int) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if samples <= 0:
        raise ValueError("samples must be > 0")


def bootstrap_mean_ci(values, confidence: float = 0.95, samples: int = 5000, seed: int = 42) -> BootstrapCI:
    """IID percentile bootstrap CI for the mean of observations.

    This is appropriate for independent window-level observations only. It is
    intentionally not used for the primary daily OOS inference, where returns
    may be serially dependent.
    """
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("at least two finite observations are required")
    _validate_confidence_samples(confidence, samples)
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(samples, x.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return BootstrapCI(float(x.mean()), float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha)), confidence, samples)


def _moving_block_sample(x: np.ndarray, rng: np.random.Generator, block_length: int) -> np.ndarray:
    n = len(x)
    if block_length < 1 or block_length > n:
        raise ValueError("block_length must be between 1 and len(values)")
    starts = rng.integers(0, n - block_length + 1, size=math.ceil(n / block_length))
    sample = np.concatenate([x[s:s + block_length] for s in starts])
    return sample[:n]


def block_bootstrap_ci(
    values,
    statistic,
    confidence: float = 0.95,
    samples: int = 5000,
    block_length: int | None = None,
    seed: int = 42,
) -> BootstrapCI:
    """Moving-block percentile bootstrap CI for a time-series statistic.

    Blocks preserve short-range serial dependence. The block length is a
    declared modelling choice and is included in the returned metadata.
    """
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        raise ValueError("at least 20 finite observations are required")
    _validate_confidence_samples(confidence, samples)
    if block_length is None:
        block_length = max(5, min(20, int(round(math.sqrt(x.size)))))
    block_length = int(block_length)
    if not 1 <= block_length <= x.size:
        raise ValueError("invalid block_length")
    estimate = float(statistic(x))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for i in range(samples):
        draws[i] = float(statistic(_moving_block_sample(x, rng, block_length)))
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        raise ValueError("bootstrap produced no finite statistics")
    alpha = (1 - confidence) / 2
    return BootstrapCI(estimate, float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha)), confidence, len(draws), "moving_block", block_length)


def sharpe_ratio(returns, annualisation: float = 252.0) -> float:
    """Annualised arithmetic Sharpe ratio for a return series."""
    x = np.asarray(list(returns), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    std = float(np.std(x, ddof=0))
    if std == 0:
        return 0.0
    return float(math.sqrt(annualisation) * np.mean(x) / std)


def _psr_z(observed_sharpe: float, benchmark_sharpe: float, n: int, skewness: float, excess_kurtosis: float) -> float:
    if n < 2:
        return float("nan")
    if not all(np.isfinite(v) for v in (observed_sharpe, benchmark_sharpe, skewness, excess_kurtosis)):
        return float("nan")
    denominator = 1.0 - skewness * observed_sharpe + ((excess_kurtosis + 3.0) - 1.0) * observed_sharpe**2 / 4.0
    denominator = max(denominator, 1e-12)
    return float((observed_sharpe - benchmark_sharpe) * math.sqrt(n - 1) / math.sqrt(denominator))


def probabilistic_sharpe_ratio(
    returns,
    benchmark_sharpe: float = 0.0,
    annualisation: float = 252.0,
) -> dict:
    """PSR for a return series, including skewness/kurtosis adjustment.

    The Sharpe ratio and benchmark are supplied in annualised units, while the
    finite-sample correction is applied after converting them to the sampling
    frequency of the return series. This is the form implied by the PSR/DSR
    framework of Bailey & López de Prado.
    """
    x = np.asarray(list(returns), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return {"probability": float("nan"), "z_score": float("nan"), "n": int(x.size)}
    daily_sr = sharpe_ratio(x, annualisation=1.0)
    daily_benchmark = float(benchmark_sharpe) / math.sqrt(annualisation)
    sk = float(skew(x, bias=False)) if x.size > 2 else 0.0
    ex_kurt = float(kurtosis(x, fisher=True, bias=False)) if x.size > 3 else 0.0
    z = _psr_z(daily_sr, daily_benchmark, len(x), sk, ex_kurt)
    probability = float(NormalDist().cdf(z)) if np.isfinite(z) else float("nan")
    return {
        "probability": probability,
        "z_score": z,
        "observed_sharpe": float(daily_sr * math.sqrt(annualisation)),
        "benchmark_sharpe": float(benchmark_sharpe),
        "n": int(len(x)),
        "skewness": sk,
        "excess_kurtosis": ex_kurt,
    }


def expected_max_sharpe_null(trials: int, observations: int) -> float:
    """Original DSR location benchmark under the zero-SR IID null.

    Bailey & López de Prado approximate the expected maximum of N trials with
    the Euler-Mascheroni correction to the normal extreme-value approximation.
    Here the per-observation Sharpe standard deviation is 1/sqrt(T-1), and the
    returned value is at the observation frequency (not annualised).
    """
    if trials < 1 or observations < 2:
        raise ValueError("trials must be >= 1 and observations >= 2")
    if trials == 1:
        return 0.0
    gamma = 0.5772156649015329
    sigma = 1.0 / math.sqrt(observations - 1)
    nd = NormalDist()
    return float(sigma * ((1 - gamma) * nd.inv_cdf(1 - 1 / trials) + gamma * nd.inv_cdf(1 - 1 / (trials * math.e))))


def deflated_sharpe_ratio(
    returns,
    trials: int,
    annualisation: float = 252.0,
) -> dict:
    """DSR probability for a selected return series.

    The trial count is deliberately explicit. It is the *nominal* number of
    alternatives in the declared search, not an inferred effective count.
    Because dependent trials are common in parameter grids, the report also
    exposes the assumption so users cannot mistake this for an exact correction.
    """
    x = np.asarray(list(returns), dtype=float)
    x = x[np.isfinite(x)]
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if x.size < 3:
        return {"probability": float("nan"), "n": int(x.size), "trials": int(trials), "status": "insufficient_observations"}
    daily_sr = sharpe_ratio(x, annualisation=1.0)
    sr_star = expected_max_sharpe_null(trials, len(x))
    sk = float(skew(x, bias=False))
    ex_kurt = float(kurtosis(x, fisher=True, bias=False))
    z = _psr_z(daily_sr, sr_star, len(x), sk, ex_kurt)
    probability = float(NormalDist().cdf(z)) if np.isfinite(z) else float("nan")
    return {
        "probability": probability,
        "z_score": z,
        "observed_sharpe": float(daily_sr * math.sqrt(annualisation)),
        "benchmark_sharpe": float(sr_star * math.sqrt(annualisation)),
        "n": int(len(x)),
        "trials": int(trials),
        "skewness": sk,
        "excess_kurtosis": ex_kurt,
        "assumption": "nominal independent trials under the zero-Sharpe IID null",
    }


def degradation(is_metric: float, oos_metric: float) -> float:
    """Relative OOS degradation. Positive means OOS is lower than IS."""
    if not np.isfinite(is_metric) or not np.isfinite(oos_metric) or is_metric == 0:
        return float("nan")
    return float((is_metric - oos_metric) / abs(is_metric))


def deflated_sharpe_score(observed_sharpe: float, trials: int, observations: int) -> float:
    """Backward-compatible z-score wrapper for callers using scalar inputs.

    This assumes IID normal returns and is retained only for compatibility;
    new reports should use ``deflated_sharpe_ratio`` with the actual returns.
    """
    if trials < 1 or observations < 2:
        raise ValueError("trials must be >= 1 and observations >= 2")
    sr_star = expected_max_sharpe_null(trials, observations)
    observed_daily = observed_sharpe / math.sqrt(252.0)
    return float((observed_daily - sr_star) * math.sqrt(observations - 1))


def multiple_testing_diagnostics(candidate_count: int, best_validation_sharpe: float, observations: int = 2) -> dict:
    """Compatibility summary; full reports should pass the selected returns."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be >= 1")
    return {
        "candidate_count": int(candidate_count),
        "best_validation_sharpe": float(best_validation_sharpe),
        "selection_log_penalty": float(math.log(candidate_count)),
        "expected_max_sharpe_null": float(expected_max_sharpe_null(candidate_count, observations)),
        "warning": "Use deflated_sharpe_ratio on the selected return series for the full non-normality-adjusted diagnostic.",
    }


def aggregate_window_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize window-level metrics without incorrectly pooling returns."""
    if metrics.empty:
        return pd.DataFrame()
    numeric = metrics.select_dtypes(include=[np.number])
    return numeric.agg(["mean", "median", "std", "min", "max"]).T
