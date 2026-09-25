"""Statistical diagnostics for walk-forward OOS research results."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int


def bootstrap_mean_ci(values, confidence: float = 0.95, samples: int = 5000, seed: int = 42) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of window observations."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("at least two finite observations are required")
    if not 0 < confidence < 1 or samples <= 0:
        raise ValueError("invalid confidence or samples")
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(samples, x.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return BootstrapCI(float(x.mean()), float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha)), confidence, samples)


def degradation(is_metric: float, oos_metric: float) -> float:
    """Relative OOS degradation. Positive means OOS is lower than IS."""
    if not np.isfinite(is_metric) or not np.isfinite(oos_metric) or is_metric == 0:
        return float("nan")
    return float((is_metric - oos_metric) / abs(is_metric))


def deflated_sharpe_score(observed_sharpe: float, trials: int, observations: int) -> float:
    """Approximate DSR-style score accounting for the number of trials.

    This is a diagnostic, not a full Bailey-López de Prado DSR implementation:
    the exact effective number of independent trials is generally unknown.
    """
    if trials < 1 or observations < 2:
        raise ValueError("trials must be >= 1 and observations >= 2")
    if not np.isfinite(observed_sharpe):
        return float("nan")
    from statistics import NormalDist
    p = max(1e-12, min(1 - 1e-12, 1 - 1 / trials))
    expected_max = NormalDist().inv_cdf(p)
    standard_error = 1 / math.sqrt(observations)
    return float((observed_sharpe - expected_max) / standard_error)


def multiple_testing_diagnostics(candidate_count: int, best_validation_sharpe: float, observations: int = 2) -> dict:
    """Return selection burden and an approximate deflated-Sharpe diagnostic."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be >= 1")
    return {
        "candidate_count": int(candidate_count),
        "best_validation_sharpe": float(best_validation_sharpe),
        "selection_log_penalty": float(math.log(candidate_count)),
        "deflated_sharpe_score": deflated_sharpe_score(best_validation_sharpe, candidate_count, observations),
        "warning": "DSR is approximate; dependent trials require an effective-trials model",
    }


def aggregate_window_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize window-level metrics without incorrectly pooling returns."""
    if metrics.empty:
        return pd.DataFrame()
    numeric = metrics.select_dtypes(include=[np.number])
    return numeric.agg(["mean", "median", "std", "min", "max"]).T
