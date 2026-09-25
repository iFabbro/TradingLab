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


def multiple_testing_diagnostics(candidate_count: int, best_validation_sharpe: float) -> dict:
    """Expose selection burden; this is not a full deflated-Sharpe correction."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be >= 1")
    return {
        "candidate_count": int(candidate_count),
        "best_validation_sharpe": float(best_validation_sharpe),
        "selection_log_penalty": float(math.log(candidate_count)),
        "warning": "best validation performance is selection-biased when multiple candidates are tested",
    }


def aggregate_window_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize window-level metrics without incorrectly pooling returns."""
    if metrics.empty:
        return pd.DataFrame()
    numeric = metrics.select_dtypes(include=[np.number])
    return numeric.agg(["mean", "median", "std", "min", "max"]).T
