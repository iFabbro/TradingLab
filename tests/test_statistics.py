import numpy as np
import pandas as pd
import pytest

from src.statistics import bootstrap_mean_ci, degradation, multiple_testing_diagnostics, aggregate_window_metrics


def test_bootstrap_ci_is_reproducible():
    a = bootstrap_mean_ci([1, 2, 3, 4], samples=1000, seed=7)
    b = bootstrap_mean_ci([1, 2, 3, 4], samples=1000, seed=7)
    assert a == b
    assert a.lower <= a.estimate <= a.upper


def test_degradation():
    assert degradation(2.0, 1.0) == 0.5
    assert np.isnan(degradation(0.0, 1.0))


def test_multiple_testing_diagnostic():
    result = multiple_testing_diagnostics(10, 1.5)
    assert result["candidate_count"] == 10
    assert result["selection_log_penalty"] > 0


def test_aggregate_window_metrics():
    result = aggregate_window_metrics(pd.DataFrame({"sharpe": [1.0, 2.0], "return": [0.1, 0.2]}))
    assert result.loc["sharpe", "mean"] == 1.5


def test_bootstrap_requires_two_observations():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0])
