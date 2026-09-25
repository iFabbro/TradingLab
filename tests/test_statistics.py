import numpy as np
import pandas as pd
import pytest

from src.statistics import (
    aggregate_window_metrics,
    block_bootstrap_ci,
    bootstrap_mean_ci,
    degradation,
    deflated_sharpe_ratio,
    expected_max_sharpe_null,
    multiple_testing_diagnostics,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def test_bootstrap_ci_is_reproducible():
    a = bootstrap_mean_ci([1, 2, 3, 4], samples=1000, seed=7)
    b = bootstrap_mean_ci([1, 2, 3, 4], samples=1000, seed=7)
    assert a == b
    assert a.lower <= a.estimate <= a.upper


def test_degradation():
    assert degradation(2.0, 1.0) == 0.5
    assert degradation(2.0, 3.0) == -0.5
    assert degradation(-0.10, -0.20, metric="max_drawdown") == 1.0
    assert degradation(-0.10, -0.05, metric="max_drawdown") == -0.5
    assert np.isnan(degradation(0.0, 1.0))


def test_multiple_testing_diagnostic():
    result = multiple_testing_diagnostics(10, 1.5)
    assert result["candidate_count"] == 10
    assert result["selection_log_penalty"] > 0
    assert result["expected_max_sharpe_null"] > 0


def test_aggregate_window_metrics():
    result = aggregate_window_metrics(pd.DataFrame({"sharpe": [1.0, 2.0], "return": [0.1, 0.2]}))
    assert result.loc["sharpe", "mean"] == 1.5


def test_aggregate_window_metrics_ignores_non_finite_values_and_reports_counts():
    result = aggregate_window_metrics(pd.DataFrame({"sharpe": [1.0, np.inf, 3.0], "return": [0.1, -np.inf, 0.2]}))
    assert result.loc["sharpe", "mean"] == 2.0
    assert result.loc["sharpe", "finite_count"] == 2
    assert result.loc["sharpe", "total_count"] == 3
    assert np.isfinite(result.loc["return", "mean"])


def test_bootstrap_requires_two_observations():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([1.0])


def test_sharpe_is_zero_for_constant_returns():
    assert sharpe_ratio(np.zeros(100)) == 0.0


def test_psr_is_above_half_for_clear_positive_signal():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.005, 1000)
    result = probabilistic_sharpe_ratio(returns)
    assert result["probability"] > 0.99


def test_dsr_penalises_more_trials():
    rng = np.random.default_rng(8)
    returns = rng.normal(0.0005, 0.01, 1000)
    one = deflated_sharpe_ratio(returns, 1)
    many = deflated_sharpe_ratio(returns, 1000)
    assert many["probability"] < one["probability"]
    assert expected_max_sharpe_null(1000, len(returns)) > expected_max_sharpe_null(2, len(returns))


def test_block_bootstrap_is_deterministic_and_declares_block_length():
    rng = np.random.default_rng(9)
    values = rng.normal(0, 0.01, 200)
    first = block_bootstrap_ci(values, np.mean, samples=200, block_length=10, seed=42)
    second = block_bootstrap_ci(values, np.mean, samples=200, block_length=10, seed=42)
    assert first == second
    assert first.method == "moving_block"
    assert first.block_length == 10


def test_invalid_dsr_trial_count_rejected():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(np.ones(100), 0)
