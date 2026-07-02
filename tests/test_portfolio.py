"""Test Fase 7 — Portfolio Construction"""

import numpy as np
import pandas as pd
import pytest
from src.portfolio import equal_weight, inverse_volatility, target_volatility, regime_allocation_hint


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    return pd.DataFrame({
        "A": np.random.normal(0.001, 0.02, 252),
        "B": np.random.normal(0.0005, 0.01, 252),
        "C": np.random.normal(0.002, 0.03, 252),
    })


# --- equal_weight ---

def test_equal_weight_sum():
    w = equal_weight(["A", "B", "C"])
    assert abs(sum(w.values()) - 1.0) < 1e-4

def test_equal_weight_values():
    w = equal_weight(["A", "B"])
    assert w["A"] == w["B"]

def test_equal_weight_empty():
    with pytest.raises(ValueError):
        equal_weight([])


# --- inverse_volatility ---

def test_inv_vol_sum(sample_returns):
    w = inverse_volatility(sample_returns)
    assert abs(sum(w.values()) - 1.0) < 1e-4

def test_inv_vol_higher_weight_lower_vol(sample_returns):
    w = inverse_volatility(sample_returns)
    # B ha vol più bassa → peso più alto
    assert w["B"] > w["A"]
    assert w["B"] > w["C"]

def test_inv_vol_keys(sample_returns):
    w = inverse_volatility(sample_returns)
    assert set(w.keys()) == {"A", "B", "C"}


# --- target_volatility ---

def test_target_vol_output_keys(sample_returns):
    w = target_volatility(sample_returns, target_vol=0.10)
    assert set(w.keys()) == {"A", "B", "C"}

def test_target_vol_scaling(sample_returns):
    w1 = target_volatility(sample_returns, target_vol=0.10)
    w2 = target_volatility(sample_returns, target_vol=0.20)
    # vol target doppia → pesi doppi
    for k in w1:
        assert abs(w2[k] / w1[k] - 2.0) < 0.01

def test_target_vol_invalid():
    with pytest.raises(ValueError):
        target_volatility(pd.DataFrame({"A": [0.01]}), target_vol=-0.05)

def test_regime_allocation_hint():
    assert regime_allocation_hint('bullish_high_vol') == 'directional'
    assert regime_allocation_hint('bearish_low_vol') == 'defensive'
    assert regime_allocation_hint('sideways_normal_vol') == 'range'
    assert regime_allocation_hint('foo_bar') == 'neutral'
