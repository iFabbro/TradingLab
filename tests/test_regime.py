"""Test minimi per src/regime.py — Fase 5"""

import numpy as np
import pandas as pd
import pytest
from src.regime import (
    classify_trend,
    classify_volatility,
    classify_volume,
    detect_regime,
    suggest_strategy,
)
from src.portfolio import regime_allocation_hint

# ---------------------------------------------------------------------------
# Fixture dati sintetici
# ---------------------------------------------------------------------------
@pytest.fixture
def trending_up():
    """Serie in trend rialzista chiaro."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.Series(np.linspace(100, 200, 100), index=idx)


@pytest.fixture
def flat_series():
    """Serie piatta (sideways)."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.Series(np.ones(100) * 100, index=idx)


@pytest.fixture
def volume_series():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.Series(np.ones(100) * 1_000_000, index=idx)


# ---------------------------------------------------------------------------
# Test trend
# ---------------------------------------------------------------------------
def test_classify_trend_bullish(trending_up):
    result = classify_trend(trending_up, fast=5, slow=20)
    # dopo warm-up, la maggioranza deve essere bullish
    counts = result.value_counts()
    assert counts.get("bullish", 0) > counts.get("bearish", 0)


def test_classify_trend_flat(flat_series):
    result = classify_trend(flat_series, fast=5, slow=20)
    valid = result[result != "unknown"]
    # serie piatta → sma_fast == sma_slow → sideways
    assert (valid == "sideways").all()


def test_classify_trend_length(trending_up):
    result = classify_trend(trending_up)
    assert len(result) == len(trending_up)


# ---------------------------------------------------------------------------
# Test volatilità
# ---------------------------------------------------------------------------
def test_classify_volatility_returns_valid_values(trending_up):
    result = classify_volatility(trending_up)
    valid_values = {"high_vol", "low_vol", "normal_vol", "unknown"}
    assert set(result.unique()).issubset(valid_values)


def test_classify_volatility_length(trending_up):
    result = classify_volatility(trending_up)
    assert len(result) == len(trending_up)


# ---------------------------------------------------------------------------
# Test volume
# ---------------------------------------------------------------------------
def test_classify_volume_returns_valid_values(volume_series):
    result = classify_volume(volume_series)
    valid_values = {"high_volume", "low_volume", "normal_volume", "unknown"}
    assert set(result.unique()).issubset(valid_values)


# ---------------------------------------------------------------------------
# Test detect_regime
# ---------------------------------------------------------------------------
def test_detect_regime_columns(trending_up, volume_series):
    df = detect_regime(trending_up, volume=volume_series, fast=5, slow=20)
    assert set(df.columns) >= {"trend", "volatility", "volume_regime", "regime"}


def test_detect_regime_no_volume(trending_up):
    df = detect_regime(trending_up, fast=5, slow=20)
    assert "volume_regime" in df.columns
    assert (df["volume_regime"] == "no_data").all()


def test_regime_string_format(trending_up):
    df = detect_regime(trending_up, fast=5, slow=20)
    # ogni regime deve essere "X_Y"
    valid = df["regime"][df["regime"] != "unknown_unknown"]
    assert valid.str.contains("_").all()


# ---------------------------------------------------------------------------
# Test suggest_strategy
# ---------------------------------------------------------------------------
def test_suggest_strategy_known():
    assert suggest_strategy("bullish_high_vol") == "momentum_long"
    assert suggest_strategy("sideways_normal_vol") == "range_trading"


def test_suggest_strategy_unknown():
    assert suggest_strategy("foo_bar") == "undefined"

def test_regime_to_portfolio_hint_alignment():
    assert regime_allocation_hint("bullish_high_vol") == "directional"
    assert regime_allocation_hint("bearish_low_vol") == "defensive"
    assert regime_allocation_hint("sideways_normal_vol") == "range"
