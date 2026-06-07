import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.strategies import (
    StrategyConfig,
    MomentumStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
N = 100

@pytest.fixture
def prices():
    np.random.seed(42)
    rng = pd.date_range("2023-01-01", periods=N, freq="B")
    data = 100 * np.exp(np.random.randn(N, len(TICKERS)).cumsum(axis=0) * 0.01)
    return pd.DataFrame(data, index=rng, columns=TICKERS)


def base_config(**kwargs) -> StrategyConfig:
    defaults = dict(name="test", universe=TICKERS, lookback=20)
    defaults.update(kwargs)
    return StrategyConfig(**defaults)


# --- StrategyConfig ---

def test_config_validate_ok():
    cfg = base_config()
    cfg.validate()  # non deve sollevare eccezioni

def test_config_validate_empty_universe():
    cfg = base_config(universe=[])
    with pytest.raises(AssertionError):
        cfg.validate()

def test_config_validate_bad_freq():
    cfg = base_config(rebalance_freq="quarterly")
    with pytest.raises(AssertionError):
        cfg.validate()


# --- MomentumStrategy ---

def test_momentum_returns_series(prices):
    strat = MomentumStrategy(base_config(top_n=3))
    sig = strat.generate_signals(prices)
    assert isinstance(sig, pd.Series)
    assert set(sig.index) == set(TICKERS)

def test_momentum_top_n(prices):
    strat = MomentumStrategy(base_config(top_n=2))
    sig = strat.generate_signals(prices)
    assert sig.sum() == pytest.approx(2.0)

def test_momentum_repr():
    strat = MomentumStrategy(base_config())
    assert "MomentumStrategy" in repr(strat)


# --- MeanReversionStrategy ---

def test_mean_reversion_long_only(prices):
    strat = MeanReversionStrategy(base_config(long_only=True))
    sig = strat.generate_signals(prices)
    assert (sig >= 0).all()

def test_mean_reversion_shape(prices):
    strat = MeanReversionStrategy(base_config())
    sig = strat.generate_signals(prices)
    assert len(sig) == len(TICKERS)


# --- TrendFollowingStrategy ---

def test_trend_binary(prices):
    strat = TrendFollowingStrategy(base_config(params={"fast": 10, "slow": 30}))
    sig = strat.generate_signals(prices)
    assert set(sig.unique()).issubset({0.0, 1.0})

def test_trend_insufficient_data(prices):
    strat = TrendFollowingStrategy(base_config(lookback=200))
    sig = strat.generate_signals(prices.iloc[:10])
    assert (sig == 0.0).all()
