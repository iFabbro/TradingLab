import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.strategies import (
    StrategyConfig,
    MomentumStrategy,
    DonchianBreakoutStrategy,
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
    cfg.validate()


def test_config_validate_empty_universe():
    cfg = base_config(universe=[])
    with pytest.raises(ValueError, match="universe cannot be empty"):
        cfg.validate()


def test_config_validate_bad_freq():
    cfg = base_config(rebalance_freq="quarterly")
    with pytest.raises(ValueError, match="invalid rebalance_freq"):
        cfg.validate()


# --- MomentumStrategy ---

def test_momentum_returns_series(prices):
    strat = MomentumStrategy(base_config(top_n=3))
    sig = strat.generate_signals(prices)
    assert isinstance(sig, pd.Series)
    assert set(sig.index) == set(TICKERS)


def test_momentum_top_n_returns_two_nonzero_scores(prices):
    strat = MomentumStrategy(base_config(top_n=2))
    sig = strat.generate_signals(prices)
    selected = set(sig.index[sig != 0])
    expected = set(sig.nlargest(2).index)
    assert len(selected) == 2
    assert selected == expected


def test_momentum_repr():
    strat = MomentumStrategy(base_config())
    assert "MomentumStrategy" in repr(strat)


# --- DonchianBreakoutStrategy ---

def test_donchian_returns_binary_asset_series(prices):
    strat = DonchianBreakoutStrategy(base_config(lookback=20))
    sig = strat.generate_signals(prices)
    assert isinstance(sig, pd.Series)
    assert set(sig.index) == set(TICKERS)
    assert set(sig.unique()).issubset({0.0, 1.0})


def test_donchian_no_lookahead_on_entry():
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    frame = pd.DataFrame({"close": [100, 101, 99, 100, 102, 105]}, index=index)
    strat = DonchianBreakoutStrategy(StrategyConfig(name="donchian", universe=["SPY"], lookback=3))
    signal = strat.generate_time_series_signals(frame)
    assert signal.iloc[3] == 0.0
    assert signal.iloc[4] == 1.0
    assert signal.iloc[5] == 1.0


def test_donchian_exits_below_prior_channel():
    index = pd.date_range("2024-01-01", periods=8, freq="B")
    frame = pd.DataFrame({"close": [100, 101, 99, 100, 102, 103, 98, 97]}, index=index)
    strat = DonchianBreakoutStrategy(StrategyConfig(name="donchian", universe=["SPY"], lookback=3))
    signal = strat.generate_time_series_signals(frame)
    assert signal.iloc[4] == 1.0
    assert signal.iloc[5] == 1.0
    assert signal.iloc[6] == 0.0
    assert signal.iloc[7] == 0.0


def test_donchian_insufficient_history_is_flat():
    strat = DonchianBreakoutStrategy(base_config(lookback=20))
    sig = strat.generate_signals(prices.iloc[:10])
    assert (sig == 0.0).all()


def test_donchian_rejects_invalid_lookback():
    strat = DonchianBreakoutStrategy(StrategyConfig(name="donchian", universe=["SPY"], lookback=1))
    with pytest.raises(ValueError, match="lookback must be > 1"):
        strat.generate_signals(prices.iloc[:10])


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
