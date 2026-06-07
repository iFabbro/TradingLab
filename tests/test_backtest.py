import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtest import BacktestEngine
from src.strategies import StrategyConfig, BaseStrategy


class DummyStrategy(BaseStrategy):
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        out = pd.Series(0.0, index=prices.columns)
        out.iloc[0] = 1.0
        return out


@pytest.fixture
def prices():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    data = pd.DataFrame(
        {
            "AAPL": np.linspace(100, 120, len(idx)),
            "MSFT": np.linspace(100, 110, len(idx)),
        },
        index=idx,
    )
    return data


def test_backtest_outputs(tmp_path, prices):
    cfg = StrategyConfig(name="dummy", universe=list(prices.columns), lookback=5)
    engine = BacktestEngine(cfg, output_dir=tmp_path)
    result = engine.run(prices, DummyStrategy(cfg))

    assert isinstance(result.trade_log, pd.DataFrame)
    assert isinstance(result.equity_curve, pd.Series)
    assert "cagr" in result.metrics
    assert "sharpe" in result.metrics
    assert "max_drawdown" in result.metrics
    assert "win_rate" in result.metrics
    assert (tmp_path / "trade_log.csv").exists()
    assert (tmp_path / "equity_curve.csv").exists()
    assert (tmp_path / "metrics.csv").exists()


def test_backtest_nonempty_equity(prices, tmp_path):
    cfg = StrategyConfig(name="dummy", universe=list(prices.columns), lookback=5)
    engine = BacktestEngine(cfg, output_dir=tmp_path)
    result = engine.run(prices, DummyStrategy(cfg))
    assert not result.equity_curve.empty
    assert result.equity_curve.iloc[-1] > 0
