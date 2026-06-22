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


class FlatStrategy(BaseStrategy):
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=prices.columns)


class TimeSignalStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig, signal_values: list[float]) -> None:
        super().__init__(config)
        self.signal_values = signal_values

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(self.signal_values, index=prices.index, dtype=float)


@pytest.fixture
def prices():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    data = pd.DataFrame(
        {
            "close": np.linspace(100, 120, len(idx)),
        },
        index=idx,
    )
    return data


def test_backtest_outputs(tmp_path, prices):
    cfg = StrategyConfig(name="dummy", universe=list(prices.columns), lookback=5)
    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, DummyStrategy(cfg))

    assert isinstance(result.trade_log, pd.DataFrame)
    assert isinstance(result.equity_curve, pd.Series)
    assert "cagr" in result.metrics
    assert "sharpe" in result.metrics
    assert "max_drawdown" in result.metrics
    assert "win_rate" in result.metrics
    assert result.trade_log.loc[0, "quantity"] == pytest.approx(1.0)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(20.0)
    assert result.trade_log.loc[0, "return_pct"] == pytest.approx(0.2)
    assert result.metrics["n_trades"] == 1
    assert result.metrics["total_return"] == pytest.approx(20.0 / 100000.0)
    assert result.metrics["win_rate"] == pytest.approx(1.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(100020.0)
    assert (tmp_path / "trade_log.csv").exists()
    assert (tmp_path / "equity_curve.csv").exists()
    assert (tmp_path / "metrics.csv").exists()


def test_backtest_nonempty_equity(prices, tmp_path):
    cfg = StrategyConfig(name="dummy", universe=list(prices.columns), lookback=5)
    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, DummyStrategy(cfg))
    assert not result.equity_curve.empty
    assert result.equity_curve.iloc[-1] > 0

def test_backtest_flat_signal_has_zero_pnl(prices, tmp_path):
    cfg = StrategyConfig(name="flat", universe=list(prices.columns), lookback=5)
    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, FlatStrategy(cfg))

    assert result.trade_log.loc[0, "pnl"] == pytest.approx(0.0)
    assert result.trade_log.loc[0, "return_pct"] == pytest.approx(0.0)
    assert result.metrics["total_return"] == pytest.approx(0.0)
    assert result.metrics["win_rate"] == pytest.approx(0.0)
    assert result.metrics["max_drawdown"] == pytest.approx(0.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(100000.0)


def test_backtest_loss_path_updates_metrics(tmp_path):
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    prices = pd.DataFrame({"close": np.linspace(120, 100, len(idx))}, index=idx)

    cfg = StrategyConfig(name="dummy-loss", universe=list(prices.columns), lookback=5)
    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, DummyStrategy(cfg))

    assert result.trade_log.loc[0, "pnl"] == pytest.approx(-20.0)
    assert result.trade_log.loc[0, "return_pct"] == pytest.approx((100.0 / 120.0) - 1.0)
    assert result.metrics["n_trades"] == 1
    assert result.metrics["total_return"] == pytest.approx(-20.0 / 100000.0)
    assert result.metrics["win_rate"] == pytest.approx(0.0)
    assert result.metrics["max_drawdown"] > 0.0
    assert result.equity_curve.iloc[-1] == pytest.approx(99980.0)

def test_backtest_mark_to_market_equity_curve_has_drawdown_and_nonzero_sharpe(tmp_path):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 105.0, 115.0]}, index=idx)

    cfg = StrategyConfig(name="dummy-mtm", universe=list(prices.columns), lookback=2)
    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, DummyStrategy(cfg))

    expected_equity = pd.Series([100000.0, 100010.0, 100005.0, 100015.0], index=idx)

    assert result.equity_curve.equals(expected_equity)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(15.0)
    assert result.metrics["total_return"] == pytest.approx(15.0 / 100000.0)
    assert result.metrics["max_drawdown"] == pytest.approx(abs((100005.0 / 100010.0) - 1.0))
    assert result.metrics["sharpe"] != pytest.approx(0.0)

def test_backtest_time_signal_controls_exposure_path(tmp_path):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 121.0, 118.58, 124.509]}, index=idx)

    cfg = StrategyConfig(name="time-signal", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [0.0, 1.0, 1.0, 0.0, 0.0])

    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, strategy)

    expected_equity = pd.Series(
        [100000.0, 100000.0, 110000.0, 107800.0, 107800.0],
        index=idx,
    )

    pd.testing.assert_series_equal(result.equity_curve, expected_equity)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(7800.0)
    assert result.metrics["total_return"] == pytest.approx(0.078)
    assert result.metrics["max_drawdown"] == pytest.approx(0.02)
    assert result.metrics["sharpe"] != pytest.approx(0.0)

