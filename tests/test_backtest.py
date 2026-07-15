import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtest import BacktestEngine
from src.output_schema import validate_metrics, validate_risk_summary, validate_trade_log
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
    assert result.trade_log.loc[0, "bars"] == 39
    assert result.metrics["n_trades"] == 1
    assert result.metrics["total_return"] == pytest.approx(20.0 / 100000.0)
    assert result.metrics["win_rate"] == pytest.approx(1.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(100020.0)
    assert (tmp_path / "trade_log.csv").exists()
    assert (tmp_path / "equity_curve.csv").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "risk_summary.csv").exists()

    output_dir = engine.output_dir
    trade_log = validate_trade_log(pd.read_csv(output_dir / "trade_log.csv"))
    metrics = validate_metrics(pd.read_csv(output_dir / "metrics.csv"))
    risk_summary = validate_risk_summary(pd.read_csv(output_dir / "risk_summary.csv"))
    equity_curve_csv = pd.read_csv(output_dir / "equity_curve.csv")

    assert trade_log.loc[0, "quantity"] == pytest.approx(1.0)
    assert trade_log.loc[0, "pnl"] == pytest.approx(result.trade_log.loc[0, "pnl"])
    assert trade_log.loc[0, "return_pct"] == pytest.approx(
        result.trade_log.loc[0, "return_pct"]
    )
    assert "pnl_realized" in trade_log.columns
    assert "pnl_unrealized" in trade_log.columns

    assert metrics.loc[0, "n_trades"] == result.metrics["n_trades"]
    assert metrics.loc[0, "total_return"] == pytest.approx(
        result.metrics["total_return"]
    )
    assert metrics.loc[0, "win_rate"] == pytest.approx(result.metrics["win_rate"])

    assert equity_curve_csv["equity"].iloc[-1] == pytest.approx(
        result.equity_curve.iloc[-1]
    )
    assert risk_summary.loc[0, "n_trades"] == 1


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
    assert result.trade_log.loc[0, "entry_date"] == idx[1]
    assert result.trade_log.loc[0, "entry_price"] == pytest.approx(110.0)
    assert result.trade_log.loc[0, "exit_date"] == idx[3]
    assert result.trade_log.loc[0, "exit_price"] == pytest.approx(118.58)
    assert result.trade_log.loc[0, "quantity"] == pytest.approx(100000.0 / 110.0)
    assert result.trade_log.loc[0, "bars"] == 2
    assert result.trade_log.loc[0, "pnl_realized"] == pytest.approx(7800.0)
    assert result.trade_log.loc[0, "pnl_unrealized"] == pytest.approx(0.0)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(7800.0)
    assert result.trade_log.loc[0, "return_pct"] == pytest.approx((118.58 / 110.0) - 1.0)
    assert result.metrics["n_trades"] == 1
    assert result.metrics["total_return"] == pytest.approx(0.078)
    assert result.metrics["max_drawdown"] == pytest.approx(0.02)
    assert result.metrics["sharpe"] != pytest.approx(0.0)


def test_backtest_time_signal_midpath_equity_matches_realized_trade_pnl(tmp_path):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"close": [100.0, 120.0, 108.0, 129.6, 129.6]}, index=idx)

    cfg = StrategyConfig(name="time-signal-midpath", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [0.0, 1.0, 1.0, 0.0, 0.0])

    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, strategy)

    expected_equity = pd.Series(
        [100000.0, 100000.0, 90000.0, 108000.0, 108000.0],
        index=idx,
    )

    pd.testing.assert_series_equal(result.equity_curve, expected_equity)
    assert result.trade_log.loc[0, "quantity"] == pytest.approx(100000.0 / 120.0)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(8000.0)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(result.equity_curve.iloc[-1] - result.equity_curve.iloc[0])
    assert result.metrics["total_return"] == pytest.approx(0.08)
    assert result.metrics["max_drawdown"] == pytest.approx(0.1)
    assert result.metrics["sharpe"] != pytest.approx(0.0)


def test_backtest_time_signal_losing_trade_sets_zero_win_rate(tmp_path):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"close": [100.0, 120.0, 108.0, 96.0, 96.0]}, index=idx)

    cfg = StrategyConfig(name="time-signal-loss", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [0.0, 1.0, 1.0, 0.0, 0.0])

    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, strategy)

    expected_equity = pd.Series(
        [100000.0, 100000.0, 90000.0, 80000.0, 80000.0],
        index=idx,
    )

    pd.testing.assert_series_equal(result.equity_curve, expected_equity)
    assert result.trade_log.loc[0, "pnl_realized"] == pytest.approx(-20000.0)
    assert result.trade_log.loc[0, "pnl_unrealized"] == pytest.approx(0.0)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(-20000.0)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(result.equity_curve.iloc[-1] - result.equity_curve.iloc[0])
    assert result.metrics["total_return"] == pytest.approx(-0.2)
    assert result.metrics["win_rate"] == pytest.approx(0.0)
    assert result.metrics["max_drawdown"] == pytest.approx(0.2)
    assert result.metrics["sharpe"] != pytest.approx(0.0)



def test_backtest_time_signal_all_zero_keeps_flat_equity_and_zero_win_rate(tmp_path):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=idx)

    cfg = StrategyConfig(name="time-signal-flat", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [0.0, 0.0, 0.0, 0.0, 0.0])

    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, strategy)

    expected_equity = pd.Series([100000.0, 100000.0, 100000.0, 100000.0, 100000.0], index=idx)

    pd.testing.assert_series_equal(result.equity_curve, expected_equity)
    assert result.trade_log.loc[0, "quantity"] == pytest.approx(0.0)
    assert result.trade_log.loc[0, "pnl_realized"] == pytest.approx(0.0)
    assert result.trade_log.loc[0, "pnl_unrealized"] == pytest.approx(0.0)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(0.0)
    assert result.metrics["total_return"] == pytest.approx(0.0)
    assert result.metrics["win_rate"] == pytest.approx(0.0)
    assert result.metrics["max_drawdown"] == pytest.approx(0.0)


def test_backtest_time_signal_all_one_tracks_full_price_path(tmp_path):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 105.0, 115.0]}, index=idx)

    cfg = StrategyConfig(name="time-signal-full", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [1.0, 1.0, 1.0, 1.0])

    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, strategy)

    expected_equity = pd.Series([100000.0, 110000.0, 105000.0, 115000.0], index=idx)

    pd.testing.assert_series_equal(result.equity_curve, expected_equity)
    assert result.trade_log.loc[0, "entry_date"] == idx[0]
    assert result.trade_log.loc[0, "exit_date"] == idx[-1]
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(15000.0)
    assert result.metrics["win_rate"] == pytest.approx(1.0)
    assert result.metrics["total_return"] == pytest.approx(0.15)


def test_backtest_time_signal_clips_above_one_to_full_exposure(tmp_path):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 105.0, 115.0]}, index=idx)

    cfg = StrategyConfig(name="time-signal-clipped-high", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [2.0, 2.0, 2.0, 2.0])

    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, strategy)

    expected_equity = pd.Series([100000.0, 110000.0, 105000.0, 115000.0], index=idx)

    pd.testing.assert_series_equal(result.equity_curve, expected_equity)
    assert result.trade_log.loc[0, "entry_date"] == idx[0]
    assert result.trade_log.loc[0, "exit_date"] == idx[-1]
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(15000.0)
    assert result.metrics["win_rate"] == pytest.approx(1.0)
    assert result.metrics["total_return"] == pytest.approx(0.15)


def test_backtest_time_signal_clips_negative_to_zero_exposure(tmp_path):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 105.0, 115.0]}, index=idx)

    cfg = StrategyConfig(name="time-signal-clipped-low", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [-1.0, -1.0, -1.0, -1.0])

    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, strategy)

    expected_equity = pd.Series([100000.0, 100000.0, 100000.0, 100000.0], index=idx)

    pd.testing.assert_series_equal(result.equity_curve, expected_equity)
    assert result.trade_log.loc[0, "quantity"] == pytest.approx(0.0)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(0.0)
    assert result.metrics["win_rate"] == pytest.approx(0.0)
    assert result.metrics["total_return"] == pytest.approx(0.0)


def test_backtest_time_signal_partial_exposure_scales_equity_path(tmp_path):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 99.0, 108.9]}, index=idx)

    cfg = StrategyConfig(name="time-signal-half", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [0.5, 0.5, 0.5, 0.5])

    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, strategy)

    expected_equity = pd.Series([100000.0, 105000.0, 99750.0, 104737.5], index=idx)

    pd.testing.assert_series_equal(result.equity_curve, expected_equity)
    assert result.trade_log.loc[0, "entry_date"] == idx[0]
    assert result.trade_log.loc[0, "exit_date"] == idx[-1]
    assert result.trade_log.loc[0, "quantity"] == pytest.approx(100000.0 / 100.0)
    assert result.trade_log.loc[0, "pnl_realized"] == pytest.approx(8900.0)
    assert result.trade_log.loc[0, "pnl_unrealized"] == pytest.approx(0.0)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(8900.0)
    assert result.trade_log.loc[0, "bars"] == 3
    assert result.metrics["win_rate"] == pytest.approx(1.0)


def test_backtest_time_signal_break_even_trade_keeps_zero_win_rate(tmp_path):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"close": [100.0, 120.0, 120.0, 120.0, 120.0]}, index=idx)

    cfg = StrategyConfig(name="time-signal-break-even", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [0.0, 1.0, 1.0, 0.0, 0.0])

    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, strategy)

    expected_equity = pd.Series(
        [100000.0, 100000.0, 100000.0, 100000.0, 100000.0],
        index=idx,
    )

    pd.testing.assert_series_equal(result.equity_curve, expected_equity)
    assert result.trade_log.loc[0, "pnl_realized"] == pytest.approx(0.0)
    assert result.trade_log.loc[0, "pnl_unrealized"] == pytest.approx(0.0)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(0.0)
    assert result.trade_log.loc[0, "bars"] == 2
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(result.equity_curve.iloc[-1] - result.equity_curve.iloc[0])
    assert result.metrics["total_return"] == pytest.approx(0.0)
    assert result.metrics["win_rate"] == pytest.approx(0.0)
    assert result.metrics["max_drawdown"] == pytest.approx(0.0)


def test_backtest_time_signal_partial_loss_scales_drawdown_and_return(tmp_path):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"close": [100.0, 120.0, 108.0, 96.0, 96.0]}, index=idx)

    cfg = StrategyConfig(name="time-signal-half-loss", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [0.0, 0.5, 0.5, 0.0, 0.0])

    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, strategy)

    expected_equity = pd.Series(
        [100000.0, 100000.0, 95000.0, 89722.22222222222, 89722.22222222222],
        index=idx,
    )

    pd.testing.assert_series_equal(result.equity_curve, expected_equity)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(-20000.0)
    assert result.trade_log.loc[0, "exit_date"] == idx[3]
    assert result.metrics["total_return"] == pytest.approx(-0.10277777777777777)
    assert result.metrics["win_rate"] == pytest.approx(0.0)
    assert result.metrics["max_drawdown"] == pytest.approx(0.10277777777777777)


def test_backtest_time_signal_mixed_path_preserves_zero_win_rate_and_drawdown(tmp_path):
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    prices = pd.DataFrame({"close": [100.0, 120.0, 108.0, 118.8, 96.0, 96.0]}, index=idx)

    cfg = StrategyConfig(name="time-signal-mixed-simple", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [0.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, strategy)

    expected_equity = pd.Series(
        [100000.0, 100000.0, 90000.0, 99000.0, 80000.0, 80000.0],
        index=idx,
    )

    pd.testing.assert_series_equal(result.equity_curve, expected_equity)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(-20000.0)
    assert result.trade_log.loc[0, "exit_date"] == idx[4]
    assert result.metrics["total_return"] == pytest.approx(-0.2)
    assert result.metrics["win_rate"] == pytest.approx(0.0)
    assert result.metrics["max_drawdown"] == pytest.approx(0.2)


def test_backtest_equity_curve_csv_has_canonical_columns(tmp_path):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 105.0, 115.0]}, index=idx)

    cfg = StrategyConfig(name="equity-curve-schema", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [1.0, 1.0, 1.0, 1.0])

    engine = BacktestEngine(output_dir=tmp_path)
    engine.run(prices, strategy)

    equity_csv = pd.read_csv(tmp_path / "equity_curve.csv")
    assert list(equity_csv.columns) == ["date", "equity"]
    assert equity_csv.iloc[0]["equity"] == pytest.approx(100000.0)
    assert equity_csv.iloc[-1]["equity"] == pytest.approx(115000.0)


def test_backtest_trade_log_csv_has_canonical_columns(tmp_path):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 105.0, 115.0]}, index=idx)

    cfg = StrategyConfig(name="trade-log-schema", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [1.0, 1.0, 1.0, 1.0])

    engine = BacktestEngine(output_dir=tmp_path)
    engine.run(prices, strategy)

    trade_log_csv = pd.read_csv(tmp_path / "trade_log.csv")
    assert list(trade_log_csv.columns) == [
        "ticker",
        "strategy_tag",
        "status",
        "entry_date",
        "entry_price",
        "exit_date",
        "exit_price",
        "side",
        "quantity",
        "pnl_realized",
        "pnl_unrealized",
        "pnl",
        "return_pct",
        "bars",
    ]




def test_backtest_time_signal_trade_log_is_coherent_with_equity_delta(tmp_path):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"close": [100.0, 120.0, 108.0, 96.0, 96.0]}, index=idx)

    cfg = StrategyConfig(name="time-signal-coherent", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [0.0, 1.0, 1.0, 0.0, 0.0])

    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, strategy)

    trade = result.trade_log.loc[0]
    assert trade["entry_date"] == idx[1]
    assert trade["quantity"] == pytest.approx(100000.0 / 120.0)
    assert trade["pnl_realized"] == pytest.approx(trade["pnl"])
    assert trade["pnl_unrealized"] == pytest.approx(0.0)
    assert trade["pnl"] == pytest.approx(result.equity_curve.iloc[-1] - result.equity_curve.iloc[0])
    assert trade["pnl"] == pytest.approx(-20000.0)
def test_backtest_metrics_csv_has_canonical_columns(tmp_path):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 105.0, 115.0]}, index=idx)

    cfg = StrategyConfig(name="metrics-schema", universe=["close"], lookback=2)
    strategy = TimeSignalStrategy(cfg, [1.0, 1.0, 1.0, 1.0])

    engine = BacktestEngine(output_dir=tmp_path)
    engine.run(prices, strategy)

    metrics_csv = pd.read_csv(tmp_path / "metrics.csv")
    assert list(metrics_csv.columns) == [
        "total_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "n_trades",
        "warning_nonpositive_sharpe",
    ]

def test_backtest_winning_trade_validates_pnl_fields(prices, tmp_path):
    cfg = StrategyConfig(name="dummy", universe=list(prices.columns), lookback=5)
    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices, DummyStrategy(cfg))

    assert len(result.trade_log) == 1
    assert result.trade_log.loc[0, "quantity"] == pytest.approx(1.0)
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(20.0)
    assert result.trade_log.loc[0, "pnl_realized"] == pytest.approx(20.0)
    assert result.trade_log.loc[0, "pnl_unrealized"] == pytest.approx(0.0)
    assert result.trade_log.loc[0, "return_pct"] == pytest.approx(0.2)

    trade_log_csv = validate_trade_log(pd.read_csv(tmp_path / "trade_log.csv"))
    assert trade_log_csv.loc[0, "pnl"] == pytest.approx(20.0)
    assert trade_log_csv.loc[0, "pnl_realized"] == pytest.approx(20.0)
    assert trade_log_csv.loc[0, "pnl_unrealized"] == pytest.approx(0.0)
    assert trade_log_csv.loc[0, "return_pct"] == pytest.approx(0.2)

def test_backtest_losing_trade_validates_pnl_fields(tmp_path):
    idx = pd.date_range("2024-01-01", periods=21, freq="D")
    prices_down = pd.DataFrame(
        {
            "close": np.linspace(120, 100, len(idx)),
        },
        index=idx,
    )

    cfg = StrategyConfig(name="dummy", universe=list(prices_down.columns), lookback=5)
    engine = BacktestEngine(output_dir=tmp_path)
    result = engine.run(prices_down, DummyStrategy(cfg))

    assert len(result.trade_log) == 1
    assert result.trade_log.loc[0, "quantity"] == pytest.approx(1.0)
    assert result.trade_log.loc[0, "pnl"] < 0
    assert result.trade_log.loc[0, "pnl_realized"] < 0
    assert result.trade_log.loc[0, "pnl_unrealized"] == pytest.approx(0.0)
    assert result.trade_log.loc[0, "return_pct"] < 0

    trade_log_csv = validate_trade_log(pd.read_csv(tmp_path / "trade_log.csv"))
    assert trade_log_csv.loc[0, "pnl"] < 0
    assert trade_log_csv.loc[0, "pnl_realized"] < 0
    assert trade_log_csv.loc[0, "pnl_unrealized"] == pytest.approx(0.0)
    assert trade_log_csv.loc[0, "return_pct"] < 0


def test_backtest_outputs_time_signal_branch():
    import pandas as pd
    import pytest
    from src.backtest import (
        BacktestEngine,
        validate_trade_log,
        validate_metrics,
        validate_risk_summary,
    )

    class FakeTimeSignalStrategy:
        def __init__(self):
            self.config = type("Cfg", (), {"name": "fake_time_signal"})()

        def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
            return pd.Series([1.0, 1.0, 0.0, 0.0, 0.0], index=prices.index, dtype=float)

    prices = pd.DataFrame(
        {"close": [100.0, 110.0, 120.0, 115.0, 118.0]},
        index=pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        ),
    )

    engine = BacktestEngine()
    result = engine.run(prices, FakeTimeSignalStrategy())

    output_dir = engine.output_dir
    trade_log = validate_trade_log(pd.read_csv(output_dir / "trade_log.csv"))
    metrics = validate_metrics(pd.read_csv(output_dir / "metrics.csv"))
    risk_summary = validate_risk_summary(pd.read_csv(output_dir / "risk_summary.csv"))
    equity_curve_csv = pd.read_csv(output_dir / "equity_curve.csv")

    assert trade_log.loc[0, "strategy_tag"] == "fake_time_signal"
    assert trade_log.loc[0, "quantity"] == pytest.approx(result.trade_log.loc[0, "quantity"])
    assert trade_log.loc[0, "pnl"] == pytest.approx(result.trade_log.loc[0, "pnl"])
    assert trade_log.loc[0, "return_pct"] == pytest.approx(result.trade_log.loc[0, "return_pct"])
    assert trade_log.loc[0, "entry_price"] == pytest.approx(100.0)
    assert trade_log.loc[0, "exit_price"] == pytest.approx(120.0)
    assert trade_log.loc[0, "bars"] == 2
    assert "pnl_realized" in trade_log.columns
    assert "pnl_unrealized" in trade_log.columns

    assert metrics.loc[0, "n_trades"] == result.metrics["n_trades"]
    assert metrics.loc[0, "total_return"] == pytest.approx(result.metrics["total_return"])
    assert metrics.loc[0, "win_rate"] == pytest.approx(result.metrics["win_rate"])

    assert equity_curve_csv["equity"].iloc[-1] == pytest.approx(result.equity_curve.iloc[-1])
    assert risk_summary.loc[0, "n_trades"] == 1


def test_backtest_metrics_positive_time_signal_branch():
    import pandas as pd
    import pytest
    from src.backtest import BacktestEngine, validate_metrics, validate_risk_summary

    class FakeWinningTimeSignalStrategy:
        def __init__(self):
            self.config = type("Cfg", (), {"name": "fake_winning_time_signal"})()

        def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
            return pd.Series([1.0, 1.0, 1.0, 0.0, 0.0], index=prices.index, dtype=float)

    prices = pd.DataFrame(
        {"close": [100.0, 110.0, 120.0, 130.0, 130.0]},
        index=pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        ),
    )

    engine = BacktestEngine()
    result = engine.run(prices, FakeWinningTimeSignalStrategy())

    output_dir = engine.output_dir
    metrics = validate_metrics(pd.read_csv(output_dir / "metrics.csv"))
    risk_summary = validate_risk_summary(pd.read_csv(output_dir / "risk_summary.csv"))

    assert metrics.loc[0, "n_trades"] == 1
    assert metrics.loc[0, "total_return"] > 0
    assert metrics.loc[0, "total_return"] == pytest.approx(result.metrics["total_return"])
    assert metrics.loc[0, "win_rate"] == pytest.approx(1.0)
    assert metrics.loc[0, "win_rate"] == pytest.approx(result.metrics["win_rate"])
    assert metrics.loc[0, "max_drawdown"] == pytest.approx(0.0)
    assert bool(metrics.loc[0, "warning_nonpositive_sharpe"]) is False

    assert risk_summary.loc[0, "n_trades"] == 1
    assert risk_summary.loc[0, "win_rate"] == pytest.approx(1.0)
    assert risk_summary.loc[0, "sharpe"] == pytest.approx(result.metrics["sharpe"])
