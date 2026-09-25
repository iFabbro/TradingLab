import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestEngine
from src.strategies import BaseStrategy, StrategyConfig


class TimeSignalStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig, values: list[float]) -> None:
        super().__init__(config)
        self.values = values

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(self.values, index=prices.index, dtype=float)


def test_signal_is_not_allowed_to_trade_on_same_bar(tmp_path):
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"close": [100.0, 200.0, 200.0]}, index=idx)
    strategy = TimeSignalStrategy(StrategyConfig(name="lookahead", universe=["DEMO"], lookback=2), [1.0, 1.0, 1.0])
    result = BacktestEngine(output_dir=tmp_path).run(prices, strategy)
    assert result.equity_curve.iloc[0] == pytest.approx(100000.0)
    assert result.equity_curve.iloc[1] == pytest.approx(200000.0)
    assert result.trade_log.loc[0, "entry_date"] == idx[0]


def test_multiple_round_trips_are_logged_and_reconcile_with_compounding_equity(tmp_path):
    idx = pd.date_range("2024-01-01", periods=7, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 110.0, 99.0, 99.0, 108.9, 108.9]}, index=idx)
    strategy = TimeSignalStrategy(StrategyConfig(name="round-trips", universe=["DEMO"], lookback=2), [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    result = BacktestEngine(output_dir=tmp_path).run(prices, strategy)
    assert len(result.trade_log) == 2
    assert result.trade_log.loc[0, "pnl"] == pytest.approx(10000.0)
    assert result.trade_log.loc[1, "pnl"] == pytest.approx(11000.0)
    assert result.trade_log.loc[0, "pnl"] + result.trade_log.loc[1, "pnl"] == pytest.approx(result.equity_curve.iloc[-1] - result.equity_curve.iloc[0])
    assert result.metrics["n_trades"] == 2
    assert result.metrics["total_return"] == pytest.approx(0.21)


def test_transaction_costs_and_slippage_reduce_equity(tmp_path):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 110.0, 100.0, 100.0]}, index=idx)
    strategy = TimeSignalStrategy(StrategyConfig(name="friction", universe=["DEMO"], lookback=2), [1.0, 1.0, 0.0, 0.0, 0.0])
    free = BacktestEngine(output_dir=tmp_path / "free").run(prices, strategy)
    net = BacktestEngine(output_dir=tmp_path / "net", transaction_cost_bps=10.0, slippage_bps=10.0).run(prices, strategy)
    assert net.equity_curve.iloc[-1] < free.equity_curve.iloc[-1]
    assert net.metrics["turnover"] == pytest.approx(2.0)


def test_exposure_is_clipped_to_safe_long_only_contract(tmp_path):
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"close": [100.0, 110.0, 120.0, 120.0]}, index=idx)
    strategy = TimeSignalStrategy(StrategyConfig(name="bounded", universe=["DEMO"], lookback=2), [-1.0, 1.5, 1.0, 0.0])
    result = BacktestEngine(output_dir=tmp_path).run(prices, strategy)
    assert result.equity_curve.iloc[0] == pytest.approx(100000.0)
    assert result.equity_curve.iloc[1] == pytest.approx(100000.0)
    assert result.equity_curve.iloc[2] == pytest.approx(120000.0)


def test_nonfinite_time_signal_is_rejected(tmp_path):
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)
    strategy = TimeSignalStrategy(StrategyConfig(name="invalid", universe=["DEMO"], lookback=2), [0.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="finite numeric"):
        BacktestEngine(output_dir=tmp_path).run(prices, strategy)


def test_invalid_price_data_is_rejected(tmp_path):
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    prices = pd.DataFrame({"close": [100.0, np.nan]}, index=idx)
    strategy = TimeSignalStrategy(StrategyConfig(name="invalid-price", universe=["DEMO"], lookback=2), [0.0, 0.0])
    with pytest.raises(ValueError, match="positive finite"):
        BacktestEngine(output_dir=tmp_path).run(prices, strategy)


def test_annualisation_uses_observed_bar_spacing(tmp_path):
    idx = pd.date_range("2024-01-01", periods=10, freq="7D")
    prices = pd.DataFrame({"close": np.linspace(100.0, 110.0, 10)}, index=idx)
    strategy = TimeSignalStrategy(StrategyConfig(name="weekly", universe=["DEMO"], lookback=2), [1.0] * 10)
    result = BacktestEngine(output_dir=tmp_path).run(prices, strategy)
    expected_annualisation = 365.25 / 7.0
    returns = result.equity_curve.pct_change().fillna(0.0)
    expected_sharpe = np.sqrt(expected_annualisation) * returns.mean() / returns.std(ddof=0)
    assert result.metrics["sharpe"] == pytest.approx(expected_sharpe)
