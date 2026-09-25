import json

import numpy as np
import pandas as pd

from src.backtest import BacktestEngine
from src.walk_forward import WalkForwardEvaluator
from experiments.audit_experiment import gross_net_report, trade_log_artifact


class ConstantTimeStrategy:
    def __init__(self, lookback):
        self.config = type("Config", (), {"name": "dummy", "universe": ["TEST"]})()
        self.lookback = lookback

    def generate_time_series_signals(self, history):
        # Alternate long/short exposure so the audit test necessarily exercises
        # transaction costs and slippage instead of remaining at constant exposure.
        values = np.where(np.arange(len(history)) % 2 == 0, 1.0, -1.0)
        return pd.Series(values, index=history.index)


def _prices(periods=12):
    idx = pd.date_range("2024-01-01", periods=periods, freq="B")
    return pd.DataFrame({"close": np.linspace(100.0, 111.0, periods)}, index=idx)


def test_gross_net_replay_has_identical_windows_and_net_cost_drag(tmp_path):
    prices = _prices()

    def strategy_factory(train, params):
        return ConstantTimeStrategy(params["lookback"])

    def net_factory():
        return BacktestEngine(output_dir=tmp_path / "net", transaction_cost_bps=5, slippage_bps=2)

    def gross_factory():
        return BacktestEngine(output_dir=tmp_path / "gross", transaction_cost_bps=0, slippage_bps=0)

    result = WalkForwardEvaluator(5, 2, 2, 2).evaluate(
        prices, strategy_factory, net_factory, {"lookback": [2, 4]}
    )
    windows = result.windows
    net_results = []
    gross_results = []
    evaluator = WalkForwardEvaluator(5, 2, 2, 2)
    for i, window in enumerate(evaluator.windows(prices.index)):
        train = prices.loc[window.train_start:window.train_end]
        test = prices.loc[window.test_start:window.test_end]
        history = pd.concat([train, prices.loc[window.validation_start:window.validation_end], test])
        frozen = strategy_factory(train, result.selected_parameters[i])
        net_results.append(evaluator._run_candidate(frozen, history, test.index, test, net_factory))
        gross_results.append(evaluator._run_candidate(frozen, history, test.index, test, gross_factory))

    comparison = gross_net_report(net_results, gross_results)
    assert len(comparison["windows"]) == 2
    assert comparison["aggregate"]["gross_total_return"] > comparison["aggregate"]["net_total_return"]
    assert comparison["aggregate"]["friction_drag_total_return"] > 0
    assert comparison["aggregate"]["friction_drag_sharpe"] >= 0


def test_trade_log_artifact_preserves_window_and_parameter_provenance(tmp_path):
    prices = _prices()
    evaluator = WalkForwardEvaluator(5, 2, 2, 2)

    def strategy_factory(train, params):
        return ConstantTimeStrategy(params["lookback"])

    def engine_factory():
        return BacktestEngine(output_dir=tmp_path, transaction_cost_bps=1, slippage_bps=1)

    result = evaluator.evaluate(prices, strategy_factory, engine_factory, {"lookback": [2]})
    frame = trade_log_artifact(result.test_results, evaluator.windows(prices.index), result.selected_parameters, "net")

    assert set(frame["window"]) == {1, 2}
    assert frame["cost_model"].eq("net").all()
    assert frame["phase"].eq("oos").all()
    assert frame["selected_parameters"].map(json.loads).map(lambda x: x["lookback"]).tolist() == [2, 2]
