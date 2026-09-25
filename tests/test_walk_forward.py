import pandas as pd
import pytest

from src.strategies import MomentumStrategy, StrategyConfig
from src.walk_forward import WalkForwardEvaluator


def make_prices(n=60):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": range(100, 100 + n)}, index=idx)


def factory(train):
    assert len(train) > 0
    return MomentumStrategy(StrategyConfig(name="momentum", universe=["DEMO"], lookback=5))


def test_windows_are_strictly_chronological():
    prices = make_prices(40)
    evaluator = WalkForwardEvaluator(train_size=20, validation_size=10, test_size=10)
    windows = evaluator.windows(prices.index)
    assert len(windows) == 1
    w = windows[0]
    assert w.train_end < w.validation_start
    assert w.validation_end < w.test_start


def test_factory_only_receives_training_slice():
    prices = make_prices(60)
    seen = []

    def tracking_factory(train):
        seen.append((train.index[0], train.index[-1]))
        return factory(train)

    result = WalkForwardEvaluator(30, 10, 10).evaluate(prices, tracking_factory)
    assert len(seen) == len(result.windows)
    for i, (_, train_end) in enumerate(seen):
        assert train_end < result.windows.iloc[i]["validation_start"]
        assert train_end < result.windows.iloc[i]["test_start"]
    assert all(len(r.metrics) > 0 for r in result.test_results)


def test_parameter_selection_uses_validation_only():
    prices = make_prices(80)
    calls = []

    class Candidate:
        def __init__(self, train, params):
            self.params = params
            self.config = None
            calls.append((params["exposure"], train.index[-1]))

        def generate_time_series_signals(self, history):
            return pd.Series(self.params["exposure"], index=history.index, dtype=float)

    def candidate_factory(train, params):
        return Candidate(train, params)

    result = WalkForwardEvaluator(40, 20, 20).evaluate(
        prices,
        candidate_factory,
        parameter_grid={"exposure": [0.0, 1.0]},
        selection_metric="total_return",
    )

    assert result.selected_parameters[0] == {"exposure": 1.0}
    assert len(result.selection_results[0]) == 2
    assert result.selection_results[0]["status"].eq("ok").all()
    assert result.windows.iloc[0]["selected_parameters"] == {"exposure": 1.0}

    validation_start = result.windows.iloc[0]["validation_start"]
    test_start = result.windows.iloc[0]["test_start"]
    assert all(train_end < validation_start for _, train_end in calls)
    assert all(train_end < test_start for _, train_end in calls)


def test_selection_grid_rejects_empty_options():
    with pytest.raises(ValueError):
        WalkForwardEvaluator(10, 5, 5)._parameter_combinations({"lookback": []})


def test_insufficient_data_returns_no_windows():
    evaluator = WalkForwardEvaluator(30, 10, 10)
    assert evaluator.windows(make_prices(49).index) == []


def test_invalid_window_sizes_rejected():
    with pytest.raises(ValueError):
        WalkForwardEvaluator(0, 10, 10)
