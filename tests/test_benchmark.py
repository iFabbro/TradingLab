import numpy as np
import pandas as pd
import pytest

from src.benchmark import BENCHMARK_ID, buy_and_hold_oos


def _prices(values):
    index = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame({"close": values}, index=index)


def _windows(prices, spans):
    idx = prices.index
    return pd.DataFrame([{"test_start": idx[a], "test_end": idx[b]} for a, b in spans])


def test_single_window_matches_hand_computation():
    prices = _prices([100.0, 110.0, 99.0, 108.9, 108.9])
    result = buy_and_hold_oos(prices, _windows(prices, [(0, 4)]), 5.0, 2.0)
    expected = np.prod([1 - 0.0007, 1.10, 0.90, 1.10, 1.0]) - 1.0
    assert result["benchmark"] == BENCHMARK_ID
    assert result["observations"] == 5
    assert result["total_return"] == pytest.approx(expected)


def test_zero_costs_equal_close_to_close_return_inside_window():
    prices = _prices([100.0, 101.0, 103.0, 102.0, 106.0, 107.0])
    result = buy_and_hold_oos(prices, _windows(prices, [(1, 5)]), 0.0, 0.0)
    assert result["total_return"] == pytest.approx(107.0 / 101.0 - 1.0)


def test_first_day_of_each_window_earns_nothing_and_cost_paid_once():
    prices = _prices([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
    windows = _windows(prices, [(0, 2), (3, 5)])
    free = buy_and_hold_oos(prices, windows, 0.0, 0.0)
    costly = buy_and_hold_oos(prices, windows, 5.0, 2.0)
    # the gap return between the two windows (104 -> 106) is never earned, as for a strategy
    assert free["total_return"] == pytest.approx((104 / 100) * (110 / 106) - 1.0)
    assert costly["total_return"] == pytest.approx((1 + free["total_return"]) * (1 - 0.0007) - 1.0, rel=1e-9)
    assert costly["entry_cost_bps"] == 7.0


def test_rejects_overlapping_windows_and_bad_inputs():
    prices = _prices([100.0 + i for i in range(10)])
    with pytest.raises(ValueError, match="overlap"):
        buy_and_hold_oos(prices, _windows(prices, [(0, 5), (4, 9)]), 5.0, 2.0)
    with pytest.raises(ValueError, match="no OOS windows"):
        buy_and_hold_oos(prices, _windows(prices, [])[0:0], 5.0, 2.0)
    with pytest.raises(ValueError, match="non-negative"):
        buy_and_hold_oos(prices, _windows(prices, [(0, 5)]), -1.0, 2.0)
    with pytest.raises(ValueError, match="close"):
        buy_and_hold_oos(prices.rename(columns={"close": "px"}), _windows(prices, [(0, 5)]), 5.0, 2.0)
