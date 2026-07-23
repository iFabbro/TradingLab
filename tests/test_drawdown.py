import pytest
import numpy as np
from src.drawdown import max_drawdown, max_drawdown_duration, drawdown_summary, equity_curve

RETURNS = [0.05, -0.10, 0.03, -0.02, 0.04]

def test_max_drawdown_negative():
    mdd = max_drawdown(RETURNS)
    assert mdd < 0

def test_max_drawdown_duration_positive():
    dur = max_drawdown_duration(RETURNS)
    assert dur == 4

def test_equity_curve_starts_above_initial():
    eq = equity_curve([0.1, 0.2])
    assert eq[0] > 1.0

def test_drawdown_summary_keys():
    summary = drawdown_summary(RETURNS)
    for k in ["max_drawdown", "avg_drawdown", "max_drawdown_duration", "calmar_ratio", "final_equity"]:
        assert k in summary


def test_flat_returns_have_zero_drawdown_and_infinite_calmar():
    summary = drawdown_summary([0.0, 0.0, 0.0])
    assert max_drawdown([0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert summary["max_drawdown"] == pytest.approx(0.0)
    assert summary["calmar_ratio"] == float("inf")


def test_drawdown_summary_can_warn_on_large_drawdown():
    summary = drawdown_summary([0.5, -0.5, 0.0])
    assert summary["max_drawdown"] <= -0.3333333333
