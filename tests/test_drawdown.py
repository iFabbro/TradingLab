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
