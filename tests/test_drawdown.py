import pytest
import numpy as np
from src.drawdown import (
    max_drawdown,
    max_drawdown_duration,
    drawdown_summary,
    equity_curve,
    drawdown_periods,
    max_recovery_time,
    drawdown_suggestions,
)

RETURNS = [0.05, -0.10, 0.03, -0.02, 0.04]
ALL_POSITIVE = [0.01, 0.02, 0.03]
FLAT = [0.0, 0.0, 0.0]


# --- test esistenti ---

def test_max_drawdown_negative():
    assert max_drawdown(RETURNS) < 0

def test_max_drawdown_duration_positive():
    assert max_drawdown_duration(RETURNS) >= 0

def test_equity_curve_starts_above_initial():
    eq = equity_curve([0.1, 0.2])
    assert eq[0] > 1.0

def test_drawdown_summary_keys():
    summary = drawdown_summary(RETURNS)
    for k in ["max_drawdown", "avg_drawdown", "max_drawdown_duration", "calmar_ratio", "final_equity"]:
        assert k in summary


# --- nuovi test ---

def test_no_drawdown_all_positive():
    assert max_drawdown(ALL_POSITIVE) == 0.0

def test_no_drawdown_flat():
    assert max_drawdown(FLAT) == 0.0

def test_drawdown_periods_returns_list():
    periods = drawdown_periods(RETURNS)
    assert isinstance(periods, list)
    assert len(periods) > 0

def test_drawdown_period_keys():
    periods = drawdown_periods(RETURNS)
    for p in periods:
        for k in ["start", "trough", "end", "depth", "duration", "recovery"]:
            assert k in p

def test_drawdown_period_depth_negative():
    periods = drawdown_periods(RETURNS)
    for p in periods:
        assert p["depth"] < 0

def test_drawdown_period_duration_positive():
    periods = drawdown_periods(RETURNS)
    for p in periods:
        assert p["duration"] >= 1

def test_no_periods_all_positive():
    periods = drawdown_periods(ALL_POSITIVE)
    assert periods == []

def test_max_recovery_time_none_when_no_dd():
    assert max_recovery_time(ALL_POSITIVE) is None

def test_max_recovery_time_positive():
    # serie con drawdown recuperato
    r = [0.1, -0.2, 0.1, 0.1, 0.1]
    rt = max_recovery_time(r)
    assert rt is not None and rt > 0

def test_drawdown_suggestions_returns_list():
    suggestions = drawdown_suggestions(RETURNS)
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0

def test_drawdown_suggestions_strings():
    suggestions = drawdown_suggestions(RETURNS)
    for s in suggestions:
        assert isinstance(s, str)
