import pytest
import numpy as np
from src.risk import win_rate, profit_factor, sharpe_ratio, sortino_ratio, risk_summary

RETURNS = [0.05, -0.02, 0.03, -0.01, 0.04, -0.03, 0.02, 0.01]

def test_win_rate():
    assert win_rate(RETURNS) == pytest.approx(5 / 8)

def test_profit_factor():
    pf = profit_factor(RETURNS)
    assert pf > 1.0

def test_sharpe():
    s = sharpe_ratio(RETURNS)
    assert isinstance(s, float)

def test_sortino():
    s = sortino_ratio(RETURNS)
    assert s > 0

def test_risk_summary_keys():
    summary = risk_summary(RETURNS)
    for k in ["n_trades", "win_rate", "profit_factor", "avg_rr", "sharpe", "sortino"]:
        assert k in summary

def test_empty_returns():
    assert win_rate([]) == 0.0

def test_risk_summary_includes_drawdown_metrics():
    summary = risk_summary([0.05, -0.1, 0.03, -0.02, 0.04])
    for k in ["max_drawdown", "avg_drawdown", "max_drawdown_duration", "calmar_ratio", "final_equity"]:
        assert k in summary
