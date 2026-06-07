import numpy as np
from typing import Sequence


def win_rate(returns: Sequence[float]) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    return float(np.sum(r > 0) / len(r))


def profit_factor(returns: Sequence[float]) -> float:
    r = np.asarray(returns, dtype=float)
    gross_profit = np.sum(r[r > 0])
    gross_loss = abs(np.sum(r[r < 0]))
    if gross_loss == 0:
        return float("inf")
    return float(gross_profit / gross_loss)


def avg_risk_reward(returns: Sequence[float]) -> float:
    r = np.asarray(returns, dtype=float)
    wins = r[r > 0]
    losses = abs(r[r < 0])
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
    if avg_loss == 0:
        return float("inf")
    return float(avg_win / avg_loss)


def sharpe_ratio(returns: Sequence[float], risk_free: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    excess = r - risk_free
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std)


def sortino_ratio(returns: Sequence[float], risk_free: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    excess = r - risk_free
    downside = excess[excess < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std)


def risk_summary(returns: Sequence[float], risk_free: float = 0.0) -> dict:
    return {
        "n_trades": len(returns),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "avg_rr": avg_risk_reward(returns),
        "sharpe": sharpe_ratio(returns, risk_free),
        "sortino": sortino_ratio(returns, risk_free),
    }
