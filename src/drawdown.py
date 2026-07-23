import numpy as np
from typing import Sequence


def equity_curve(returns: Sequence[float], initial: float = 1.0) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return initial * np.cumprod(1 + r)


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / peak


def max_drawdown(returns: Sequence[float]) -> float:
    eq = equity_curve(returns)
    dd = drawdown_series(eq)
    return float(np.min(dd))


def max_drawdown_duration(returns: Sequence[float]) -> int:
    eq = equity_curve(returns)
    peak = np.maximum.accumulate(eq)
    underwater = eq < peak
    max_dur = 0
    cur_dur = 0
    for u in underwater:
        if u:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return max_dur


def calmar_ratio(returns: Sequence[float]) -> float:
    ann_return = float(np.mean(returns)) * len(returns)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float("inf")
    return ann_return / mdd


def drawdown_summary(returns: Sequence[float]) -> dict:
    eq = equity_curve(returns)
    dd = drawdown_series(eq)
    max_dd = float(np.min(dd))
    return {
        "max_drawdown": max_dd,
        "avg_drawdown": float(np.mean(dd[dd < 0])) if np.any(dd < 0) else 0.0,
        "max_drawdown_duration": max_drawdown_duration(returns),
        "calmar_ratio": calmar_ratio(returns),
        "final_equity": float(eq[-1]) if len(eq) > 0 else 1.0,
        "warning_high_drawdown": bool(max_dd <= -0.2),
    }
