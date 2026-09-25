from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.regime import regime_decision


@dataclass(frozen=True)
class TradeSetup:
    ticker: str
    direction: str
    entry: float
    stop: float
    target: float
    risk_reward: float
    strategy_tag: str
    regime: str
    note: str
    status: str = "open"
    open_date: str = ""
    position_size: float = 0.0


def _atr_like(df: pd.DataFrame, window: int = 14) -> float:
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    if pd.isna(atr) or atr <= 0:
        raise ValueError("ATR non valido per generare il setup")
    return float(atr)


def _validate_direction(direction: str) -> str:
    d = direction.lower().strip()
    if d not in {"long", "short"}:
        raise ValueError("direction deve essere 'long' o 'short'")
    return d


def generate_setup(
    data: pd.DataFrame,
    ticker: str,
    direction: str,
    *,
    atr_window: int = 14,
    stop_atr_mult: float = 1.5,
    target_rr: float = 2.0,
    volume_col: Optional[str] = "volume",
    regime_name: str = "unknown",
) -> TradeSetup:
    direction = _validate_direction(direction)
    if len(data) < atr_window + 1:
        raise ValueError("Dati insufficienti per generare un setup")

    last = data.iloc[-1]
    entry = float(last["close"])
    atr = _atr_like(data, window=atr_window)
    decision = regime_decision(regime_name)
    regime = decision["regime"]
    strategy_tag = decision["strategy"]

    if direction == "long":
        stop = entry - stop_atr_mult * atr
        risk = entry - stop
        target = entry + (target_rr * risk)
    else:
        stop = entry + stop_atr_mult * atr
        risk = stop - entry
        target = entry - (target_rr * risk)

    if risk <= 0:
        raise ValueError("Risk non valido per generare il setup")

    rr = abs(target - entry) / abs(entry - stop)
    note = f'ATR={atr_window}x{stop_atr_mult}; volume_col={volume_col}; allocation={decision["allocation"]}'

    return TradeSetup(
        ticker=ticker,
        direction=direction,
        entry=float(entry),
        stop=float(stop),
        target=float(target),
        risk_reward=float(rr),
        strategy_tag=strategy_tag,
        regime=regime,
        note=note,
        status="open",
        open_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
        position_size=1.0,
    )
