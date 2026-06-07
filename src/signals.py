from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .regime import detect_regime, suggest_strategy


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


def _atr_like(df: pd.DataFrame, window: int = 14) -> float:
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

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
) -> TradeSetup:
    direction = _validate_direction(direction)

    if len(data) < atr_window + 1:
        raise ValueError("Dati insufficienti per generare un setup")

    last = data.iloc[-1]
    entry = float(last["close"])
    atr = _atr_like(data, window=atr_window)

    volume = data[volume_col] if volume_col and volume_col in data.columns else None
    regime_df = detect_regime(data["close"], volume=volume)
    regime = str(regime_df["regime"].iloc[-1])
    strategy_tag = suggest_strategy(regime)

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
    note = f"Regime={regime}; strategia suggerita={strategy_tag}; ATR={atr_window}x{stop_atr_mult}"

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
    )


def generate_setups(
    market: str,
    strategies: list[str],
    data_live: pd.DataFrame,
    ticker: str,
    *,
    top_n: int = 3,
) -> list[TradeSetup]:
    direction = "long" if market.lower() != "bear" else "short"
    setups: list[TradeSetup] = []

    for strategy in strategies[:top_n]:
        setup = generate_setup(
            data_live,
            ticker=ticker,
            direction=direction,
        )
        setups.append(
            TradeSetup(
                ticker=setup.ticker,
                direction=setup.direction,
                entry=setup.entry,
                stop=setup.stop,
                target=setup.target,
                risk_reward=setup.risk_reward,
                strategy_tag=strategy,
                regime=setup.regime,
                note=setup.note,
            )
        )

    return setups
