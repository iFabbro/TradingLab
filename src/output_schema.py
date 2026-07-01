from __future__ import annotations

from pathlib import Path

import pandas as pd


COMMON_TRADE_COLUMNS = [
    "ticker",
    "strategy_tag",
    "status",
    "side",
]

TRADE_LOG_COLUMNS = [
    "ticker",
    "strategy_tag",
    "status",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "side",
    "pnl",
    "return_pct",
    "bars",
]

OPEN_TRADES_COLUMNS = [
    "ticker",
    "side",
    "entry_price",
    "stop_price",
    "target_price",
    "risk_reward",
    "strategy_tag",
    "regime",
    "note",
    "status",
    "entry_date",
    "position_size",
]

EQUITY_CURVE_COLUMNS = [
    "date",
    "equity",
]

METRICS_COLUMNS = [
    "total_return",
    "cagr",
    "sharpe",
    "max_drawdown",
    "win_rate",
    "n_trades",
    "warning_nonpositive_sharpe",
]

MACRO_SNAPSHOT_COLUMNS = [
    "date",
    "rate",
    "inflation",
    "gdp",
    "rate_trend",
    "inflation_yoy",
    "inflation_trend",
    "gdp_trend",
    "macro_regime",
]


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: colonne mancanti: {missing}")


def validate_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _require_columns(out, TRADE_LOG_COLUMNS, "trade_log")
    return out[TRADE_LOG_COLUMNS]


def validate_open_trades(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _require_columns(out, OPEN_TRADES_COLUMNS, "open_trades")
    return out[OPEN_TRADES_COLUMNS]


def validate_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _require_columns(out, EQUITY_CURVE_COLUMNS, "equity_curve")
    return out[EQUITY_CURVE_COLUMNS]


def validate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _require_columns(out, METRICS_COLUMNS, "metrics")
    return out[METRICS_COLUMNS]


def validate_macro_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _require_columns(out, MACRO_SNAPSHOT_COLUMNS, "macro_snapshot")
    return out[MACRO_SNAPSHOT_COLUMNS]


def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
