from __future__ import annotations

from typing import Iterable

import pandas as pd


def _ensure_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if df.empty:
        raise ValueError(f"{name} is empty")
    return df.copy()


def _require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str], name: str) -> pd.DataFrame:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df


def validate_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_dataframe(df, "trade_log")
    required = [
        "timestamp",
        "symbol",
        "side",
        "quantity",
        "entry_price",
        "exit_price",
        "pnl",
        "return_pct",
        "pnl_realized",
        "pnl_unrealized",
    ]
    _require_columns(df, required, "trade_log")
    df = _coerce_numeric(
        df,
        ["quantity", "entry_price", "exit_price", "pnl", "return_pct", "pnl_realized", "pnl_unrealized"],
        "trade_log",
    )
    if (df["quantity"] <= 0).any():
        raise ValueError("trade_log contains non-positive quantity")
    return df


def validate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_dataframe(df, "metrics")
    required = ["n_trades", "total_return", "win_rate"]
    _require_columns(df, required, "metrics")
    df = _coerce_numeric(df, required, "metrics")
    if len(df) != 1:
        raise ValueError("metrics must contain exactly one row")
    if df.loc[df.index[0], "n_trades"] < 0:
        raise ValueError("metrics contains negative n_trades")
    return df


def validate_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_dataframe(df, "risk_summary")
    required = ["n_trades"]
    _require_columns(df, required, "risk_summary")
    df = _coerce_numeric(df, required, "risk_summary")
    if len(df) != 1:
        raise ValueError("risk_summary must contain exactly one row")
    if df.loc[df.index[0], "n_trades"] < 0:
        raise ValueError("risk_summary contains negative n_trades")
    return df


def validate_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_dataframe(df, "equity_curve")
    required = ["equity"]
    _require_columns(df, required, "equity_curve")
    df = _coerce_numeric(df, ["equity"], "equity_curve")
    return df
