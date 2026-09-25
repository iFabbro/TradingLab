from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.output_schema import (
    validate_equity_curve,
    validate_metrics,
    validate_risk_summary,
    validate_trade_log,
)


@dataclass
class BacktestResult:
    trade_log: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict


class BacktestEngine:
    """Long-only close-to-close backtester.

    A signal at timestamp t is assumed to be known at the close of t and
    therefore affects returns starting at t+1. This prevents using the
    current bar's return with information generated from that same bar.
    """

    def __init__(self, initial_capital: float = 100000.0, output_dir: str | Path = "data/backtests", strategy_tag: str = "backtest", ticker: str = "") -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        self.initial_capital = float(initial_capital)
        self.output_dir = Path(output_dir)
        self.strategy_tag = strategy_tag
        self.ticker = ticker
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, prices: pd.DataFrame, strategy) -> BacktestResult:
        prices = self._validate_prices(prices)
        signals = strategy.generate_signals(prices)

        is_time_signal = isinstance(signals, pd.Series) and signals.index.equals(prices.index)
        if is_time_signal:
            exposure = self._normalise_exposure(signals)
            equity_curve = self._build_equity_curve(prices["close"], exposure)
            trade_log = self._build_trade_log(prices, strategy, exposure, equity_curve)
        else:
            equity_curve, trade_log = self._legacy_snapshot_result(prices, strategy, signals)

        metrics = self._metrics(equity_curve, trade_log)
        self._save_outputs(trade_log, equity_curve, metrics)
        return BacktestResult(trade_log=trade_log, equity_curve=equity_curve, metrics=metrics)

    @staticmethod
    def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            raise ValueError("prices cannot be empty")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("prices index must be a DatetimeIndex")
        prices = prices.sort_index().copy()
        if "close" not in prices.columns:
            raise ValueError("prices must contain a 'close' column")
        if prices.index.has_duplicates:
            raise ValueError("prices index must not contain duplicate timestamps")
        close = pd.to_numeric(prices["close"], errors="coerce")
        if close.isna().any() or not np.isfinite(close.to_numpy()).all() or (close <= 0).any():
            raise ValueError("prices['close'] must contain only positive finite values")
        prices["close"] = close.astype(float)
        return prices

    @staticmethod
    def _normalise_exposure(signals: pd.Series) -> pd.Series:
        exposure = pd.to_numeric(signals, errors="coerce")
        if exposure.isna().any() or not np.isfinite(exposure.to_numpy()).all():
            raise ValueError("signals must contain finite numeric values")
        if ((exposure < 0) | (exposure > 1)).any():
            raise ValueError("time-series exposure signals must be between 0 and 1")
        return exposure.astype(float)

    def _build_equity_curve(self, close: pd.Series, exposure: pd.Series) -> pd.Series:
        returns = close.pct_change().fillna(0.0)
        # Signal at t is applied from t+1 onward: no same-bar look-ahead.
        strategy_returns = returns * exposure.shift(1).fillna(0.0)
        equity = self.initial_capital * (1.0 + strategy_returns).cumprod()
        equity.name = "equity"
        return equity

    def _build_trade_log(self, prices: pd.DataFrame, strategy, exposure: pd.Series, equity_curve: pd.Series) -> pd.DataFrame:
        ticker = self.ticker or (
            str(getattr(strategy, "config", None).universe[0])
            if getattr(getattr(strategy, "config", None), "universe", None)
            else "UNKNOWN"
        )
        tag = getattr(getattr(strategy, "config", None), "name", self.strategy_tag)
        rows: list[dict] = []
        active = False
        entry_idx = None
        entry_price = 0.0
        entry_equity = 0.0
        entry_exposure = 0.0

        for i, (timestamp, current) in enumerate(exposure.items()):
            current = float(current)
            prev = float(exposure.iloc[i - 1]) if i else 0.0

            if not active and current > 0 and prev <= 0:
                active = True
                entry_idx = timestamp
                entry_price = float(prices.loc[timestamp, "close"])
                entry_equity = float(equity_curve.loc[timestamp])
                entry_exposure = current

            should_exit = active and current <= 0 and prev > 0
            is_last = i == len(exposure) - 1
            if active and (should_exit or is_last):
                exit_idx = timestamp
                exit_price = float(prices.loc[exit_idx, "close"])
                quantity = (entry_equity * entry_exposure) / entry_price
                pnl = (exit_price - entry_price) * quantity
                bars = int(prices.index.get_loc(exit_idx) - prices.index.get_loc(entry_idx))
                rows.append({
                    "ticker": ticker,
                    "strategy_tag": tag,
                    "status": "closed",
                    "entry_date": entry_idx,
                    "entry_price": entry_price,
                    "exit_date": exit_idx,
                    "exit_price": exit_price,
                    "side": "long",
                    "quantity": quantity,
                    "pnl_realized": pnl,
                    "pnl_unrealized": 0.0,
                    "pnl": pnl,
                    "return_pct": exit_price / entry_price - 1.0,
                    "bars": bars,
                })
                active = False
                entry_idx = None

        if not rows:
            return pd.DataFrame([{
                "ticker": ticker,
                "strategy_tag": tag,
                "status": "flat",
                "entry_date": prices.index[0],
                "entry_price": float(prices["close"].iloc[0]),
                "exit_date": prices.index[-1],
                "exit_price": float(prices["close"].iloc[-1]),
                "side": "long",
                "quantity": 0.0,
                "pnl_realized": 0.0,
                "pnl_unrealized": 0.0,
                "pnl": 0.0,
                "return_pct": 0.0,
                "bars": max(len(prices) - 1, 0),
            }])
        return pd.DataFrame(rows)

    def _legacy_snapshot_result(self, prices, strategy, signals):
        """Compatibility path for the existing cross-sectional strategy API."""
        entry_price = float(prices["close"].iloc[0])
        exit_price = float(prices["close"].iloc[-1])
        positive = not signals.empty and float(signals.iloc[0]) > 0
        pnl = exit_price - entry_price if positive else 0.0
        return_pct = exit_price / entry_price - 1.0 if positive else 0.0
        equity = self.initial_capital + (prices["close"].astype(float) - entry_price if positive else 0.0)
        ticker = self.ticker or (
            str(getattr(strategy, "config", None).universe[0])
            if getattr(getattr(strategy, "config", None), "universe", None)
            else "UNKNOWN"
        )
        tag = getattr(getattr(strategy, "config", None), "name", self.strategy_tag)
        trade = pd.DataFrame([{
            "ticker": ticker,
            "strategy_tag": tag,
            "status": "closed",
            "entry_date": prices.index[0],
            "entry_price": entry_price,
            "exit_date": prices.index[-1],
            "exit_price": exit_price,
            "side": "long",
            "quantity": 1.0 if positive else 0.0,
            "pnl_realized": pnl,
            "pnl_unrealized": 0.0,
            "pnl": pnl,
            "return_pct": return_pct,
            "bars": max(len(prices) - 1, 0),
        }])
        return equity, trade

    def _metrics(self, equity_curve: pd.Series, trade_log: pd.DataFrame) -> dict:
        rets = equity_curve.pct_change().fillna(0.0)
        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
        annualisation = self._annualisation_factor(equity_curve.index)
        std = rets.std(ddof=0)
        sharpe = 0.0 if std == 0 else float(np.sqrt(annualisation) * rets.mean() / std)
        running_peak = equity_curve.cummax()
        max_drawdown = abs(float((equity_curve / running_peak - 1.0).min()))
        n_trades = int(len(trade_log))
        win_rate = float((trade_log["pnl"] > 0).mean()) if n_trades else 0.0
        losses = abs(float(trade_log["pnl"].clip(upper=0).sum())) if n_trades else 0.0
        gains = float(trade_log["pnl"].clip(lower=0).sum()) if n_trades else 0.0
        profit_factor = float("inf") if losses == 0 else gains / losses
        cagr = 0.0
        if len(equity_curve) > 1:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            if days > 0 and equity_curve.iloc[-1] > 0:
                cagr = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365.25 / days) - 1.0)
        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trades": n_trades,
            "warning_nonpositive_sharpe": bool(sharpe <= 0.0),
        }

    @staticmethod
    def _annualisation_factor(index: pd.DatetimeIndex) -> float:
        if len(index) < 2:
            return 252.0
        deltas = index.to_series().diff().dropna().dt.total_seconds() / 86400.0
        median_days = float(deltas.median())
        return 252.0 if median_days <= 0 else max(1.0, 365.25 / median_days)

    def _save_outputs(self, trade_log, equity_curve, metrics) -> None:
        validate_trade_log(trade_log).to_csv(self.output_dir / "trade_log.csv", index=False)
        validate_equity_curve(equity_curve.rename("equity").rename_axis("date").reset_index()).to_csv(self.output_dir / "equity_curve.csv", index=False)
        validate_metrics(pd.DataFrame([metrics])).to_csv(self.output_dir / "metrics.csv", index=False)

        rets = equity_curve.pct_change().fillna(0.0)
        annualisation = self._annualisation_factor(equity_curve.index)
        downside = rets[rets < 0]
        downside_std = downside.std(ddof=0)
        sortino = 0.0 if pd.isna(downside_std) or downside_std == 0 else float(np.sqrt(annualisation) * rets.mean() / downside_std)
        gains = float(trade_log["pnl"].clip(lower=0).sum())
        losses = abs(float(trade_log["pnl"].clip(upper=0).sum()))
        pf = float("inf") if losses == 0 else gains / losses
        avg_win = float(trade_log.loc[trade_log["pnl"] > 0, "pnl"].mean()) if (trade_log["pnl"] > 0).any() else 0.0
        avg_loss = abs(float(trade_log.loc[trade_log["pnl"] < 0, "pnl"].mean())) if (trade_log["pnl"] < 0).any() else 0.0
        avg_rr = float("inf") if avg_loss == 0 else avg_win / avg_loss
        risk = pd.DataFrame([{
            "n_trades": len(trade_log),
            "win_rate": metrics["win_rate"],
            "profit_factor": pf,
            "avg_rr": avg_rr,
            "sharpe": metrics["sharpe"],
            "sortino": sortino,
            "warning_low_pf": bool(pf < 1.2),
            "warning_nonpositive_sharpe": metrics["warning_nonpositive_sharpe"],
        }])
        validate_risk_summary(risk).to_csv(self.output_dir / "risk_summary.csv", index=False)
