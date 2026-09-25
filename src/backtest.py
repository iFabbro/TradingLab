from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.output_schema import validate_equity_curve, validate_metrics, validate_risk_summary, validate_trade_log


@dataclass
class BacktestResult:
    trade_log: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict


class BacktestEngine:
    """Long-only close-to-close backtester with explicit trading frictions.

    Contract:
    - signals are point-in-time and applied from the following bar;
    - exposure is a finite numeric value in [0, 1] and is clipped to that
      interval so a malformed strategy cannot introduce unintended leverage;
    - equity is fully marked-to-market and is always a pandas Series;
    - position size is based on equity at trade entry, so successive trades
      compound by default through the evolving equity curve;
    - transaction costs and slippage are charged on every exposure change via
      turnover * (transaction_cost_bps + slippage_bps) / 10_000;
    - trade-log P&L reconciles to the change in marked-to-market equity between
      the trade entry and exit timestamps.

    A signal at timestamp t is known at the close of t and is applied from
    t+1. Transaction costs and slippage are charged on exposure changes.
    """

    def __init__(self, initial_capital: float = 100000.0, output_dir: str | Path = "data/backtests", strategy_tag: str = "backtest", ticker: str = "", transaction_cost_bps: float = 0.0, slippage_bps: float = 0.0) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if transaction_cost_bps < 0 or slippage_bps < 0:
            raise ValueError("transaction_cost_bps and slippage_bps must be >= 0")
        self.initial_capital = float(initial_capital)
        self.output_dir = Path(output_dir)
        self.strategy_tag = strategy_tag
        self.ticker = ticker
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.slippage_bps = float(slippage_bps)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def friction_rate(self) -> float:
        return (self.transaction_cost_bps + self.slippage_bps) / 10_000.0

    def run(self, prices: pd.DataFrame, strategy) -> BacktestResult:
        prices = self._validate_prices(prices)
        signals = strategy.generate_signals(prices)
        is_time_signal = isinstance(signals, pd.Series) and signals.index.equals(prices.index)
        if is_time_signal:
            exposure = self._normalise_exposure(signals)
            self._last_exposure = exposure
            equity_curve = self._build_equity_curve(prices["close"], exposure)
            trade_log = self._build_trade_log(prices, strategy, exposure, equity_curve)
        else:
            self._last_exposure = pd.Series(0.0, index=prices.index, dtype=float)
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
        return exposure.clip(lower=0.0, upper=1.0).astype(float)

    def _build_equity_curve(self, close: pd.Series, exposure: pd.Series) -> pd.Series:
        returns = close.pct_change().fillna(0.0)
        held_exposure = exposure.shift(1).fillna(0.0)
        gross_returns = returns * held_exposure
        turnover = exposure.diff().abs().fillna(exposure.abs())
        strategy_returns = gross_returns - turnover * self.friction_rate
        equity = self.initial_capital * (1.0 + strategy_returns).cumprod()
        return pd.Series(equity.to_numpy(dtype=float), index=close.index)

    def _build_trade_log(self, prices: pd.DataFrame, strategy, exposure: pd.Series, equity_curve: pd.Series) -> pd.DataFrame:
        cfg = getattr(strategy, "config", None)
        ticker = self.ticker or (str(cfg.universe[0]) if getattr(cfg, "universe", None) else "UNKNOWN")
        tag = getattr(cfg, "name", self.strategy_tag)
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
                pnl = float(equity_curve.loc[exit_idx] - entry_equity)
                net_return = 0.0 if entry_equity == 0 else pnl / entry_equity
                bars = int(prices.index.get_loc(exit_idx) - prices.index.get_loc(entry_idx))
                rows.append({"ticker": ticker, "strategy_tag": tag, "status": "closed", "entry_date": entry_idx, "entry_price": entry_price, "exit_date": exit_idx, "exit_price": exit_price, "side": "long", "quantity": quantity, "pnl_realized": pnl, "pnl_unrealized": 0.0, "pnl": pnl, "return_pct": net_return, "bars": bars})
                active = False
                entry_idx = None
        if not rows:
            return pd.DataFrame([{"ticker": ticker, "strategy_tag": tag, "status": "flat", "entry_date": prices.index[0], "entry_price": float(prices["close"].iloc[0]), "exit_date": prices.index[-1], "exit_price": float(prices["close"].iloc[-1]), "side": "long", "quantity": 0.0, "pnl_realized": 0.0, "pnl_unrealized": 0.0, "pnl": 0.0, "return_pct": 0.0, "bars": max(len(prices) - 1, 0)}])
        return pd.DataFrame(rows)

    def _legacy_snapshot_result(self, prices, strategy, signals):
        entry_price = float(prices["close"].iloc[0])
        exit_price = float(prices["close"].iloc[-1])
        positive = not signals.empty and float(signals.iloc[0]) > 0
        quantity = 1.0 if positive else 0.0
        pnl = (exit_price - entry_price) * quantity
        return_pct = exit_price / entry_price - 1.0 if positive else 0.0
        price_delta = prices["close"].astype(float) - entry_price
        equity = self.initial_capital + (price_delta * quantity)
        equity = pd.Series(equity.to_numpy(dtype=float), index=prices.index)
        cfg = getattr(strategy, "config", None)
        ticker = self.ticker or (str(cfg.universe[0]) if getattr(cfg, "universe", None) else "UNKNOWN")
        tag = getattr(cfg, "name", self.strategy_tag)
        trade = pd.DataFrame([{"ticker": ticker, "strategy_tag": tag, "status": "closed", "entry_date": prices.index[0], "entry_price": entry_price, "exit_date": prices.index[-1], "exit_price": exit_price, "side": "long", "quantity": quantity, "pnl_realized": pnl, "pnl_unrealized": 0.0, "pnl": pnl, "return_pct": return_pct, "bars": max(len(prices) - 1, 0)}])
        return equity, trade

    def _metrics(self, equity_curve: pd.Series, trade_log: pd.DataFrame) -> dict:
        equity_curve = pd.Series(equity_curve, index=equity_curve.index, dtype=float)
        rets = equity_curve.pct_change().fillna(0.0)
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
        warning_low_pf = bool(profit_factor < 1.2)
        cagr = 0.0
        if len(equity_curve) > 1:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            if days > 0 and equity_curve.iloc[-1] > 0:
                cagr = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365.25 / days) - 1.0)
        turnover = float(self._last_exposure.diff().abs().fillna(self._last_exposure.abs()).sum())
        return {"total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0), "cagr": cagr, "sharpe": sharpe, "max_drawdown": max_drawdown, "win_rate": win_rate, "profit_factor": profit_factor, "n_trades": n_trades, "turnover": turnover, "transaction_cost_bps": self.transaction_cost_bps, "slippage_bps": self.slippage_bps, "warning_low_pf": warning_low_pf, "warning_nonpositive_sharpe": bool(sharpe <= 0.0)}

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
        downside_std = rets[rets < 0].std(ddof=0)
        sortino = 0.0 if pd.isna(downside_std) or downside_std == 0 else float(np.sqrt(annualisation) * rets.mean() / downside_std)
        gains = float(trade_log["pnl"].clip(lower=0).sum())
        losses = abs(float(trade_log["pnl"].clip(upper=0).sum()))
        pf = float("inf") if losses == 0 else gains / losses
        avg_win = float(trade_log.loc[trade_log["pnl"] > 0, "pnl"].mean()) if (trade_log["pnl"] > 0).any() else 0.0
        avg_loss = abs(float(trade_log.loc[trade_log["pnl"] < 0, "pnl"].mean())) if (trade_log["pnl"] < 0).any() else 0.0
        avg_rr = float("inf") if avg_loss == 0 else avg_win / avg_loss
        risk = pd.DataFrame([{"n_trades": len(trade_log), "win_rate": metrics["win_rate"], "profit_factor": pf, "avg_rr": avg_rr, "sharpe": metrics["sharpe"], "sortino": sortino, "warning_low_pf": bool(pf < 1.2), "warning_nonpositive_sharpe": metrics["warning_nonpositive_sharpe"]}])
        validate_risk_summary(risk).to_csv(self.output_dir / "risk_summary.csv", index=False)
