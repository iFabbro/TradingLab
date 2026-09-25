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
    def __init__(self, initial_capital: float = 100000.0, output_dir: str | Path = "data/backtests", strategy_tag: str = "backtest", ticker: str = "") -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        self.initial_capital = float(initial_capital)
        self.output_dir = Path(output_dir)
        self.strategy_tag = strategy_tag
        self.ticker = ticker
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, prices: pd.DataFrame, strategy) -> BacktestResult:
        if prices.empty:
            raise ValueError("prices cannot be empty")
        prices = prices.sort_index().copy()
        if "close" not in prices.columns:
            raise ValueError("prices must contain a 'close' column")
        if not prices.index.is_monotonic_increasing:
            raise ValueError("prices index must be sortable as dates")

        signals = strategy.generate_signals(prices)
        equity_curve = pd.Series(self.initial_capital, index=prices.index, dtype=float)
        entry_price = float(prices["close"].iloc[0])
        exit_price = float(prices["close"].iloc[-1])
        pnl = 0.0
        return_pct = 0.0

        is_time_signal = isinstance(signals, pd.Series) and signals.index.equals(prices.index)
        if is_time_signal:
            position = signals.astype(float).clip(lower=0.0, upper=1.0).fillna(0.0)
            close_returns = prices["close"].astype(float).pct_change().fillna(0.0)
            strategy_returns = close_returns * position.shift(1).fillna(0.0)
            equity_curve = self.initial_capital * (1.0 + strategy_returns).cumprod()
            pnl = float(equity_curve.iloc[-1] - self.initial_capital)
            return_pct = float(equity_curve.iloc[-1] / self.initial_capital - 1.0)
        elif not signals.empty and float(signals.iloc[0]) > 0:
            pnl = (exit_price - entry_price)
            return_pct = exit_price / entry_price - 1.0
            equity_curve = self.initial_capital + (prices["close"].astype(float) - entry_price)

        trade_log = self._build_trade_log(prices, strategy, is_time_signal, signals, entry_price, exit_price, pnl, return_pct)
        metrics = self._metrics(equity_curve, trade_log)
        self._save_outputs(trade_log, equity_curve, metrics)
        return BacktestResult(trade_log=trade_log, equity_curve=equity_curve, metrics=metrics)

    def _build_trade_log(self, prices, strategy, is_time_signal, signals, entry_price, exit_price, pnl, return_pct):
        ticker = self.ticker or (str(getattr(strategy, "config", None).universe[0]) if getattr(getattr(strategy, "config", None), "universe", None) else "UNKNOWN")
        tag = getattr(getattr(strategy, "config", None), "name", self.strategy_tag)

        if is_time_signal:
            position = signals.astype(float).clip(lower=0.0, upper=1.0).fillna(0.0)
            pos_diff = position.diff().fillna(position.iloc[0])
            entries = pos_diff[pos_diff > 0].index
            exits = pos_diff[pos_diff < 0].index
            entry_idx = entries[0] if len(entries) else (prices.index[0] if position.iloc[0] > 0 else None)
            if entry_idx is not None:
                later_exits = exits[exits > entry_idx]
                exit_idx = later_exits[0] if len(later_exits) else prices.index[-1]
                entry_price = float(prices.loc[entry_idx, "close"])
                exit_price = float(prices.loc[exit_idx, "close"])
                quantity = self.initial_capital / entry_price if entry_price else 0.0
                pnl = (exit_price - entry_price) * quantity
                return_pct = exit_price / entry_price - 1.0 if entry_price else 0.0
                bars = int(prices.index.get_loc(exit_idx) - prices.index.get_loc(entry_idx))
                return pd.DataFrame([{"ticker": ticker, "strategy_tag": tag, "status": "closed", "entry_date": entry_idx, "entry_price": entry_price, "exit_date": exit_idx, "exit_price": exit_price, "side": "long", "quantity": quantity, "pnl_realized": pnl, "pnl_unrealized": 0.0, "pnl": pnl, "return_pct": return_pct, "bars": bars}])

        return pd.DataFrame([{"ticker": ticker, "strategy_tag": tag, "status": "closed", "entry_date": prices.index[0], "entry_price": entry_price, "exit_date": prices.index[-1], "exit_price": exit_price, "side": "long", "quantity": 1.0, "pnl_realized": pnl, "pnl_unrealized": 0.0, "pnl": pnl, "return_pct": return_pct, "bars": max(len(prices) - 1, 0)}])

    def _metrics(self, equity_curve: pd.Series, trade_log: pd.DataFrame) -> dict:
        rets = equity_curve.pct_change().fillna(0.0)
        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
        std = rets.std(ddof=0)
        sharpe = 0.0 if std == 0 else float(np.sqrt(252) * rets.mean() / std)
        running_peak = equity_curve.cummax()
        max_drawdown = abs(float((equity_curve / running_peak - 1.0).min()))
        n_trades = int(len(trade_log))
        win_rate = float((trade_log["pnl"] > 0).mean()) if n_trades else 0.0
        losses = abs(float(trade_log["pnl"].clip(upper=0).sum())) if n_trades else 0.0
        gains = float(trade_log["pnl"].clip(lower=0).sum()) if n_trades else 0.0
        profit_factor = float("inf") if losses == 0 else gains / losses

        cagr = 0.0
        if len(equity_curve) > 1:
            days = (pd.Timestamp(equity_curve.index[-1]) - pd.Timestamp(equity_curve.index[0])).days
            if days > 0 and equity_curve.iloc[-1] > 0:
                cagr = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365.25 / days) - 1.0)

        return {"total_return": total_return, "cagr": cagr, "sharpe": sharpe, "max_drawdown": max_drawdown, "win_rate": win_rate, "profit_factor": profit_factor, "n_trades": n_trades, "warning_nonpositive_sharpe": bool(sharpe <= 0.0)}

    def _save_outputs(self, trade_log, equity_curve, metrics) -> None:
        validate_trade_log(trade_log).to_csv(self.output_dir / "trade_log.csv", index=False)
        validate_equity_curve(equity_curve.rename("equity").rename_axis("date").reset_index()).to_csv(self.output_dir / "equity_curve.csv", index=False)
        validate_metrics(pd.DataFrame([metrics])).to_csv(self.output_dir / "metrics.csv", index=False)

        rets = equity_curve.pct_change().fillna(0.0)
        downside = rets[rets < 0]
        downside_std = downside.std(ddof=0)
        sortino = 0.0 if pd.isna(downside_std) or downside_std == 0 else float(np.sqrt(252) * rets.mean() / downside_std)
        gains = float(trade_log["pnl"].clip(lower=0).sum())
        losses = abs(float(trade_log["pnl"].clip(upper=0).sum()))
        pf = float("inf") if losses == 0 else gains / losses
        avg_win = float(trade_log.loc[trade_log["pnl"] > 0, "pnl"].mean()) if (trade_log["pnl"] > 0).any() else 0.0
        avg_loss = abs(float(trade_log.loc[trade_log["pnl"] < 0, "pnl"].mean())) if (trade_log["pnl"] < 0).any() else 0.0
        avg_rr = float("inf") if avg_loss == 0 else avg_win / avg_loss
        risk = pd.DataFrame([{"n_trades": len(trade_log), "win_rate": metrics["win_rate"], "profit_factor": pf, "avg_rr": avg_rr, "sharpe": metrics["sharpe"], "sortino": sortino, "warning_low_pf": bool(pf < 1.2), "warning_nonpositive_sharpe": metrics["warning_nonpositive_sharpe"]}])
        validate_risk_summary(risk).to_csv(self.output_dir / "risk_summary.csv", index=False)
