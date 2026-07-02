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
    def __init__(
        self,
        initial_capital: float = 100000.0,
        output_dir: str | Path = "data/backtests",
        strategy_tag: str = "backtest",
        ticker: str = "",
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.output_dir = Path(output_dir)
        self.strategy_tag = strategy_tag
        self.ticker = ticker
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, prices: pd.DataFrame, strategy) -> BacktestResult:
        if prices.empty:
            raise ValueError("prices non può essere vuoto")

        prices = prices.sort_index().copy()
        if "close" not in prices.columns:
            raise ValueError("prices deve contenere la colonna 'close'")

        signals = strategy.generate_signals(prices)
        quantity = 1.0
        entry_price = float(prices["close"].iloc[0])
        exit_price = float(prices["close"].iloc[-1])
        pnl = 0.0
        return_pct = 0.0

        equity_curve = pd.Series(index=prices.index, dtype=float)
        equity_curve.iloc[0:] = self.initial_capital

        is_time_signal = isinstance(signals, pd.Series) and signals.index.equals(prices.index)

        if is_time_signal:
            position = signals.astype(float).clip(lower=0.0, upper=1.0).fillna(0.0)
            close_returns = prices["close"].astype(float).pct_change().fillna(0.0)
            strategy_returns = close_returns * position.shift(1).fillna(0.0)
            equity_curve = self.initial_capital * (1.0 + strategy_returns).cumprod()
            pnl = float(equity_curve.iloc[-1] - self.initial_capital)
            return_pct = float(equity_curve.iloc[-1] / self.initial_capital - 1.0)
        elif not signals.empty and float(signals.iloc[0]) > 0:
            pnl = (exit_price - entry_price) * quantity
            return_pct = exit_price / entry_price - 1.0
            equity_curve = self.initial_capital + (prices["close"].astype(float) - entry_price) * quantity

        if is_time_signal:
            entry_idx = None
            exit_idx = None
            pos_diff = position.diff().fillna(position.iloc[0])

            entry_points = pos_diff[pos_diff > 0].index
            exit_points = pos_diff[pos_diff < 0].index

            if len(entry_points) > 0:
                entry_idx = entry_points[0]
                later_exits = [ts for ts in exit_points if ts > entry_idx]
                exit_idx = later_exits[0] if later_exits else prices.index[-1]
            elif position.iloc[0] > 0:
                entry_idx = prices.index[0]
                exit_idx = exit_points[0] if len(exit_points) > 0 else prices.index[-1]

            if entry_idx is not None:
                entry_price = float(prices.loc[entry_idx, "close"])
                exit_price = float(prices.loc[exit_idx, "close"])
                trade_quantity = self.initial_capital / entry_price if entry_price != 0 else 0.0
                trade_pnl = (exit_price - entry_price) * trade_quantity
                trade_return_pct = exit_price / entry_price - 1.0
                trade_bars = int(prices.index.get_loc(exit_idx) - prices.index.get_loc(entry_idx))
                trade_log = pd.DataFrame(
                    [
                        {
                            "ticker": prices.columns[0],
                            "strategy_tag": getattr(getattr(strategy, "config", None), "name", self.strategy_tag),
                            "status": "closed",
                            "entry_date": entry_idx,
                            "entry_price": entry_price,
                            "exit_date": exit_idx,
                            "exit_price": exit_price,
                            "side": "long",
                            "quantity": trade_quantity,
                            "pnl": trade_pnl,
                            "return_pct": trade_return_pct,
                            "bars": trade_bars,
                        }
                    ]
                )
            else:
                trade_log = pd.DataFrame(
                    [
                        {
                            "ticker": prices.columns[0],
                            "strategy_tag": getattr(getattr(strategy, "config", None), "name", self.strategy_tag),
                            "status": "closed",
                            "entry_date": prices.index[0],
                            "entry_price": entry_price,
                            "exit_date": prices.index[-1],
                            "exit_price": exit_price,
                            "side": "long",
                            "quantity": 0.0,
                            "pnl": 0.0,
                            "return_pct": 0.0,
                            "bars": 0,
                        }
                    ]
                )
        else:
            trade_log = pd.DataFrame(
                [
                    {
                        "ticker": prices.columns[0],
                        "strategy_tag": getattr(getattr(strategy, "config", None), "name", self.strategy_tag),
                        "status": "closed",
                        "entry_date": prices.index[0],
                        "entry_price": entry_price,
                        "exit_date": prices.index[-1],
                        "exit_price": exit_price,
                        "side": "long",
                        "quantity": quantity,
                        "pnl": pnl,
                        "return_pct": return_pct,
                        "bars": max(len(prices) - 1, 0),
                    }
                ]
            )

        metrics = self._metrics(equity_curve, trade_log)
        self._save_outputs(trade_log, equity_curve, metrics)
        return BacktestResult(trade_log=trade_log, equity_curve=equity_curve, metrics=metrics)

    def _metrics(self, equity_curve: pd.Series, trade_log: pd.DataFrame) -> dict:
        rets = equity_curve.pct_change().fillna(0.0)
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0 if len(equity_curve) else 0.0
        sharpe = 0.0 if rets.std(ddof=0) == 0 else np.sqrt(252) * rets.mean() / rets.std(ddof=0)
        running_peak = equity_curve.cummax() if len(equity_curve) else equity_curve
        drawdown = (equity_curve / running_peak - 1.0).fillna(0.0) if len(equity_curve) else pd.Series(dtype=float)
        max_drawdown = abs(float(drawdown.min())) if len(drawdown) else 0.0
        win_rate = float((trade_log["pnl"] > 0).mean()) if not trade_log.empty else 0.0
        return {
            "total_return": float(total_return),
            "cagr": 0.0,
            "sharpe": float(sharpe),
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "n_trades": int(len(trade_log)),
            "warning_nonpositive_sharpe": bool(sharpe <= 0.0),
        }

    def _save_outputs(self, trade_log: pd.DataFrame, equity_curve: pd.Series, metrics: dict) -> None:
        trade_log_out = validate_trade_log(trade_log)
        equity_curve_out = validate_equity_curve(
            equity_curve.rename("equity").rename_axis("date").reset_index()
        )
        metrics_out = validate_metrics(pd.DataFrame([metrics]))
        rets = equity_curve.pct_change().fillna(0.0)
        downside = rets[rets < 0]
        sortino = 0.0 if downside.std(ddof=0) == 0 else np.sqrt(252) * rets.mean() / downside.std(ddof=0)

        risk_summary_out = validate_risk_summary(
            pd.DataFrame([{
                "n_trades": int(metrics.get("n_trades", 0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
                "profit_factor": float("inf") if trade_log.empty or float(trade_log["pnl"].clip(lower=0).sum()) == 0 and float(abs(trade_log["pnl"].clip(upper=0).sum())) == 0 else float(trade_log["pnl"].clip(lower=0).sum() / abs(trade_log["pnl"].clip(upper=0).sum())) if float(abs(trade_log["pnl"].clip(upper=0).sum())) != 0 else float("inf"),
                "avg_rr": float("inf") if trade_log.empty or float(abs(trade_log["pnl"].clip(upper=0).sum())) == 0 else float(trade_log["pnl"].clip(lower=0).mean() / abs(trade_log["pnl"].clip(upper=0).mean())) if float(trade_log["pnl"].clip(upper=0).mean()) != 0 else float("inf"),
                "sharpe": float(metrics.get("sharpe", 0.0)),
                "sortino": float(sortino),
                "warning_low_pf": bool(False),
                "warning_nonpositive_sharpe": bool(metrics.get("warning_nonpositive_sharpe", False)),
            }])
        )

        trade_log_out.to_csv(self.output_dir / "trade_log.csv", index=False)
        equity_curve_out.to_csv(self.output_dir / "equity_curve.csv", index=False)
        metrics_out.to_csv(self.output_dir / "metrics.csv", index=False)
        risk_summary_out.to_csv(self.output_dir / "risk_summary.csv", index=False)
