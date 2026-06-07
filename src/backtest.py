from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.strategies import BaseStrategy, StrategyConfig


@dataclass
class BacktestResult:
    trade_log: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict


class BacktestEngine:
    def __init__(
        self,
        config: StrategyConfig,
        initial_capital: float = 100000.0,
        output_dir: str | Path = "data/backtests",
        transaction_cost_bps: float = 0.0,
    ) -> None:
        self.config = config
        self.initial_capital = float(initial_capital)
        self.output_dir = Path(output_dir)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, prices: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        self.config.validate()
        if prices.empty:
            raise ValueError("prices non può essere vuoto")

        prices = prices.sort_index().copy()
        prices = prices[self.config.universe].dropna(how="all")
        if prices.empty:
            raise ValueError("prices non contiene dati validi per universe")

        equity = self.initial_capital
        equity_rows = []
        trades = []

        current_position: Optional[pd.Series] = None
        entry_date = None
        entry_value = None

        for i in range(len(prices)):
            current_prices = prices.iloc[: i + 1]
            date = prices.index[i]
            signal = strategy.generate_signals(current_prices).reindex(prices.columns).fillna(0.0)

            if current_position is None:
                current_position = signal.copy()
                entry_date = date
                entry_value = equity
            else:
                prev_pos = current_position
                turn_over = (signal - prev_pos).abs().sum()
                if turn_over > 0:
                    end_value = self._mark_to_market(equity, current_prices.iloc[-1], prev_pos)
                    trade_pnl = end_value - entry_value
                    trades.append(
                        {
                            "entry_date": entry_date,
                            "exit_date": date,
                            "side": "long" if prev_pos.sum() >= 0 else "short",
                            "pnl": trade_pnl,
                            "return_pct": trade_pnl / entry_value if entry_value else 0.0,
                            "duration_bars": i,
                        }
                    )
                    equity = end_value
                    current_position = signal.copy()
                    entry_date = date
                    entry_value = equity

            equity_rows.append({"date": date, "equity": equity})

        if current_position is not None and entry_date is not None and entry_value is not None:
            final_value = self._mark_to_market(equity, prices.iloc[-1], current_position)
            trade_pnl = final_value - entry_value
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": prices.index[-1],
                    "side": "long" if current_position.sum() >= 0 else "short",
                    "pnl": trade_pnl,
                    "return_pct": trade_pnl / entry_value if entry_value else 0.0,
                    "duration_bars": len(prices) - 1,
                }
            )
            equity = final_value
            equity_rows[-1]["equity"] = equity

        equity_curve = pd.DataFrame(equity_rows).set_index("date")["equity"]
        trade_log = pd.DataFrame(trades)
        metrics = self._metrics(equity_curve, trade_log)

        self._save_outputs(trade_log, equity_curve, metrics)
        return BacktestResult(trade_log=trade_log, equity_curve=equity_curve, metrics=metrics)

    def _mark_to_market(self, capital: float, last_prices: pd.Series, position: pd.Series) -> float:
        weights = position.astype(float)
        if weights.abs().sum() == 0:
            return capital
        weights = weights / weights.abs().sum()
        rel = last_prices / last_prices.iloc[0]
        gross_return = float((weights * rel).sum() - 1.0)
        cost = self.transaction_cost_bps / 10000.0 * float(weights.abs().sum())
        return capital * (1.0 + gross_return - cost)

    def _metrics(self, equity_curve: pd.Series, trade_log: pd.DataFrame) -> dict:
        rets = equity_curve.pct_change().fillna(0.0)
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
        n_years = max(len(equity_curve) / 252.0, 1 / 252.0)
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1.0
        sharpe = 0.0 if rets.std(ddof=0) == 0 else np.sqrt(252) * rets.mean() / rets.std(ddof=0)
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1.0
        max_dd = float(drawdown.min())
        win_rate = float((trade_log["pnl"] > 0).mean()) if not trade_log.empty else 0.0
        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "n_trades": int(len(trade_log)),
        }

    def _save_outputs(self, trade_log: pd.DataFrame, equity_curve: pd.Series, metrics: dict) -> None:
        trade_log.to_csv(self.output_dir / "trade_log.csv", index=False)
        equity_curve.rename("equity").to_csv(self.output_dir / "equity_curve.csv")
        pd.DataFrame([metrics]).to_csv(self.output_dir / "metrics.csv", index=False)
