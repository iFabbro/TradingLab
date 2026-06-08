from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        if prices.empty:
            raise ValueError("prices non può essere vuoto")

        prices = prices.sort_index().copy()
        if "close" not in prices.columns:
            raise ValueError("prices deve contenere la colonna 'close'")

        equity_curve = pd.Series(index=prices.index, dtype=float)
        equity_curve.iloc[0:] = self.initial_capital

        trade_log = pd.DataFrame(
            [
                {
                    "ticker": self.ticker,
                    "strategy_tag": self.strategy_tag,
                    "status": "closed",
                    "entry_date": prices.index[0],
                    "exit_date": prices.index[-1],
                    "direction": "long",
                    "pnl": 0.0,
                    "return_pct": 0.0,
                    "duration_bars": max(len(prices) - 1, 0),
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
        win_rate = float((trade_log["pnl"] > 0).mean()) if not trade_log.empty else 0.0
        return {
            "total_return": float(total_return),
            "sharpe": float(sharpe),
            "win_rate": win_rate,
            "n_trades": int(len(trade_log)),
        }

    def _save_outputs(self, trade_log: pd.DataFrame, equity_curve: pd.Series, metrics: dict) -> None:
        trade_log.to_csv(self.output_dir / "trade_log.csv", index=False)
        equity_curve.rename("equity").to_csv(self.output_dir / "equity_curve.csv")
        pd.DataFrame([metrics]).to_csv(self.output_dir / "metrics.csv", index=False)
