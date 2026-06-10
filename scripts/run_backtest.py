from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest import BacktestEngine
from src.output_schema import validate_equity_curve, validate_metrics, validate_trade_log
from src.strategies import MomentumStrategy, StrategyConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", required=True)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    args = parser.parse_args()

    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    if "close" not in prices.columns:
        first_col = prices.columns[0]
        ticker = first_col
        prices = prices[[first_col]].rename(columns={first_col: "close"})
    else:
        ticker = "close"
    config = StrategyConfig(name="momentum", universe=["close"], lookback=min(20, len(prices)))
    strategy = MomentumStrategy(config)
    engine = BacktestEngine(
        initial_capital=args.initial_capital,
        output_dir=args.output_dir,
        strategy_tag="momentum",
        ticker=ticker,
    )
    result = engine.run(prices)

    validate_trade_log(pd.read_csv(Path(args.output_dir) / "trade_log.csv"))
    validate_metrics(pd.read_csv(Path(args.output_dir) / "metrics.csv"))
    validate_equity_curve(pd.read_csv(Path(args.output_dir) / "equity_curve.csv"))

    print(result.metrics)
    print(result.trade_log.tail())
    print(result.equity_curve.tail())


if __name__ == "__main__":
    main()
