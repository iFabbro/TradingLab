from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest import BacktestEngine
from src.strategies import MomentumStrategy, StrategyConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", required=True)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    args = parser.parse_args()

    prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
    config = StrategyConfig(name="momentum", universe=list(prices.columns), lookback=min(20, len(prices)))
    strategy = MomentumStrategy(config)
    engine = BacktestEngine(config=config, initial_capital=args.initial_capital, output_dir=args.output_dir)
    result = engine.run(prices, strategy)

    print(result.metrics)
    print(result.trade_log.tail())
    print(result.equity_curve.tail())


if __name__ == "__main__":
    main()
