from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import BacktestEngine
from src.optimization import format_optimization_report, optimize_strategy
from src.strategies import StrategyConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strategy optimization")
    parser.add_argument("--name", required=True)
    parser.add_argument("--universe", required=True, nargs="+")
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument("--strategy", required=True)
    args = parser.parse_args()

    base_config = StrategyConfig(
        name=args.name,
        universe=args.universe,
        lookback=args.lookback,
    )

    def engine_factory(cfg):
        return BacktestEngine(cfg)

    result = optimize_strategy(
        engine_factory=engine_factory,
        base_config=base_config,
        train_data="train",
        test_data="test",
        param_grid={"lookback": [args.lookback - 2, args.lookback - 1, args.lookback, args.lookback + 1]},
    )

    print(format_optimization_report(result))
    best = result.best_candidate
    if best is not None:
        print("\nBest params:")
        pprint(best.params)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
