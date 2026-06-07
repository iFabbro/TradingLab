from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import BacktestEngine
from src.data import load_ohlcv
from src.optimization import format_optimization_report, optimize_strategy
from src.strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    StrategyConfig,
    TrendFollowingStrategy,
)


def build_strategy(config: StrategyConfig, strategy_name: str):
    mapping = {
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "trend_following": TrendFollowingStrategy,
    }
    try:
        strategy_cls = mapping[strategy_name]
    except KeyError as exc:
        raise ValueError(f"Strategia non valida: {strategy_name}") from exc
    return strategy_cls(config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strategy optimization")
    parser.add_argument("--name", required=True)
    parser.add_argument("--universe", required=True, nargs="+")
    parser.add_argument("--lookback", type=int, required=True)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["momentum", "mean_reversion", "trend_following"],
    )
    args = parser.parse_args()

    base_config = StrategyConfig(
        name=args.name,
        universe=args.universe,
        lookback=args.lookback,
    )

    def engine_factory(cfg: StrategyConfig) -> BacktestEngine:
        return BacktestEngine(cfg)

    prices = load_ohlcv(
        ticker=args.universe[0],
        start="2020-01-01",
        end="2024-12-31",
        interval="1d",
    )

    split_idx = max(int(len(prices) * 0.7), 1)
    train_data = prices.iloc[:split_idx]
    test_data = prices.iloc[split_idx:]

    result = optimize_strategy(
        engine_factory=engine_factory,
        strategy_factory=lambda cfg: build_strategy(cfg, args.strategy),
        base_config=base_config,
        train_data=train_data,
        test_data=test_data,
        param_grid={"lookback": [args.lookback - 2, args.lookback - 1, args.lookback, args.lookback + 1]},
    )

    print(format_optimization_report(result))
    if result.best_candidate is None:
        print("No candidate passed anti-overfitting filters.")
    return 0 if result.best_candidate is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
