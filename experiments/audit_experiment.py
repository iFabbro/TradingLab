"""Produce audit artifacts from a completed reproducible walk-forward experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import BacktestEngine
from src.strategies import MeanReversionStrategy, MomentumStrategy, StrategyConfig, TrendFollowingStrategy
from src.walk_forward import WalkForwardEvaluator

STRATEGIES = {"momentum": MomentumStrategy, "mean_reversion": MeanReversionStrategy, "trend": TrendFollowingStrategy}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def trade_log_artifact(results, windows, selected_parameters, cost_model):
    frames = []
    for i, (result, window) in enumerate(zip(results, windows), 1):
        frame = result.trade_log.copy()
        frame.insert(0, "window", i)
        frame.insert(1, "phase", "oos")
        frame.insert(2, "cost_model", cost_model)
        frame.insert(3, "selected_parameters", json.dumps(selected_parameters[i - 1], sort_keys=True))
        frame.insert(4, "test_start", window.test_start)
        frame.insert(5, "test_end", window.test_end)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def gross_net_report(net_results, gross_results):
    rows = []
    for i, (net, gross) in enumerate(zip(net_results, gross_results), 1):
        row = {"window": i}
        for metric in ("total_return", "cagr", "sharpe", "max_drawdown"):
            row[f"gross_{metric}"] = float(gross.metrics[metric])
            row[f"net_{metric}"] = float(net.metrics[metric])
            row[f"friction_drag_{metric}"] = float(gross.metrics[metric] - net.metrics[metric])
        row["gross_turnover"] = float(gross.metrics["turnover"])
        row["net_turnover"] = float(net.metrics["turnover"])
        rows.append(row)

    def pooled(results):
        series = [r.equity_curve.pct_change().fillna(0.0) for r in results]
        if not series:
            return pd.Series(dtype=float)
        out = pd.concat(series)
        if out.index.has_duplicates:
            raise ValueError("OOS windows overlap; cannot pool audit returns")
        return out.sort_index().astype(float)

    net = pooled(net_results)
    gross = pooled(gross_results)
    if net.empty or gross.empty:
        return {"windows": rows, "aggregate": {}}
    gross_return = float(np.prod(1.0 + gross.to_numpy()) - 1.0)
    net_return = float(np.prod(1.0 + net.to_numpy()) - 1.0)
    gross_std = gross.std(ddof=0)
    net_std = net.std(ddof=0)
    gross_sharpe = 0.0 if gross_std == 0 else float(np.sqrt(252.0) * gross.mean() / gross_std)
    net_sharpe = 0.0 if net_std == 0 else float(np.sqrt(252.0) * net.mean() / net_std)
    return {
        "windows": rows,
        "aggregate": {
            "gross_total_return": gross_return,
            "net_total_return": net_return,
            "friction_drag_total_return": gross_return - net_return,
            "gross_sharpe": gross_sharpe,
            "net_sharpe": net_sharpe,
            "friction_drag_sharpe": gross_sharpe - net_sharpe,
            "oos_observations": int(len(net)),
        },
    }


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    report_path = run_dir / "report.json"
    dataset_path = run_dir / "dataset.csv"
    metadata_path = run_dir / "metadata.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prices = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    selected = report["selected_parameters"]
    walk = metadata["walk_forward"]
    friction = metadata["frictions"]
    net_cost = float(friction.get("net_transaction_cost_bps", friction.get("transaction_cost_bps", 0.0)))
    net_slippage = float(friction.get("net_slippage_bps", friction.get("slippage_bps", 0.0)))
    strategy_name = metadata["strategy"]
    symbol = metadata["symbol"]
    strategy_cls = STRATEGIES[strategy_name]

    evaluator = WalkForwardEvaluator(walk["train_size"], walk["validation_size"], walk["test_size"], walk.get("step_size"))
    windows = evaluator.windows(prices.index)
    if len(windows) != len(selected):
        raise ValueError("selected parameter count does not match walk-forward windows")

    def strategy_factory(train, params):
        cfg = StrategyConfig(
            name=strategy_name,
            universe=[symbol],
            lookback=int(params["lookback"]),
            params={k: v for k, v in params.items() if k != "lookback"},
        )
        return strategy_cls(cfg)

    def engine_factory(cost_bps, slippage_bps, label):
        def build():
            return BacktestEngine(
                initial_capital=100000,
                output_dir=run_dir / f"audit_backtest_{label}",
                strategy_tag=strategy_name,
                ticker=symbol,
                transaction_cost_bps=cost_bps,
                slippage_bps=slippage_bps,
            )
        return build

    net_results = []
    gross_results = []
    for i, window in enumerate(windows, 1):
        train = prices.loc[window.train_start:window.train_end]
        validation = prices.loc[window.validation_start:window.validation_end]
        test = prices.loc[window.test_start:window.test_end]
        history = pd.concat([train, validation, test])
        frozen = strategy_factory(train, selected[i - 1])
        net_results.append(evaluator._run_candidate(frozen, history, test.index, test, engine_factory(net_cost, net_slippage, "net")))
        gross_results.append(evaluator._run_candidate(frozen, history, test.index, test, engine_factory(0.0, 0.0, "gross")))

    trade_net = trade_log_artifact(net_results, windows, selected, "net")
    trade_gross = trade_log_artifact(gross_results, windows, selected, "gross")
    pd.concat([trade_net, trade_gross], ignore_index=True).to_csv(run_dir / "trade_log_oos_all.csv", index=False)

    comparison = gross_net_report(net_results, gross_results)
    (run_dir / "gross_vs_net_oos.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    pd.DataFrame(comparison["windows"]).to_csv(run_dir / "gross_vs_net_oos.csv", index=False)

    report["audit"] = {
        "gross_vs_net": "gross_vs_net_oos.json",
        "trade_log_oos_all": "trade_log_oos_all.csv",
        "same_dataset": True,
        "same_frozen_parameters": True,
        "same_oos_windows": True,
        "gross_costs_bps": 0.0,
        "gross_slippage_bps": 0.0,
        "net_transaction_cost_bps": net_cost,
        "net_slippage_bps": net_slippage,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
