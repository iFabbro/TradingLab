"""Produce audit artifacts from a completed reproducible walk-forward experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import BacktestEngine
from src.strategies import (
    DonchianBreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    StrategyConfig,
    TrendFollowingStrategy,
)
from src.walk_forward import WalkForwardEvaluator

STRATEGIES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "trend": TrendFollowingStrategy,
    "donchian": DonchianBreakoutStrategy,
}
COUNTERFACTUAL_TRANSACTION_COST_BPS = 5.0
COUNTERFACTUAL_SLIPPAGE_BPS = 2.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _finite_series(series):
    values = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return values.astype(float)


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


def _metric_pair(gross, net, metric):
    gross_value = _finite_or_none(gross.metrics[metric])
    net_value = _finite_or_none(net.metrics[metric])
    drag = None if gross_value is None or net_value is None else gross_value - net_value
    return gross_value, net_value, drag


def _aggregate_finite(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def gross_net_report(net_results, gross_results):
    """Compare identical frozen OOS paths under gross and configured-net frictions.

    Non-finite metrics are excluded from aggregates rather than allowing +/-inf
    to contaminate the pooled result. The per-window record keeps nulls so an
    undefined statistic remains visible instead of being silently coerced.
    """
    rows = []
    for i, (net, gross) in enumerate(zip(net_results, gross_results), 1):
        row = {"window": i}
        for metric in ("total_return", "cagr", "sharpe", "max_drawdown"):
            gross_value, net_value, drag = _metric_pair(gross, net, metric)
            row[f"gross_{metric}"] = gross_value
            row[f"net_{metric}"] = net_value
            row[f"friction_drag_{metric}"] = drag
        row["gross_turnover"] = _finite_or_none(gross.metrics["turnover"])
        row["net_turnover"] = _finite_or_none(net.metrics["turnover"])
        rows.append(row)

    def pooled(results):
        series = []
        for result in results:
            returns = result.equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            if not returns.empty:
                series.append(returns.astype(float))
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
    gross_return = _finite_or_none(np.prod(1.0 + gross.to_numpy()) - 1.0)
    net_return = _finite_or_none(np.prod(1.0 + net.to_numpy()) - 1.0)
    gross_std = float(gross.std(ddof=0))
    net_std = float(net.std(ddof=0))
    gross_sharpe = _finite_or_none(0.0 if gross_std == 0 else np.sqrt(252.0) * gross.mean() / gross_std)
    net_sharpe = _finite_or_none(0.0 if net_std == 0 else np.sqrt(252.0) * net.mean() / net_std)
    return {
        "windows": rows,
        "aggregate": {
            "gross_total_return": gross_return,
            "net_total_return": net_return,
            "friction_drag_total_return": None if gross_return is None or net_return is None else gross_return - net_return,
            "gross_sharpe": gross_sharpe,
            "net_sharpe": net_sharpe,
            "friction_drag_sharpe": None if gross_sharpe is None or net_sharpe is None else gross_sharpe - net_sharpe,
            "oos_observations": int(len(net)),
        },
    }


def counterfactual_report(gross_results, counterfactual_results, baseline_net_results):
    """Report a fixed 5 bps cost + 2 bps slippage scenario independently.

    The scenario is replayed from the same frozen parameters, dataset and OOS
    windows. It is not substituted for the configured production friction model.
    """
    rows = []
    for i, (gross, cf, baseline) in enumerate(zip(gross_results, counterfactual_results, baseline_net_results), 1):
        row = {"window": i}
        for metric in ("total_return", "cagr", "sharpe", "max_drawdown"):
            row[f"gross_{metric}"] = _finite_or_none(gross.metrics[metric])
            row[f"counterfactual_5bps_2bps_{metric}"] = _finite_or_none(cf.metrics[metric])
            row[f"baseline_net_{metric}"] = _finite_or_none(baseline.metrics[metric])
            gross_value = row[f"gross_{metric}"]
            cf_value = row[f"counterfactual_5bps_2bps_{metric}"]
            base_value = row[f"baseline_net_{metric}"]
            row[f"counterfactual_vs_gross_drag_{metric}"] = None if gross_value is None or cf_value is None else gross_value - cf_value
            row[f"counterfactual_vs_baseline_net_delta_{metric}"] = None if cf_value is None or base_value is None else cf_value - base_value
        row["counterfactual_transaction_cost_bps"] = COUNTERFACTUAL_TRANSACTION_COST_BPS
        row["counterfactual_slippage_bps"] = COUNTERFACTUAL_SLIPPAGE_BPS
        rows.append(row)

    def pooled(results):
        series = []
        for result in results:
            returns = result.equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            if not returns.empty:
                series.append(returns.astype(float))
        if not series:
            return pd.Series(dtype=float)
        out = pd.concat(series)
        if out.index.has_duplicates:
            raise ValueError("OOS windows overlap; cannot pool counterfactual returns")
        return out.sort_index().astype(float)

    gross = pooled(gross_results)
    cf = pooled(counterfactual_results)
    baseline = pooled(baseline_net_results)
    aggregate = {
        "gross_total_return": None,
        "counterfactual_5bps_2bps_total_return": None,
        "baseline_net_total_return": None,
        "counterfactual_vs_gross_drag_total_return": None,
        "counterfactual_vs_baseline_net_delta_total_return": None,
        "gross_sharpe": None,
        "counterfactual_5bps_2bps_sharpe": None,
        "baseline_net_sharpe": None,
        "counterfactual_vs_gross_drag_sharpe": None,
        "counterfactual_vs_baseline_net_delta_sharpe": None,
        "oos_observations": int(len(cf)),
    }
    if not gross.empty and not cf.empty and not baseline.empty:
        gross_return = _finite_or_none(np.prod(1.0 + gross.to_numpy()) - 1.0)
        cf_return = _finite_or_none(np.prod(1.0 + cf.to_numpy()) - 1.0)
        baseline_return = _finite_or_none(np.prod(1.0 + baseline.to_numpy()) - 1.0)
        gross_std = float(gross.std(ddof=0))
        cf_std = float(cf.std(ddof=0))
        baseline_std = float(baseline.std(ddof=0))
        gross_sharpe = _finite_or_none(0.0 if gross_std == 0 else np.sqrt(252.0) * gross.mean() / gross_std)
        cf_sharpe = _finite_or_none(0.0 if cf_std == 0 else np.sqrt(252.0) * cf.mean() / cf_std)
        baseline_sharpe = _finite_or_none(0.0 if baseline_std == 0 else np.sqrt(252.0) * baseline.mean() / baseline_std)
        aggregate.update(
            {
                "gross_total_return": gross_return,
                "counterfactual_5bps_2bps_total_return": cf_return,
                "baseline_net_total_return": baseline_return,
                "counterfactual_vs_gross_drag_total_return": None if gross_return is None or cf_return is None else gross_return - cf_return,
                "counterfactual_vs_baseline_net_delta_total_return": None if cf_return is None or baseline_return is None else cf_return - baseline_return,
                "gross_sharpe": gross_sharpe,
                "counterfactual_5bps_2bps_sharpe": cf_sharpe,
                "baseline_net_sharpe": baseline_sharpe,
                "counterfactual_vs_gross_drag_sharpe": None if gross_sharpe is None or cf_sharpe is None else gross_sharpe - cf_sharpe,
                "counterfactual_vs_baseline_net_delta_sharpe": None if cf_sharpe is None or baseline_sharpe is None else cf_sharpe - baseline_sharpe,
            }
        )
    return {
        "scenario": {"transaction_cost_bps": COUNTERFACTUAL_TRANSACTION_COST_BPS, "slippage_bps": COUNTERFACTUAL_SLIPPAGE_BPS},
        "windows": rows,
        "aggregate": aggregate,
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
    counterfactual_results = []
    for i, window in enumerate(windows, 1):
        train = prices.loc[window.train_start:window.train_end]
        validation = prices.loc[window.validation_start:window.validation_end]
        test = prices.loc[window.test_start:window.test_end]
        history = pd.concat([train, validation, test])
        frozen = strategy_factory(train, selected[i - 1])
        net_results.append(evaluator._run_candidate(frozen, history, test.index, test, engine_factory(net_cost, net_slippage, "net")))
        gross_results.append(evaluator._run_candidate(frozen, history, test.index, test, engine_factory(0.0, 0.0, "gross")))
        counterfactual_results.append(
            evaluator._run_candidate(
                frozen,
                history,
                test.index,
                test,
                engine_factory(COUNTERFACTUAL_TRANSACTION_COST_BPS, COUNTERFACTUAL_SLIPPAGE_BPS, "counterfactual_5bps_2bps"),
            )
        )

    trade_net = trade_log_artifact(net_results, windows, selected, "net")
    trade_gross = trade_log_artifact(gross_results, windows, selected, "gross")
    trade_cf = trade_log_artifact(counterfactual_results, windows, selected, "counterfactual_5bps_2bps")
    pd.concat([trade_net, trade_gross, trade_cf], ignore_index=True).to_csv(run_dir / "trade_log_oos_all.csv", index=False)
    trade_cf.to_csv(run_dir / "trade_log_oos_5bps_2bps.csv", index=False)

    comparison = gross_net_report(net_results, gross_results)
    (run_dir / "gross_vs_net_oos.json").write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(comparison["windows"]).to_csv(run_dir / "gross_vs_net_oos.csv", index=False)

    counterfactual = counterfactual_report(gross_results, counterfactual_results, net_results)
    (run_dir / "counterfactual_5bps_2bps.json").write_text(json.dumps(counterfactual, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(counterfactual["windows"]).to_csv(run_dir / "counterfactual_5bps_2bps.csv", index=False)

    report["audit"] = {
        "gross_vs_net": "gross_vs_net_oos.json",
        "trade_log_oos_all": "trade_log_oos_all.csv",
        "trade_log_oos_5bps_2bps": "trade_log_oos_5bps_2bps.csv",
        "counterfactual_5bps_2bps": "counterfactual_5bps_2bps.json",
        "counterfactual_5bps_2bps_csv": "counterfactual_5bps_2bps.csv",
        "same_dataset": True,
        "same_frozen_parameters": True,
        "same_oos_windows": True,
        "gross_costs_bps": 0.0,
        "gross_slippage_bps": 0.0,
        "net_transaction_cost_bps": net_cost,
        "net_slippage_bps": net_slippage,
        "counterfactual_transaction_cost_bps": COUNTERFACTUAL_TRANSACTION_COST_BPS,
        "counterfactual_slippage_bps": COUNTERFACTUAL_SLIPPAGE_BPS,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
