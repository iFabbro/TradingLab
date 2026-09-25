"""End-to-end reproducible quantitative experiment runner."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import BacktestEngine
from src.providers import StooqCSVProvider, YahooFinanceProvider, save_dataset, save_metadata
from src.strategies import MeanReversionStrategy, MomentumStrategy, StrategyConfig, TrendFollowingStrategy
from src.walk_forward import WalkForwardEvaluator

STRATEGIES = {"momentum": MomentumStrategy, "mean_reversion": MeanReversionStrategy, "trend": TrendFollowingStrategy}
PROVIDERS = {"stooq": StooqCSVProvider, "yfinance": YahooFinanceProvider}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--interval", default="1d")
    p.add_argument("--provider", default="yfinance", choices=sorted(PROVIDERS))
    p.add_argument("--strategy", default="momentum", choices=sorted(STRATEGIES))
    p.add_argument("--parameter-grid", required=True)
    p.add_argument("--selection-metric", default="sharpe")
    p.add_argument("--train-size", type=int, required=True)
    p.add_argument("--validation-size", type=int, required=True)
    p.add_argument("--test-size", type=int, required=True)
    p.add_argument("--step-size", type=int, default=None)
    p.add_argument("--transaction-cost-bps", type=float, default=0.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--oos-block-length", type=int, default=None)
    p.add_argument("--output-root", default="experiments/results")
    return p.parse_args()


def clean(v):
    if is_dataclass(v): return clean(asdict(v))
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, pd.DataFrame): return clean(v.to_dict(orient="records"))
    if isinstance(v, pd.Series): return clean(v.to_dict())
    if isinstance(v, pd.Timestamp): return v.isoformat()
    if hasattr(v, "item"):
        try: return v.item()
        except Exception: pass
    if isinstance(v, float) and math.isnan(v): return None
    if v == float("inf"): return "inf"
    if v == float("-inf"): return "-inf"
    return v


def git_commit() -> str | None:
    value = os.getenv("GITHUB_SHA")
    if value:
        return value
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    a = parse_args()
    grid = json.loads(a.parameter_grid)
    if not isinstance(grid, dict) or not grid:
        raise ValueError("--parameter-grid must be a non-empty JSON object")

    provider = PROVIDERS[a.provider]()
    df = provider.fetch(a.symbol, a.start, a.end, a.interval)
    if df.empty or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("provider returned invalid market data")
    if df["close"].isna().any() or not (df["close"] > 0).all():
        raise ValueError("provider returned invalid close prices")

    run_id = f"{a.symbol.lower()}_{a.strategy}_{a.start}_{a.end}_{a.interval}_{a.provider}"
    out = Path(a.output_root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    dataset_path, metadata_path = out / "dataset.csv", out / "metadata.json"
    report_path, selection_path = out / "report.json", out / "parameter_selection.csv"
    ledger_path = out / "research_trial_ledger.csv"

    sha = save_dataset(df, dataset_path)
    candidate_count = 1
    for values in grid.values():
        values = list(values)
        candidate_count *= len(values)
    metadata = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "symbol": a.symbol.upper(),
        "strategy": a.strategy,
        "requested_period": {"start": a.start, "end": a.end},
        "actual_period": {"start": df.index.min().isoformat(), "end": df.index.max().isoformat()},
        "interval": a.interval,
        "provider": provider.metadata(),
        "rows": int(len(df)),
        "dataset_sha256": sha,
        "synthetic_data": False,
        "local_fallback": False,
        "walk_forward": {"train_size": a.train_size, "validation_size": a.validation_size, "test_size": a.test_size, "step_size": a.step_size},
        "frictions": {"transaction_cost_bps": a.transaction_cost_bps, "slippage_bps": a.slippage_bps},
        "parameter_grid": grid,
        "nominal_candidates_per_window": candidate_count,
        "selection_metric": a.selection_metric,
        "inference": {
            "confidence": a.confidence,
            "bootstrap_samples": a.bootstrap_samples,
            "oos_block_length": a.oos_block_length,
            "trial_ledger": "research_trial_ledger.csv",
            "dsr": "nominal candidate count per validation window; effective independent trials are not assumed known",
        },
    }
    save_metadata(metadata, metadata_path)

    cls = STRATEGIES[a.strategy]

    def factory(train, params):
        if "lookback" not in params:
            raise ValueError("parameter grid must contain lookback")
        cfg = StrategyConfig(
            name=a.strategy,
            universe=[a.symbol.upper()],
            lookback=int(params["lookback"]),
            params={k: v for k, v in params.items() if k != "lookback"},
        )
        return cls(cfg)

    def bt():
        return BacktestEngine(
            initial_capital=100000,
            output_dir=out / "backtest_outputs",
            strategy_tag=a.strategy,
            ticker=a.symbol.upper(),
            transaction_cost_bps=a.transaction_cost_bps,
            slippage_bps=a.slippage_bps,
        )

    result = WalkForwardEvaluator(a.train_size, a.validation_size, a.test_size, a.step_size).evaluate(
        df, factory, bt, grid, a.selection_metric, a.selection_metric != "max_drawdown"
    )
    selection_metrics = result.selection_metrics
    selection_metrics.to_csv(selection_path, index=False)
    selection_metrics.to_csv(ledger_path, index=False)

    robustness = result.robustness_report(a.confidence, a.bootstrap_samples, a.oos_block_length)
    report = {
        "provenance": metadata,
        "windows": result.windows,
        "selected_parameters": result.selected_parameters,
        "train_metrics": result.train_metrics,
        "validation_metrics": result.validation_metrics,
        "test_metrics": result.test_metrics,
        "research_trial_ledger": ledger_path.name,
        "nominal_total_candidate_trials": int(len(selection_metrics)),
        "robustness": robustness,
    }
    report_path.write_text(json.dumps(clean(report), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "dataset": str(dataset_path), "metadata": str(metadata_path), "report": str(report_path), "parameter_selection": str(selection_path), "trial_ledger": str(ledger_path), "windows": len(result.test_results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
