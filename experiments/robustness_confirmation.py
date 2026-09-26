"""Pre-registered robustness, benchmark and untouched confirmation study.

The development period is 2015-01-01..2025-01-01. No confirmation-period
observation is used for parameter selection. The confirmation period is
2025-01-02..2026-06-30 and is evaluated only after a deterministic parameter
rule has been derived from development validation windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import BacktestEngine
from src.providers import YahooFinanceProvider, save_dataset, save_metadata
from src.strategies import DonchianBreakoutStrategy, StrategyConfig
from src.walk_forward import WalkForwardEvaluator

DEVELOPMENT_START = "2015-01-01"
DEVELOPMENT_END = "2025-01-01"
CONFIRMATION_START = "2025-01-02"
CONFIRMATION_END = "2026-06-30"
PLATEAU_GRID = list(range(20, 101, 10))
TRAIN_SIZE = 756
VALIDATION_SIZE = 252
TEST_SIZE = 252
STEP_SIZE = 252
COST_BPS = 5.0
SLIPPAGE_BPS = 2.0


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="spy")
    p.add_argument("--output-root", default="experiments/results")
    return p.parse_args()


def engine(out, label):
    return lambda: BacktestEngine(100000, out / f"bt_{label}", "donchian", "SPY", COST_BPS, SLIPPAGE_BPS)


def strategy(symbol, lookback):
    return DonchianBreakoutStrategy(StrategyConfig("donchian", [symbol.upper()], int(lookback), {}))


def metric(result, name):
    value = result.metrics.get(name)
    return None if value is None or not np.isfinite(float(value)) else float(value)


def benchmark_original_oos(dev_dir: Path, prices: pd.DataFrame, out: Path):
    report = json.loads((dev_dir / "report.json").read_text())
    selected = report["selected_parameters"]
    evaluator = WalkForwardEvaluator(TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, STEP_SIZE)
    windows = evaluator.windows(prices.index)
    if len(windows) != len(selected):
        raise ValueError("development selected parameters do not match windows")

    rows = []
    for i, (w, params) in enumerate(zip(windows, selected), 1):
        train = prices.loc[w.train_start:w.train_end]
        test = prices.loc[w.test_start:w.test_end]
        frozen = strategy("SPY", params["lookback"])
        strat = evaluator._run_candidate(frozen, pd.concat([train, prices.loc[w.validation_start:w.validation_end], test]), test.index, test, engine(out, f"benchmark_strat_{i}"))

        # Buy-and-hold uses the identical OOS dates and the same initial capital.
        bh = test["close"].astype(float) / float(test["close"].iloc[0])
        daily = bh.pct_change().fillna(0.0)
        equity = 100000.0 * bh
        vol = float(daily.std(ddof=0))
        sharpe = None if vol == 0 else float(np.sqrt(252.0) * daily.mean() / vol)
        running = equity.cummax()
        max_dd = float((equity / running - 1.0).min())
        rows.append({
            "window": i,
            "test_start": w.test_start,
            "test_end": w.test_end,
            "donchian_lookback": params["lookback"],
            "donchian_total_return": metric(strat, "total_return"),
            "donchian_cagr": metric(strat, "cagr"),
            "donchian_sharpe": metric(strat, "sharpe"),
            "donchian_max_drawdown": metric(strat, "max_drawdown"),
            "buy_hold_total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
            "buy_hold_sharpe": sharpe,
            "buy_hold_max_drawdown": max_dd,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "benchmark_donchian_vs_buy_hold.csv", index=False)
    return frame


def plateau_study(prices: pd.DataFrame, out: Path):
    evaluator = WalkForwardEvaluator(TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, STEP_SIZE)
    windows = evaluator.windows(prices.index)
    rows = []
    for i, w in enumerate(windows, 1):
        train = prices.loc[w.train_start:w.train_end]
        validation = prices.loc[w.validation_start:w.validation_end]
        for lb in PLATEAU_GRID:
            frozen = strategy("SPY", lb)
            result = evaluator._run_candidate(frozen, pd.concat([train, validation]), validation.index, validation, engine(out, f"plateau_{i}_{lb}"))
            rows.append({"window": i, "lookback": lb, "validation_sharpe": metric(result, "sharpe"), "validation_return": metric(result, "total_return"), "validation_drawdown": metric(result, "max_drawdown")})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "validation_plateau.csv", index=False)

    # Pre-registered plateau rule: parameters within 80% of the window's
    # best validation Sharpe form the plateau; choose the median member.
    selected = []
    for window, group in frame.groupby("window"):
        group = group.dropna(subset=["validation_sharpe"]).sort_values("lookback")
        best = group["validation_sharpe"].max()
        plateau = group[group["validation_sharpe"] >= 0.80 * best]
        selected.append({"window": int(window), "plateau": plateau["lookback"].astype(int).tolist(), "plateau_center": int(round(float(plateau["lookback"].median())))})
    selected_frame = pd.DataFrame(selected)
    selected_frame.to_csv(out / "validation_plateau_selection.csv", index=False)
    chosen = int(round(float(selected_frame["plateau_center"].median())))
    summary = {"grid": PLATEAU_GRID, "rule": "within 80% of best validation Sharpe; median plateau member", "window_selection": selected, "confirmation_lookback": chosen}
    (out / "validation_plateau.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return chosen, summary


def confirmation(prices: pd.DataFrame, lookback: int, out: Path):
    test = prices.loc[CONFIRMATION_START:CONFIRMATION_END]
    if test.empty:
        raise ValueError("confirmation period returned no data")
    frozen = strategy("SPY", lookback)
    evaluator = WalkForwardEvaluator(TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, STEP_SIZE)
    # The strategy parameter is frozen before this period. We use prior history
    # only as indicator warm-up; no confirmation observations enter selection.
    history = prices.loc[:CONFIRMATION_START].iloc[:-1]
    result = evaluator._run_candidate(frozen, pd.concat([history, test]), test.index, test, engine(out, "confirmation"))
    output = {"period": {"start": CONFIRMATION_START, "end": CONFIRMATION_END}, "frozen_lookback": lookback, "metrics": {k: metric(result, k) for k in ("total_return", "cagr", "sharpe", "max_drawdown", "turnover")}}
    (out / "untouched_confirmation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main():
    a = args()
    out = Path(a.output_root) / "donchian_robustness_study"
    out.mkdir(parents=True, exist_ok=True)
    provider = YahooFinanceProvider()
    dev = provider.fetch(a.symbol, DEVELOPMENT_START, DEVELOPMENT_END, "1d")
    conf = provider.fetch(a.symbol, CONFIRMATION_START, CONFIRMATION_END, "1d")
    all_prices = pd.concat([dev, conf]).loc[~pd.concat([dev, conf]).index.duplicated()].sort_index()
    save_dataset(all_prices, out / "dataset.csv")
    benchmark = benchmark_original_oos(Path(a.output_root) / "spy_donchian_2015-01-01_2025-01-01_1d_yfinance", dev, out)
    chosen, plateau = plateau_study(dev, out)
    confirmation_result = confirmation(all_prices, chosen, out)
    metadata = {"development_period": [DEVELOPMENT_START, DEVELOPMENT_END], "confirmation_period": [CONFIRMATION_START, CONFIRMATION_END], "plateau_grid": PLATEAU_GRID, "plateau_rule": plateau["rule"], "confirmation_lookback": chosen, "no_confirmation_data_used_for_selection": True, "cost_bps": COST_BPS, "slippage_bps": SLIPPAGE_BPS}
    (out / "study_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"benchmark_rows": len(benchmark), "confirmation_lookback": chosen, "confirmation": confirmation_result}, indent=2))


if __name__ == "__main__":
    main()
