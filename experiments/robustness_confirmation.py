"""Pre-registered robustness, benchmark and untouched confirmation study.

The development period is 2015-01-01..2025-01-01. No confirmation-period
observation is used for parameter selection. The confirmation period is
2025-01-02..2026-06-30 and is evaluated only after a deterministic parameter
rule has been derived from development validation windows.

The generic point-in-time strategy adapter recomputes the full Donchian state
from every historical prefix. That is O(N^2) per parameter trial and made the
pre-registered plateau study unnecessarily expensive. This module therefore
uses a local linear-time adapter implementing the same point-in-time rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import BacktestEngine
from src.providers import YahooFinanceProvider, save_dataset
from src.strategies import DonchianBreakoutStrategy, StrategyConfig
from src.walk_forward import WalkForwardEvaluator

DEVELOPMENT_START = "2015-01-01"
DEVELOPMENT_END = "2025-01-01"
CONFIRMATION_START = "2025-01-02"
CONFIRMATION_END = "2026-06-30"
PLATEAU_GRID = [20, 30, 40, 50, 55, 60, 70, 80, 90, 100]
TRAIN_SIZE = 756
VALIDATION_SIZE = 252
TEST_SIZE = 252
STEP_SIZE = 252
COST_BPS = 5.0
SLIPPAGE_BPS = 2.0


class _FixedSignalStrategy:
    def __init__(self, signals: pd.Series, source_strategy):
        self.signals = signals
        self.config = getattr(source_strategy, "config", None)

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if not prices.index.equals(self.signals.index):
            raise ValueError("signal index must exactly match price index")
        return self.signals


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="spy")
    p.add_argument("--output-root", default="experiments/results")
    return p.parse_args()


def engine(out, label):
    return lambda: BacktestEngine(100000, out / f"bt_{label}", "donchian", "SPY", COST_BPS, SLIPPAGE_BPS)


def strategy(symbol, lookback):
    config = StrategyConfig(
        name="donchian",
        universe=[symbol.upper()],
        lookback=int(lookback),
    )
    return DonchianBreakoutStrategy(config)


def metric(result, name):
    value = result.metrics.get(name)
    return None if value is None or not np.isfinite(float(value)) else float(value)


def donchian_time_series_signals(history: pd.DataFrame, lookback: int) -> pd.Series:
    """Generate the Donchian state in O(N), preserving point-in-time semantics.

    At timestamp t the channel uses only the previous `lookback` closes. A
    close above the previous maximum enters/holds long; a close below the
    previous minimum exits; otherwise the prior state is held. This matches
    the state transition in DonchianBreakoutStrategy.generate_signals while
    avoiding a full historical recomputation at every timestamp.
    """
    if history.empty or "close" not in history.columns:
        raise ValueError("history must contain a non-empty close column")
    if lookback <= 1:
        raise ValueError("Donchian lookback must be > 1")
    close = pd.to_numeric(history["close"], errors="coerce")
    if close.isna().any() or not np.isfinite(close.to_numpy()).all() or (close <= 0).any():
        raise ValueError("Donchian prices must contain positive finite values")

    upper = close.rolling(lookback, min_periods=lookback).max().shift(1)
    lower = close.rolling(lookback, min_periods=lookback).min().shift(1)
    close_values = close.to_numpy(dtype=float)
    upper_values = upper.to_numpy(dtype=float)
    lower_values = lower.to_numpy(dtype=float)
    state = 0.0
    output = np.zeros(len(close), dtype=float)
    for i in range(len(close)):
        if np.isfinite(upper_values[i]) and np.isfinite(lower_values[i]):
            if close_values[i] > upper_values[i]:
                state = 1.0
            elif close_values[i] < lower_values[i]:
                state = 0.0
        output[i] = state
    return pd.Series(output, index=history.index, name="signal")


def run_candidate(history, evaluation_index, evaluation_prices, lookback, backtest_factory):
    source = strategy("SPY", lookback)
    signals = donchian_time_series_signals(history, int(lookback)).loc[evaluation_index]
    return backtest_factory().run(evaluation_prices, _FixedSignalStrategy(signals, source))


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
        validation = prices.loc[w.validation_start:w.validation_end]
        test = prices.loc[w.test_start:w.test_end]
        history = pd.concat([train, validation, test])
        strat = run_candidate(history, test.index, test, params["lookback"], engine(out, f"benchmark_strat_{i}"))
        bh = test["close"].astype(float) / float(test["close"].iloc[0])
        daily = bh.pct_change().fillna(0.0)
        equity = 100000.0 * bh
        vol = float(daily.std(ddof=0))
        sharpe = None if vol == 0 else float(np.sqrt(252.0) * daily.mean() / vol)
        running = equity.cummax()
        max_dd = float((equity / running - 1.0).min())
        rows.append({"window": i, "test_start": w.test_start, "test_end": w.test_end, "donchian_lookback": params["lookback"], "donchian_total_return": metric(strat, "total_return"), "donchian_cagr": metric(strat, "cagr"), "donchian_sharpe": metric(strat, "sharpe"), "donchian_max_drawdown": metric(strat, "max_drawdown"), "buy_hold_total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0), "buy_hold_sharpe": sharpe, "buy_hold_max_drawdown": max_dd})
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
        history = pd.concat([train, validation])
        for lb in PLATEAU_GRID:
            result = run_candidate(history, validation.index, validation, lb, engine(out, f"plateau_{i}_{lb}"))
            rows.append({"window": i, "lookback": lb, "validation_sharpe": metric(result, "sharpe"), "validation_return": metric(result, "total_return"), "validation_drawdown": metric(result, "max_drawdown")})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "validation_plateau.csv", index=False)
    selected = []
    for window, group in frame.groupby("window"):
        group = group.dropna(subset=["validation_sharpe"]).sort_values("lookback")
        if group.empty:
            selected.append({"window": int(window), "plateau": [], "plateau_center": None, "status": "no_finite_validation_sharpe"})
            continue
        best = group["validation_sharpe"].max()
        plateau = group[group["validation_sharpe"] >= 0.80 * best]
        if plateau.empty:
            selected.append({"window": int(window), "plateau": [], "plateau_center": None, "status": "empty_plateau"})
            continue
        selected.append({"window": int(window), "plateau": plateau["lookback"].astype(int).tolist(), "plateau_center": int(round(float(plateau["lookback"].median()))), "status": "ok"})
    selected_frame = pd.DataFrame(selected)
    selected_frame.to_csv(out / "validation_plateau_selection.csv", index=False)
    centers = [int(row["plateau_center"]) for row in selected if row["plateau_center"] is not None]
    if not centers:
        chosen = None
        selection_status = "no_valid_plateau"
    elif len(centers) < len(selected):
        chosen = None
        selection_status = "incomplete_plateau"
    else:
        chosen = int(round(float(np.median(centers))))
        selection_status = "ok"
    summary = {"grid": PLATEAU_GRID, "rule": "within 80% of best validation Sharpe; median plateau member", "window_selection": selected, "confirmation_lookback": chosen, "selection_status": selection_status}
    (out / "validation_plateau.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return chosen, summary


def confirmation(prices: pd.DataFrame, lookback: int, out: Path):
    test = prices.loc[CONFIRMATION_START:CONFIRMATION_END]
    if test.empty:
        raise ValueError("confirmation period returned no data")
    history = prices.loc[:CONFIRMATION_START].iloc[:-1]
    result = run_candidate(pd.concat([history, test]), test.index, test, lookback, engine(out, "confirmation"))
    output = {"status": "confirmed_run", "period": {"start": CONFIRMATION_START, "end": CONFIRMATION_END}, "frozen_lookback": lookback, "metrics": {k: metric(result, k) for k in ("total_return", "cagr", "sharpe", "max_drawdown", "turnover")}}
    (out / "untouched_confirmation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def write_unconfirmed_confirmation(out: Path, reason: str):
    output = {"status": "not_confirmed", "period": {"start": CONFIRMATION_START, "end": CONFIRMATION_END}, "frozen_lookback": None, "metrics": {}, "reason": reason}
    (out / "untouched_confirmation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main():
    a = args()
    total_start = time.perf_counter()
    out = Path(a.output_root) / "donchian_robustness_study"
    out.mkdir(parents=True, exist_ok=True)
    provider = YahooFinanceProvider()

    started = time.perf_counter()
    all_prices = provider.fetch(a.symbol, DEVELOPMENT_START, CONFIRMATION_END, "1d")
    download_seconds = time.perf_counter() - started
    dev = all_prices.loc[DEVELOPMENT_START:DEVELOPMENT_END]
    conf = all_prices.loc[CONFIRMATION_START:CONFIRMATION_END]
    if dev.empty or conf.empty:
        raise ValueError("development or confirmation period returned no data")
    save_dataset(all_prices, out / "dataset.csv")

    started = time.perf_counter()
    benchmark = benchmark_original_oos(Path(a.output_root) / "spy_donchian_2015-01-01_2025-01-01_1d_yfinance", dev, out)
    benchmark_seconds = time.perf_counter() - started

    started = time.perf_counter()
    chosen, plateau = plateau_study(dev, out)
    plateau_seconds = time.perf_counter() - started

    started = time.perf_counter()
    if chosen is None:
        confirmation_result = write_unconfirmed_confirmation(out, f"plateau selection not confirmed: {plateau['selection_status']}")
    else:
        confirmation_result = confirmation(all_prices, chosen, out)
    confirmation_seconds = time.perf_counter() - started

    timings = {
        "single_price_download_seconds": round(download_seconds, 3),
        "benchmark_seconds": round(benchmark_seconds, 3),
        "plateau_seconds": round(plateau_seconds, 3),
        "confirmation_seconds": round(confirmation_seconds, 3),
        "total_seconds": round(time.perf_counter() - total_start, 3),
    }
    metadata = {"development_period": [DEVELOPMENT_START, DEVELOPMENT_END], "confirmation_period": [CONFIRMATION_START, CONFIRMATION_END], "plateau_grid": PLATEAU_GRID, "plateau_rule": plateau["rule"], "confirmation_lookback": chosen, "confirmation_status": confirmation_result["status"], "no_confirmation_data_used_for_selection": True, "cost_bps": COST_BPS, "slippage_bps": SLIPPAGE_BPS, "implementation": "linear_time_donchian_signal_adapter", "timings_seconds": timings}
    (out / "study_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"benchmark_rows": len(benchmark), "confirmation_lookback": chosen, "confirmation": confirmation_result, "timings_seconds": timings}, indent=2))


if __name__ == "__main__":
    main()
