"""Walk-forward train/validation/test and parameter-selection utilities."""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Any, Iterable
import pandas as pd
from src.backtest import BacktestEngine, BacktestResult
from src.statistics import aggregate_window_metrics, bootstrap_mean_ci, degradation, multiple_testing_diagnostics

@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp; train_end: pd.Timestamp; validation_start: pd.Timestamp; validation_end: pd.Timestamp; test_start: pd.Timestamp; test_end: pd.Timestamp

@dataclass
class WalkForwardResult:
    windows: pd.DataFrame
    validation_results: list[BacktestResult] = field(default_factory=list)
    test_results: list[BacktestResult] = field(default_factory=list)
    selected_parameters: list[dict[str, Any]] = field(default_factory=list)
    selection_results: list[pd.DataFrame] = field(default_factory=list)
    train_results: list[BacktestResult] = field(default_factory=list)
    @property
    def validation_metrics(self): return pd.DataFrame([r.metrics for r in self.validation_results])
    @property
    def test_metrics(self): return pd.DataFrame([r.metrics for r in self.test_results])
    @property
    def train_metrics(self): return pd.DataFrame([r.metrics for r in self.train_results])
    @property
    def selection_metrics(self):
        frames=[]
        for window, frame in enumerate(self.selection_results,1):
            current=frame.copy(); current.insert(0,"window",window); frames.append(current)
        return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    def robustness_report(self, confidence=0.95, bootstrap_samples=5000):
        train, validation, test=self.train_metrics,self.validation_metrics,self.test_metrics
        report={"train_summary":aggregate_window_metrics(train),"validation_summary":aggregate_window_metrics(validation),"test_summary":aggregate_window_metrics(test),"oos_bootstrap":{},"degradation":{},"multiple_testing":[]}
        if not test.empty:
            for metric in ("total_return","cagr","sharpe","max_drawdown","turnover"):
                if metric in test and len(test[metric].dropna())>=2: report["oos_bootstrap"][metric]=bootstrap_mean_ci(test[metric].dropna().to_numpy(),confidence,bootstrap_samples)
        for metric in ("total_return","cagr","sharpe","max_drawdown"):
            if metric in train and metric in test: report["degradation"][metric]=degradation(float(train[metric].mean()),float(test[metric].mean()))
        for frame in self.selection_results:
            if not frame.empty and "sharpe" in frame: report["multiple_testing"].append(multiple_testing_diagnostics(len(frame),float(frame["sharpe"].max()),observations=max(2,len(frame))))
        return report

class WalkForwardEvaluator:
    """Chronological expanding-window evaluator with explicit leakage boundaries."""
    def __init__(self,train_size,validation_size,test_size,step_size=None):
        if min(train_size,validation_size,test_size)<=0: raise ValueError("window sizes must be > 0")
        self.train_size=train_size; self.validation_size=validation_size; self.test_size=test_size; self.step_size=step_size or test_size
        if self.step_size<=0: raise ValueError("step_size must be > 0")
    def windows(self,index):
        if not isinstance(index,pd.DatetimeIndex): raise ValueError("index must be a DatetimeIndex")
        if index.has_duplicates or not index.is_monotonic_increasing: raise ValueError("index must be unique and sorted")
        total=self.train_size+self.validation_size+self.test_size; result=[]; start=0
        while start+total<=len(index):
            train=index[start:start+self.train_size]; validation=index[start+self.train_size:start+self.train_size+self.validation_size]; test=index[start+self.train_size+self.validation_size:start+total]
            result.append(WalkForwardWindow(train[0],train[-1],validation[0],validation[-1],test[0],test[-1])); start+=self.step_size
        return result
    @staticmethod
    def _signals(strategy,history,evaluation_index):
        generator=getattr(strategy,"generate_time_series_signals",None)
        if generator is None: raise TypeError("strategy must implement generate_time_series_signals for walk-forward evaluation")
        signals=generator(history)
        if not signals.index.equals(history.index): raise ValueError("time-series signal index must match history index")
        # BacktestEngine models long-only exposure. Convert strategy scores to exposure
        # at this adapter boundary rather than silently allowing short positions.
        signals=pd.to_numeric(signals,errors="coerce")
        if signals.isna().any(): raise ValueError("strategy produced non-numeric/NaN signals")
        return signals.clip(0.0,1.0).loc[evaluation_index]
    @staticmethod
    def _parameter_combinations(parameter_grid):
        if not parameter_grid: raise ValueError("parameter_grid cannot be empty")
        keys=list(parameter_grid); values=[list(parameter_grid[k]) for k in keys]
        if any(not options for options in values): raise ValueError("parameter_grid values cannot be empty")
        return [dict(zip(keys,combo)) for combo in product(*values)]
    @staticmethod
    def _metric_value(result,metric):
        if metric not in result.metrics: raise ValueError(f"selection metric not available: {metric}")
        value=float(result.metrics[metric]); return float("-inf") if pd.isna(value) else value
    def _run_candidate(self,strategy,history,evaluation_index,evaluation_prices,backtest_factory):
        signals=self._signals(strategy,history,evaluation_index); engine=backtest_factory() if backtest_factory else BacktestEngine(); return engine.run(evaluation_prices,_SeriesSignalStrategy(signals,strategy))
    def evaluate(self,prices,strategy_factory,backtest_factory=None,parameter_grid=None,selection_metric="sharpe",maximize=True):
        prices=prices.sort_index(); windows=self.windows(prices.index); candidates=self._parameter_combinations(parameter_grid) if parameter_grid is not None else [None]
        validation_results=[]; test_results=[]; train_results=[]; selected_parameters=[]; selection_results=[]; rows=[]
        for number,window in enumerate(windows,1):
            train=prices.loc[window.train_start:window.train_end]; validation=prices.loc[window.validation_start:window.validation_end]; test=prices.loc[window.test_start:window.test_end]
            validation_history=pd.concat([train,validation]); candidate_rows=[]; candidate_objects=[]
            for params in candidates:
                try:
                    strategy=strategy_factory(train,params) if params is not None else strategy_factory(train)
                    vr=self._run_candidate(strategy,validation_history,validation.index,validation,backtest_factory); score=self._metric_value(vr,selection_metric)
                    candidate_rows.append({**(params or {}),selection_metric:score,"status":"ok"}); candidate_objects.append((params or {},strategy,vr,score))
                except Exception as exc: candidate_rows.append({**(params or {}),selection_metric:float("-inf"),"status":f"error: {exc}"})
            if not candidate_objects: raise RuntimeError(f"no valid parameter candidate in window {number}")
            selected=max(candidate_objects,key=lambda x:x[3]) if maximize else min(candidate_objects,key=lambda x:x[3])
            selected_params,frozen_strategy,validation_result,selected_score=selected
            selection_frame=pd.DataFrame(candidate_rows).sort_values(selection_metric,ascending=not maximize,ignore_index=True)
            train_result=self._run_candidate(frozen_strategy,train,train.index,train,backtest_factory)
            test_result=self._run_candidate(frozen_strategy,pd.concat([train,validation,test]),test.index,test,backtest_factory)
            train_results.append(train_result); validation_results.append(validation_result); test_results.append(test_result); selected_parameters.append(selected_params); selection_results.append(selection_frame)
            rows.append({"window":number,"train_start":window.train_start,"train_end":window.train_end,"validation_start":window.validation_start,"validation_end":window.validation_end,"test_start":window.test_start,"test_end":window.test_end,"selected_parameters":selected_params,"validation_selection_score":selected_score})
        return WalkForwardResult(pd.DataFrame(rows),validation_results,test_results,selected_parameters,selection_results,train_results)

class _SeriesSignalStrategy:
    def __init__(self,signals,source_strategy): self.signals=signals; self.config=getattr(source_strategy,"config",None)
    def generate_signals(self,prices):
        if not prices.index.equals(self.signals.index): raise ValueError("signal index must exactly match price index")
        return self.signals
