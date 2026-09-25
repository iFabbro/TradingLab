# Quantitative validity and backtesting assumptions

This document records what TradingLab currently models and what it does not claim to model.

## Signal timing

Signals are assumed to be generated at the close of timestamp `t` and affect returns from `t+1`. A strategy must not use future rows when generating a signal.

## Investable universe

`StrategyConfig.universe` is currently a static list. It is therefore **not** a survivorship-bias-free historical universe. Historical constituent membership, delistings, corporate actions and point-in-time security availability are not currently represented by the public backtester.

Results using today's surviving tickers must therefore be described as research on a selected universe, not as a historical investable-universe simulation.

## Costs and slippage

The backtester supports explicit transaction-cost and slippage assumptions in basis points. They are charged against absolute exposure changes. A zero-cost run is therefore an optimistic friction-free scenario and should not be used as the only reported result.

Example:

```python
BacktestEngine(
    transaction_cost_bps=5.0,
    slippage_bps=5.0,
)
```

These are still simplified assumptions: market impact, spread dynamics, liquidity constraints and partial fills are not modeled.

## Turnover

Gross turnover is reported as the cumulative absolute change in exposure. It should be evaluated alongside returns because high turnover can make an apparently attractive strategy economically unrealistic.

## Position sizing

The time-series backtester currently supports normalized long exposure in `[0, 1]`. Portfolio-level position sizing is handled separately. This is intentionally narrower than claiming a full portfolio optimizer or execution simulator.

## Out-of-sample validation

A strategy should not be judged from a single in-sample backtest. The intended validation sequence is:

1. reserve a chronological holdout period before parameter selection;
2. select parameters only on the training period;
3. freeze parameters;
4. evaluate once on the untouched out-of-sample period;
5. repeat with walk-forward windows when enough data is available;
6. compare gross and net-of-cost results;
7. report drawdown, turnover and trade count alongside return metrics.

TradingLab does not currently ship a complete point-in-time universe database or an automated walk-forward optimizer. Those are explicitly future work rather than capabilities implied by the current repository.

## Interpretation

Backtest statistics are software outputs, not evidence of future profitability. Any portfolio or strategy result should be accompanied by its data period, universe definition, costs, slippage assumptions, parameter-selection procedure and out-of-sample status.
