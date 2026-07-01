# TradingLab Data Schema

## trade_log.csv
Canonical columns:
- ticker
- strategy_tag
- status
- entry_date
- entry_price
- exit_date
- exit_price
- side
- pnl
- return_pct
- bars

Notes:
- Use `side` as canonical field name, not `direction`.
- Use `bars` as canonical field name, not `duration_bars`.
- `status` should be `open` or `closed`.

## metrics.csv
Canonical columns:
- total_return
- cagr
- sharpe
- max_drawdown
- win_rate
- n_trades
- warning_nonpositive_sharpe

Notes:
- All summary metrics files should expose the same header, even if some values are temporarily `0.0`.

## equity_curve.csv
Canonical columns:
- date
- equity

## open_trades.csv
Canonical columns:
- ticker
- side
- entry_price
- stop_price
- target_price
- risk_reward
- strategy_tag
- regime
- note
- status
- entry_date
- position_size

Notes:
- Use `side` as canonical field name, not `direction`.

## current_prices.csv
Canonical columns:
- ticker
- current_price
- asof

## macro_snapshot.csv
Canonical columns:
- date
- rate
- inflation
- gdp
- rate_trend
- inflation_yoy
- inflation_trend
- gdp_trend
- macro_regime
