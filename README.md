# TradingLab

TradingLab is a modular quantitative trading lab for the full research-to-execution workflow: data ingestion, strategy generation, backtesting, risk analysis, regime detection, multi-factor modeling, optimization, portfolio construction, trade setup generation, Monte Carlo simulation, drawdown analysis, macro-based strategy, and alpha edge detection.

The project is organized as a lightweight Python micro-platform with reusable modules, CLI-driven scripts, and a roadmap-based development flow.

## Project Status

TradingLab is actively developed around the main quantitative research workflow.

TradingLab includes modules for:
- data ingestion and setup;
- strategy generation and validation;
- backtesting and performance analysis;
- risk and reward management;
- market regime detection;
- multi-factor strategy logic;
- strategy optimization;
- portfolio construction;
- trade setup generation;
- Monte Carlo simulation;
- drawdown analysis;
- macro-based strategy mapping;
- alpha edge detection.

The codebase is designed as a compact framework for systematic research, repeatable analysis, and controlled execution workflows.

## Tech Stack

- Python 3.10+.
- Pytest.
- CLI-based workflow scripts.
- Modular source layout under `src/`.

## Repository Structure

```text
src/
  data.py
  strategies.py
  backtest.py
  risk.py
  regime.py
  factors.py
  optimization.py
  portfolio.py
  signals.py
  montecarlo.py
  drawdown.py
  macro.py
  alpha.py

scripts/
  download_data.py
  run_backtest.py
  analyze_risk.py
  generate_trades.py
  run_montecarlo.py
  run_macro.py
  run_alpha.py

tests/
  test_*.py

config/
notebooks/
reports/
data/
```

## Key Features

- Download and cache market data.
- Generate and evaluate trading strategies.
- Run backtests and compute performance metrics.
- Analyze risk, drawdowns, and trade statistics.
- Detect market regimes and adapt logic accordingly.
- Build and score multi-factor strategies.
- Optimize strategy parameters.
- Construct diversified portfolios.
- Generate high-probability trade setups.
- Simulate outcomes with Monte Carlo methods.
- Analyze macro regimes and map them to exposure profiles.
- Detect alpha edges and convert them into actionable playbooks.

## Quick Start

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/iFabbro/TradingLab.git
cd TradingLab
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run a sample workflow:

```bash
python scripts/download_data.py --ticker BTC-USD --start 2016-01-01 --timeframe 1d
python scripts/run_backtest.py --strategy-id ma_crossover --ticker BTC-USD --start 2016-01-01 --end 2026-01-01 --timeframe 1d --capital 10000
python scripts/analyze_risk.py --strategy-id ma_crossover --ticker BTC-USD
python scripts/run_montecarlo.py --strategy-id ma_crossover --ticker BTC-USD
python scripts/generate_trades.py --market crypto --ticker BTC-USD
```

## Usage

TradingLab is intended for iterative quantitative research. Start by downloading data, then run a backtest, inspect risk metrics, and refine strategy logic through the available scripts and modules.

For example, a typical workflow is:
1. download historical data;
2. validate a strategy with backtesting;
3. inspect risk and drawdown behavior;
4. test robustness with Monte Carlo analysis;
5. generate trade setups or macro-aware signals.

## Testing

Run the test suite with:

```bash
pytest -v
```

## Roadmap

The repository follows a roadmap-driven development model, where each major capability is implemented as an independent module and tested in isolation.

## Contributing

Contributions are welcome. If you want to suggest an improvement, open an issue or submit a pull request.

Before contributing, please make sure:
- the change is focused;
- tests still pass;
- the README and docs stay consistent with the codebase.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.