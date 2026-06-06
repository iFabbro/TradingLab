# TradingLab

Modular quantitative trading lab — 12-module architecture.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Structure
data/ → raw, cache, backtests, macro
src/ → core modules (data, backtest, strategies, risk, ...)
scripts/ → CLI entry points
tests/ → pytest test suite
config/ → YAML configs (not versioned)
notebooks/ → exploratory analysis
reports/ → output reports

text

## Roadmap

See [ROADMAP.md](ROADMAP.md).
