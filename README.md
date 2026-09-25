# TradingLab

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue.svg">
  <img alt="Tests" src="https://img.shields.io/badge/Tests-pytest-brightgreen.svg">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <img alt="Status" src="https://img.shields.io/badge/Status-active%20development-orange.svg">
</p>

A modular Python laboratory for quantitative research, controlled backtesting, risk gating and supervised execution workflows.

> **Status:** active development. TradingLab is a research and supervised-execution project, not a production trading system and not a promise of profitable trading.

## Why this project

Trading systems become difficult to reason about when research, risk, execution and runtime state are mixed together. TradingLab explores a more explicit separation of those concerns so that each stage can be tested independently.

The project is intentionally conservative about execution: the current public implementation focuses on order intent, safety checks, paper/simulated flows and persistent local state rather than presenting autonomous live trading as a finished capability.

## Architecture

```text
Market / research data
        │
        ▼
  Research & signals
        │
        ▼
   Risk / safety gate ──────► BLOCK
        │
        ▼
   Order intent
        │
        ├── dry-run / paper flow
        │
        ▼
 Supervised execution layer
        │
        ▼
 Persistent bot state + report
```

Core responsibilities are separated across modules for research/backtesting, portfolio and risk logic, execution, safety policy, reporting and runtime state.

## Current capabilities

- build order intents from a signal and risk decision;
- block an order when the risk layer rejects it;
- apply safety controls such as kill-switch, exposure and position-size limits;
- support dry-run and paper-live execution paths;
- persist operational bot state between runs;
- run quantitative backtests and supporting research workflows;
- calculate portfolio/risk-related analytics and reports;
- validate the main execution path with automated tests.

These are software capabilities of the current codebase. They should not be interpreted as evidence of trading profitability or production readiness.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Safe demo

The repository includes a deterministic demo that exercises the order-building and risk-gating path without connecting to a broker or submitting a real order:

```bash
python scripts/demo.py
```

Example output is intentionally simple:

```text
TradingLab safe demo
---------------------
Allowed order : {... 'status': 'pending'}
Blocked order : {... 'status': 'blocked'}
No broker connection or real order submission is performed.
```

## Execution example

The command-line runner supports explicit safety flags and local state:

```bash
python scripts/run_bot.py \
  --ticker DEMO \
  --side long \
  --position-size 2 \
  --dry-run \
  --kill-switch
```

For paper-live examples, use the documented test fixtures and configuration files. Do not use real credentials in the repository.

## Runtime state

TradingLab keeps local operational state outside version control.

```bash
cp config/bot_state.example.json config/bot_state.json
```

- `config/bot_state.example.json` documents the expected schema and is versioned.
- `config/bot_state.json` is local runtime state and is ignored by Git.
- local configuration and secrets are excluded by `.gitignore`.

## Testing & CI

Run the local suite with:

```bash
pytest -q
```

GitHub Actions runs the test suite with coverage on pushes to `main` and pull requests, and executes the safe portfolio demo as part of CI.

## Repository structure

```text
src/                    Core research, risk, execution and state modules
scripts/                CLI runners, research utilities and safe demo
tests/                  Automated tests
config/                 Versioned example configuration
 data/                  Small versioned fixtures / sample data
docs/                   Project and data documentation
.github/                Issue, PR and CI configuration
```

## Project status

The project is under active development. Current work prioritizes reliability, explicit safety boundaries, testability and clear separation between research and execution before adding broader autonomous capabilities.

See [`ROADMAP.md`](ROADMAP.md) for planned work and [`CHANGELOG.md`](CHANGELOG.md) for project history.

## Security

Never commit API keys, broker credentials, tokens, passwords, private keys or local runtime state. See [`SECURITY.md`](SECURITY.md) for the reporting and secret-handling policy.

## License

MIT — see [`LICENSE`](LICENSE).
