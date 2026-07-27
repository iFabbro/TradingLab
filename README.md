# TradingLab

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue.svg">
  <img alt="Tests" src="https://img.shields.io/badge/Tests-pytest-brightgreen.svg">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <img alt="Status" src="https://img.shields.io/badge/Status-active%20development-orange.svg">
</p>

TradingLab is a modular quantitative trading lab focused on repeatable research, controlled backtesting, risk checks, and supervised execution.

## 🚀 What it is

TradingLab is a terminal-first Python codebase for building and validating trading workflows in a structured way. It is designed to keep research, risk management, execution logic, and runtime state clearly separated so the system stays understandable as it grows.

## ✅ What it does now

TradingLab currently supports:

- loading and testing execution logic;
- building order intents from signals and risk checks;
- blocking execution when the risk layer disallows a trade;
- persisting bot state across runs;
- keeping runtime state out of version control;
- validating the main execution path with automated tests.

The project is intentionally practical: it is built for supervised use, not for pretending to be a fully autonomous trading system.

## ⚙️ Core workflow

The current execution flow is simple:

1. load a risk summary;
2. build an order intent;
3. block the order if the risk check fails;
4. persist the latest bot state;
5. keep local runtime files separate from Git.

## 🧱 Repository structure

```text
src/
  execution.py
  bot_state.py
  ...

scripts/
  run_bot.py
  ...

tests/
  test_execution.py

config/
  bot_state.example.json
  bot_state.json  # ignored by Git

ROADMAP.md
README.md
```

## 🏁 Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## 🗂️ Runtime state

The bot uses a local state file for operational continuity.

```bash
cp config/bot_state.example.json config/bot_state.json
```

- `config/bot_state.example.json` is versioned and documents the expected structure.
- `config/bot_state.json` is the local runtime file and is ignored by Git.
- If the schema changes, update the example file first.

## 🧪 Testing

Run the test suite with:

```bash
pytest -q
```

The tests cover order building, kill-switch blocking, paper-live behavior, risk loading, and bot state persistence.

## 🧭 Project status

TradingLab is under active development with a strong focus on reliability, clarity, and operational safety. The current priority is to keep the execution and state layers robust before expanding into more advanced research capabilities.

## 🗺️ Roadmap

`ROADMAP.md` contains the broader development plan and future phases. The README is deliberately narrower and describes the current working surface of the project.

## 🤝 Contributing

Contributions are welcome. Keep changes focused, keep tests passing, and update the documentation when the runtime flow changes.

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
