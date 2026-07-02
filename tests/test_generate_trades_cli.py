import pandas as pd
import subprocess
import sys
from pathlib import Path


def test_generate_trades_cli_runs(tmp_path):
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            "volume": [1000, 1100, 1050, 1200, 1300, 1250, 1400, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850],
        }
    )
    csv_path = tmp_path / "ohlcv.csv"
    df.to_csv(csv_path, index=False)

    result = subprocess.run(
        [sys.executable, "scripts/generate_trades.py", "--file", str(csv_path), "--ticker", "TEST", "--direction", "long"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ticker: TEST" in result.stdout
    assert "entry:" in result.stdout
    assert "risk_reward:" in result.stdout

def test_generate_trades_cli_writes_valid_open_trades_csv(tmp_path):
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            "volume": [1000, 1100, 1050, 1200, 1300, 1250, 1400, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850],
        }
    )
    csv_path = tmp_path / "ohlcv.csv"
    out_path = tmp_path / "open_trades.csv"
    df.to_csv(csv_path, index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_trades.py",
            "--file",
            str(csv_path),
            "--ticker",
            "TEST",
            "--direction",
            "long",
            "--output",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    saved = pd.read_csv(out_path)
    assert list(saved.columns) == [
        "ticker",
        "side",
        "entry_price",
        "stop_price",
        "target_price",
        "risk_reward",
        "strategy_tag",
        "regime",
        "note",
        "status",
        "entry_date",
        "position_size",
    ]
    assert len(saved) == 1
    assert saved.loc[0, "ticker"] == "TEST"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_trades.py",
            "--file",
            str(csv_path),
            "--ticker",
            "TEST2",
            "--direction",
            "long",
            "--output",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    saved = pd.read_csv(out_path)
    assert len(saved) == 2
    assert list(saved["ticker"]) == ["TEST", "TEST2"]

