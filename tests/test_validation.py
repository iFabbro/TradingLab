import pandas as pd
import pytest

from src.validation import (
    validate_equity_curve,
    validate_metrics,
    validate_risk_summary,
    validate_trade_log,
)


def test_validate_trade_log_accepts_valid_dataframe():
    df = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01",
                "symbol": "AAPL",
                "side": "long",
                "quantity": 1.0,
                "entry_price": 100.0,
                "exit_price": 105.0,
                "pnl": 5.0,
                "return_pct": 0.05,
                "pnl_realized": 5.0,
                "pnl_unrealized": 0.0,
            }
        ]
    )

    result = validate_trade_log(df)

    assert not result.empty
    assert "pnl_realized" in result.columns
    assert "pnl_unrealized" in result.columns


def test_validate_trade_log_rejects_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="trade_log is empty"):
        validate_trade_log(df)


def test_validate_trade_log_rejects_missing_columns():
    df = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01",
                "symbol": "AAPL",
                "side": "long",
                "quantity": 1.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="missing required columns"):
        validate_trade_log(df)


def test_validate_trade_log_rejects_non_positive_quantity():
    df = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01",
                "symbol": "AAPL",
                "side": "long",
                "quantity": 0.0,
                "entry_price": 100.0,
                "exit_price": 105.0,
                "pnl": 5.0,
                "return_pct": 0.05,
                "pnl_realized": 5.0,
                "pnl_unrealized": 0.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="non-positive quantity"):
        validate_trade_log(df)


def test_validate_metrics_accepts_valid_dataframe():
    df = pd.DataFrame(
        [
            {
                "n_trades": 1,
                "total_return": 0.05,
                "win_rate": 1.0,
            }
        ]
    )

    result = validate_metrics(df)

    assert result.loc[0, "n_trades"] == 1


def test_validate_metrics_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="metrics is empty"):
        validate_metrics(pd.DataFrame())


def test_validate_metrics_rejects_multiple_rows():
    df = pd.DataFrame(
        [
            {"n_trades": 1, "total_return": 0.05, "win_rate": 1.0},
            {"n_trades": 2, "total_return": 0.07, "win_rate": 0.5},
        ]
    )

    with pytest.raises(ValueError, match="exactly one row"):
        validate_metrics(df)


def test_validate_risk_summary_accepts_valid_dataframe():
    df = pd.DataFrame([{"n_trades": 1}])

    result = validate_risk_summary(df)

    assert result.loc[0, "n_trades"] == 1


def test_validate_risk_summary_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="risk_summary is empty"):
        validate_risk_summary(pd.DataFrame())


def test_validate_equity_curve_accepts_valid_dataframe():
    df = pd.DataFrame([{"equity": 10000.0}, {"equity": 10050.0}])

    result = validate_equity_curve(df)

    assert result["equity"].iloc[-1] == 10050.0


def test_validate_equity_curve_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="equity_curve is empty"):
        validate_equity_curve(pd.DataFrame())


def test_validate_equity_curve_rejects_missing_equity_column():
    df = pd.DataFrame([{"value": 10000.0}])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_equity_curve(df)
