from types import SimpleNamespace

import json
import pandas as pd

import experiments.robustness_confirmation as rc


def test_empty_validation_plateau_is_recorded_without_nan_conversion(monkeypatch, tmp_path):
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    prices = pd.DataFrame({"close": range(100, 112)}, index=dates)

    class FakeEvaluator:
        def __init__(self, *args):
            pass

        def windows(self, index):
            return [
                SimpleNamespace(
                    train_start=dates[0],
                    train_end=dates[5],
                    validation_start=dates[6],
                    validation_end=dates[11],
                )
            ]

    class FakeResult:
        metrics = {"sharpe": None, "total_return": None, "max_drawdown": None}

    monkeypatch.setattr(rc, "WalkForwardEvaluator", FakeEvaluator)
    monkeypatch.setattr(rc, "run_candidate", lambda *args, **kwargs: FakeResult())

    chosen, summary = rc.plateau_study(prices, tmp_path)

    assert chosen is None
    assert summary["selection_status"] == "no_valid_plateau"
    assert summary["window_selection"] == [
        {"window": 1, "plateau": [], "plateau_center": None, "status": "no_finite_validation_sharpe"}
    ]

    selection = pd.read_csv(tmp_path / "validation_plateau_selection.csv")
    assert pd.isna(selection.loc[0, "plateau_center"])
    assert selection.loc[0, "status"] == "no_finite_validation_sharpe"

    payload = json.loads((tmp_path / "validation_plateau.json").read_text())
    assert payload["confirmation_lookback"] is None
    assert payload["selection_status"] == "no_valid_plateau"


def test_empty_plateau_is_not_promoted_to_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "CONFIRMATION_START", "2025-01-02")
    monkeypatch.setattr(rc, "CONFIRMATION_END", "2026-06-30")

    output = rc.write_unconfirmed_confirmation(tmp_path, "plateau selection not confirmed: incomplete_plateau")

    assert output["status"] == "not_confirmed"
    assert output["frozen_lookback"] is None
    assert output["metrics"] == {}
    assert output["reason"] == "plateau selection not confirmed: incomplete_plateau"
