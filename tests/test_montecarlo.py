import pytest
import numpy as np
from src.montecarlo import run_montecarlo, montecarlo_summary

RETURNS = [0.02, -0.01, 0.03, 0.01, -0.02, 0.04]

def test_run_montecarlo_shape():
    sims = run_montecarlo(RETURNS, n_simulations=500, seed=42)
    assert sims.shape == (500,)

def test_run_montecarlo_positive_equity():
    sims = run_montecarlo(RETURNS, n_simulations=200, seed=1)
    assert np.all(sims > 0)

def test_montecarlo_summary_keys():
    summary = montecarlo_summary(RETURNS, n_simulations=200)
    for k in ["n_simulations", "mean_final_equity", "std_final_equity", "prob_profit", "prob_loss_50pct", "p5", "p50", "p95"]:
        assert k in summary

def test_prob_profit_range():
    summary = montecarlo_summary(RETURNS, n_simulations=500, seed=42)
    assert 0.0 <= summary["prob_profit"] <= 1.0


def test_prob_loss_50pct_range():
    summary = montecarlo_summary(RETURNS, n_simulations=500, seed=42)
    assert 0.0 <= summary["prob_loss_50pct"] <= 1.0


def test_montecarlo_summary_shows_loss_warning_flag():
    summary = montecarlo_summary(RETURNS, n_simulations=500, seed=42)
    assert "warning_high_loss_prob" in summary


def test_montecarlo_summary_warns_on_high_loss_probability():
    summary = montecarlo_summary([-0.4, -0.3, -0.2, -0.1], n_simulations=500, seed=42)
    assert summary["warning_high_loss_prob"] is True
