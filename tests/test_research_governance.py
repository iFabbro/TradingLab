import json

from src.research_governance import evaluate_acceptance, protocol_fingerprint


def _report(total_returns):
    return {
        "provenance": {
            "git_commit": "abc123",
            "dataset_sha256": "dataset",
            "frictions": {"transaction_cost_bps": 5.0, "slippage_bps": 2.0},
        },
        "test_metrics": [{"total_return": value} for value in total_returns],
        "robustness": {
            "oos_daily_block_bootstrap": {
                "total_return": {"estimate": 0.10, "lower": -0.01, "upper": 0.20},
                "sharpe": {"estimate": 1.0, "lower": 0.1, "upper": 1.8},
            },
            "oos_confirmation_psr": {"probability": 0.97},
            "validation_dsr": [{"window": 1, "probability": 0.96}],
            "multiple_testing": [{"window": 1, "candidate_count": 4}],
        },
    }


def test_protocol_fingerprint_is_versioned_and_sha256_backed():
    fingerprint = protocol_fingerprint()
    assert fingerprint["protocol_version"] == "1.0"
    assert len(fingerprint["protocol_sha256"]) == 64


def test_acceptance_candidate_requires_positive_oos_and_multiple_positive_windows():
    result = evaluate_acceptance(_report([0.04, 0.03, -0.01]), _report([])["provenance"])
    assert result["decision"] == "CANDIDATE"
    assert result["reasons"] == []


def test_acceptance_fails_negative_oos():
    report = _report([-0.02, 0.01, -0.01])
    report["robustness"]["oos_daily_block_bootstrap"]["total_return"]["estimate"] = -0.01
    result = evaluate_acceptance(report, report["provenance"])
    assert result["decision"] == "FAILED"
    assert "aggregate net OOS performance is not positive" in result["reasons"]


def test_acceptance_rejects_non_protocol_frictions():
    report = _report([0.04, 0.03])
    report["provenance"]["frictions"]["transaction_cost_bps"] = 0.0
    result = evaluate_acceptance(report, report["provenance"])
    assert result["decision"] == "FAILED"
    assert any("5 bps" in reason for reason in result["reasons"])
