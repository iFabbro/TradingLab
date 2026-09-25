from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "docs" / "RESEARCH_PROTOCOL.md"
PROTOCOL_VERSION_RE = re.compile(r"\*\*Version:\*\*\s*([^\s]+)")

REGISTRY_FIELDS = [
    "run_id", "timestamp_utc", "git_commit", "protocol_version", "protocol_sha256",
    "dataset_sha256", "strategy", "symbol", "requested_start", "requested_end",
    "provider", "interval", "windows", "nominal_candidate_trials", "oos_observations",
    "oos_total_return", "oos_sharpe", "oos_psr", "decision", "reasons_json", "warnings_json",
]


def protocol_fingerprint(path: Path = PROTOCOL_PATH) -> dict[str, str]:
    data = path.read_bytes()
    text = data.decode("utf-8")
    match = PROTOCOL_VERSION_RE.search(text)
    if not match:
        raise ValueError("RESEARCH_PROTOCOL.md does not declare a version")
    return {"protocol_version": match.group(1), "protocol_sha256": hashlib.sha256(data).hexdigest()}


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def evaluate_acceptance(report: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    hard_fail = False

    provenance = report.get("provenance", metadata)
    frictions = provenance.get("frictions", {})
    if _finite(frictions.get("transaction_cost_bps")) != 5.0:
        reasons.append("transaction_cost_bps must equal the protocol baseline of 5")
        hard_fail = True
    if _finite(frictions.get("slippage_bps")) != 2.0:
        reasons.append("slippage_bps must equal the protocol baseline of 2")
        hard_fail = True
    if not provenance.get("dataset_sha256"):
        reasons.append("dataset provenance hash is missing")
        hard_fail = True
    if not provenance.get("git_commit"):
        warnings.append("git commit provenance is unavailable")

    test_metrics = report.get("test_metrics", [])
    if isinstance(test_metrics, dict):
        test_metrics = [test_metrics]
    returns = [_finite(row.get("total_return")) for row in test_metrics if isinstance(row, dict)]
    returns = [x for x in returns if x is not None]
    if not returns:
        reasons.append("no finite OOS total-return metrics are available")
        hard_fail = True
    else:
        oos_return = _finite(
            report.get("robustness", {})
            .get("oos_daily_block_bootstrap", {})
            .get("total_return", {})
            .get("estimate")
        )
        if oos_return is None:
            reasons.append("aggregate OOS total-return estimate is missing")
            hard_fail = True
            oos_return = sum(returns)
        if oos_return <= 0:
            reasons.append("aggregate net OOS performance is not positive")
            hard_fail = True

        positive_windows = sum(x > 0 for x in returns)
        if len(returns) > 1 and positive_windows < 2:
            reasons.append("OOS result is positive in fewer than two windows")
            hard_fail = True

    robustness = report.get("robustness", {})
    if not robustness.get("oos_daily_block_bootstrap"):
        reasons.append("OOS bootstrap diagnostics are missing")
        hard_fail = True
    if not robustness.get("oos_confirmation_psr"):
        reasons.append("OOS PSR diagnostic is missing")
        hard_fail = True
    if not robustness.get("validation_dsr"):
        reasons.append("validation DSR diagnostics are missing")
        hard_fail = True
    if not robustness.get("multiple_testing"):
        reasons.append("multiple-testing diagnostics are missing")
        hard_fail = True

    sharpe_ci = robustness.get("oos_daily_block_bootstrap", {}).get("sharpe", {})
    if isinstance(sharpe_ci, dict) and _finite(sharpe_ci.get("lower")) is not None and sharpe_ci["lower"] <= 0:
        warnings.append("OOS Sharpe bootstrap confidence interval includes zero")

    psr = robustness.get("oos_confirmation_psr", {})
    if isinstance(psr, dict) and _finite(psr.get("probability")) is not None and psr["probability"] < 0.95:
        warnings.append("OOS PSR probability is below 0.95")

    decision = "FAILED" if hard_fail else "CANDIDATE"
    return {
        "decision": decision,
        "reasons": reasons,
        "warnings": warnings,
        "rules_version": "RESEARCH_PROTOCOL.md v1.0",
    }


def append_registry(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({
            field: json.dumps(row[field], sort_keys=True) if field in {"reasons_json", "warnings_json"} else row.get(field, "")
            for field in REGISTRY_FIELDS
        })
